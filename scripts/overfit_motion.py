"""C3 acceptance: overfit the motion model on a handful of real episodes (TASKS.md C3, gate G1).

If the model, loss and sampler are correct, a full-size network trained on ~10 motions should reproduce them:
given an episode's start and end, the generated middle should match that episode closely. This script trains,
evaluates every --eval-every steps, and writes to --out:

    log.jsonl        one line per evaluation (losses and errors in physical units)
    report.json      final summary, incl. per-episode errors and the exact-endpoint check
    checkpoint.pt    model weights + config + duration stats (needed to decode generated durations)

Evaluation (endpoints given exactly, no endpoint noise): for every training episode, sample one motion from its
start + goal and measure it against (a) that episode and (b) the closest training episode. (b) matters because
episodes from the same start scene can have nearly the same end; then generating the "other" one is a correct
answer, not an overfitting failure. --eval-goals picks the goal modes evaluated: "full" (start + full end given) and/or
"objects" (start + the objects' end given, robot end free: the mode used to generate downstream demos). Keys of
non-full modes are prefixed with the mode, e.g. "objects_own_obj_pos_mm".

Training setup: --goals full (default) trains only the full-goal mode, so an exact answer exists and the loss can
approach 0; --goals all uses the dataset's goal-mode mix (none 0.1 / objects 0.3 / full 0.6). No endpoint noise,
AdamW with linear warm-up then constant learning rate, gradient clipping 1.0, no EMA. The real training recipe
(noise, cosine schedule, EMA, bf16, checkpoint/resume) is task C4.

usage (CHTC GPU job, see chtc/overfit.sub; or a quick local smoke test with --size tiny --steps 20):
    python scripts/overfit_motion.py --episodes data/episodes/StackCube-v1/dev --out runs/overfit
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from gaussbridge.model.dataset import GOAL_FULL, MotionDataset
from gaussbridge.model.dit import MotionDiT
from gaussbridge.model.evaluate import motion_errors
from gaussbridge.model.flow import flow_loss, model_inputs, sample
from gaussbridge.trajectory import Episode, known_mask

SIZES = {"full": {}, "tiny": {"d_model": 64, "depth": 2, "heads": 4}}  # full = CLAUDE.md §8.1 defaults


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episodes", required=True, type=Path, help="directory of Episode .npz files")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--size", choices=sorted(SIZES), default="full")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--sample-steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--goals", choices=["full", "all"], default="full", help="goal modes used in training")
    p.add_argument("--eval-goals", default=None, help="comma list of full,objects (default: full; full,objects "
                   "when --goals all)")
    args = p.parse_args()
    args.eval_goals = (args.eval_goals or ("full,objects" if args.goals == "all" else "full")).split(",")
    assert set(args.eval_goals) <= {"full", "objects"}, args.eval_goals
    return args


@torch.no_grad()
def evaluate(model, ds, device, sample_steps, seed, goal="full"):
    """Sample one motion per training episode in goal mode `goal`; errors vs itself and vs the closest training
    episode."""
    model.eval()
    n = len(ds)
    batch = torch.utils.data.default_collate([ds[i] for i in range(n)])
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["known"] = known_mask(batch["ent_mask"], goal)
    batch["x_known"] = torch.where(batch["known"][..., None], batch["x0"], torch.zeros_like(batch["x0"]))
    x, dur = sample(model, **model_inputs(batch), steps=sample_steps,
                    generator=torch.Generator(device).manual_seed(seed))
    x, dur = x.float().cpu(), dur.float().cpu()
    x0, dur0 = batch["x0"].cpu(), batch["log_dur"].cpu()
    known = batch["known"].cpu()
    endpoint_max_diff = (x[known] - x0[known]).abs().max().item()  # must be exactly 0 (inpainting)

    own = motion_errors(x, dur, x0, dur0, ds.n_objects, ds.dur_stats)
    # every sample against every training episode -> pick the closest by object position error
    all_err = motion_errors(x.repeat_interleave(n, 0), dur.repeat_interleave(n, 0), x0.repeat(n, 1, 1, 1),
                            dur0.repeat(n), ds.n_objects, ds.dur_stats)
    pos = all_err["obj_pos_mm"].view(n, n) + 100 * all_err["joint_rad"].view(n, n)  # 1 rad ~ 100 mm for ranking
    nearest = pos.argmin(1)
    near = {k: v.view(n, n)[torch.arange(n), nearest] for k, v in all_err.items()}
    model.train()
    summary = {f"own_{k}": v.mean().item() for k, v in own.items()}
    summary |= {f"nearest_{k}": v.mean().item() for k, v in near.items()}
    summary |= {"own_obj_pos_mm_max": own["obj_pos_mm"].max().item(),
                "nearest_is_own": (nearest == torch.arange(n)).float().mean().item(),
                "endpoint_max_abs_diff": endpoint_max_diff}
    per_episode = [{"source": ds.meta[i].get("source"), "nearest": int(nearest[i]),
                    **{k: round(v[i].item(), 4) for k, v in own.items()}} for i in range(n)]
    return summary, per_episode


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    files = sorted(args.episodes.glob("*.npz"))[:args.num_episodes]
    if not files:
        raise SystemExit(f"no .npz episodes in {args.episodes}")
    episodes = [Episode.load(f) for f in files]
    mix = {} if args.goals == "all" else {"p_none": 0.0, "p_objects": 0.0}  # see docstring
    ds = MotionDataset(episodes, endpoint_noise=False, **mix)
    print(f"{len(ds)} episodes ({ds.obj_names}), duration stats {ds.dur_stats}, device {args.device}", flush=True)

    model = MotionDiT(**SIZES[args.size]).to(args.device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))

    with open(args.out / "log.jsonl", "w") as log:
        t0, running = time.time(), []
        for step in range(1, args.steps + 1):
            idx = torch.randint(len(ds), (args.batch,))
            batch = torch.utils.data.default_collate([ds[int(i)] for i in idx])
            assert args.goals == "all" or (batch["goal"] == GOAL_FULL).all()
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, stats = flow_loss(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step()
            sched.step()
            running.append(stats["loss"])
            if not math.isfinite(stats["loss"]):
                raise SystemExit(f"loss is {stats['loss']} at step {step}")

            if step % args.eval_every == 0 or step == args.steps:
                row = {"step": step, "loss_avg": sum(running) / len(running), "grad_norm": grad_norm,
                       "seconds": round(time.time() - t0, 1), **stats}
                running, per_episode = [], {}
                for goal in args.eval_goals:
                    summary, per_episode[goal] = evaluate(model, ds, args.device, args.sample_steps, args.seed, goal)
                    prefix = "" if goal == "full" else f"{goal}_"
                    row |= {prefix + k: v for k, v in summary.items()}
                    print(f"step {step:6d} [{goal:7s}] loss {row['loss_avg']:.4f}  joint {summary['own_joint_rad']:.4f}"
                          f" rad  obj {summary['own_obj_pos_mm']:.1f} mm (nearest {summary['nearest_obj_pos_mm']:.1f})"
                          f"  rot {summary['own_obj_rot_deg']:.2f} deg  dur {summary['own_dur_s']:.3f} s  "
                          f"endpoints {summary['endpoint_max_abs_diff']:.0e}  {row['seconds']:.0f}s", flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()

    torch.save({"model": model.state_dict(), "config": model.config, "dur_stats": ds.dur_stats,
                "obj_names": ds.obj_names, "step": args.steps, "args": {k: str(v) for k, v in vars(args).items()}},
               args.out / "checkpoint.pt")
    (args.out / "report.json").write_text(json.dumps(
        {"final": row, "per_episode": per_episode, "files": [f.name for f in files],
         "args": {k: str(v) for k, v in vars(args).items()}}, indent=1))
    print(f"done: {args.out}", flush=True)


if __name__ == "__main__":
    main()
