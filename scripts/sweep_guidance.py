"""C6: guidance-scale sweep (CLAUDE.md §8.3: tune lambda in {0, 0.25, 0.5, 1, 2} on val).

For every test case (one start seed with its reference motions, as in scripts/eval_motion.py), goal mode and lambda,
draw S motions with the SAME noise (so lambda is the only difference), then measure:

    pen_mm          mean over samples of the largest excess penetration (guidance/collision.py), raw samples
    pen_rate        fraction of samples with max excess penetration > 5 mm (the §8.3 reject threshold)
    attach_mm       sqrt of the attachment energy per held step: typical step-to-step slip of a held object (mm)
    proj_ok         fraction of samples the grasp projection accepts (guidance/attach.py)
    pen_proj_mm     penetration AFTER the projection, accepted samples (what the filters will see)
    jerk, minADE, coverage@3cm/@1cm, APD   on the projected samples (eval/metrics.py), against the references
    end_obj_cm      object end error (0 when the end is given: the given cells are imposed)
    s_per_sample    sampling time per motion (GPU), including guidance

lambda = 0 is the unguided sampler bit for bit. Writes <out>.json and <out>.md.

usage (CHTC GPU job, chtc/sweep_guidance.sub):
    python scripts/sweep_guidance.py --ckpt run/checkpoint.pt --episodes dev --out sweep_stack
"""
import argparse
import json
import time
from pathlib import Path

import torch

from eval_motion import load_cases                # scripts/ is on sys.path when this file is run
from gaussbridge.eval import metrics as M
from gaussbridge.guidance.attach import project_attachment
from gaussbridge.guidance.guide import PhysicsEnergy
from gaussbridge.model.dit import MotionDiT
from gaussbridge.model.flow import sample
from gaussbridge.sim.kinematics import PandaFK
from gaussbridge.trajectory import decode_motion, known_mask

PEN_REJECT = 0.005   # m (§8.3)


def load_model(path, device):
    """A MotionDiT from an overfit checkpoint (scripts/overfit_motion.py) or a C4 final_ema.pt / ckpt_*.pt
    (EMA weights preferred). Returns (model, dur_stats)."""
    s = torch.load(path, map_location=device, weights_only=False)
    model = MotionDiT(**(s.get("model_config") or s["config"])).to(device)
    model.load_state_dict(s["ema"] if "ema" in s else s["model"])
    return model.eval().requires_grad_(False), s["dur_stats"]


def run_case(model, ds, idx, goal, lam, n_samples, steps, seed, energy, fk, dur_stats, device):
    x0 = ds.x0[idx].to(device)
    ent = ds.ent_mask.to(device).expand(n_samples, -1)
    known = known_mask(ent, goal)
    x_known = torch.where(known[..., None], x0[:1].expand(n_samples, -1, -1, -1), torch.zeros_like(x0[:1]))
    shapes = ds.shapes.to(device).expand(n_samples, *ds.shapes.shape)
    g = torch.Generator(device).manual_seed(seed)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        x, log_dur = sample(model, x_known, known, ent, shapes, steps=steps, generator=g,
                            guidance=energy if lam else None, guide_scale=lam)
    if device == "cuda":
        torch.cuda.synchronize()
    secs = (time.perf_counter() - t0) / n_samples
    x, log_dur = x.float(), log_dur.float()
    k = ds.n_objects

    pen = energy.scene.max_penetration(x)
    held_steps = (decode_motion(x, 0.0)["attached"][..., :k] > 0).float().sum((1, 2)).clamp_min(1)
    attach = (energy.terms(x)["attach"] / held_steps).sqrt()
    xp, ok, _ = project_attachment(x, known, k, fk)
    pen_p = energy.scene.max_penetration(xp)
    dur = decode_motion(xp, log_dur, dur_stats)["duration"]
    tracks, refs = M.motion_tracks(xp, k, fk), M.motion_tracks(x0, k, fk)
    row = {"pen_mm": 1000 * pen.mean().item(), "pen_rate": (pen > PEN_REJECT).float().mean().item(),
           "attach_mm": 1000 * attach.mean().item(), "proj_ok": ok.float().mean().item(),
           "pen_proj_mm": 1000 * pen_p[ok].mean().item() if ok.any() else float("nan"),
           **M.case_metrics(tracks, refs, dur),
           "coverage_1cm": M.coverage(tracks, refs, 0.01).item(),
           "end_obj_cm": M.endpoint_errors(xp, x0[:1].expand_as(xp), k, fk)["obj_pos_cm"].mean().item(),
           "s_per_sample": secs}
    return row


COLUMNS = [("pen_mm", "pen (mm)"), ("pen_rate", "pen>5mm"), ("attach_mm", "slip (mm)"), ("proj_ok", "proj ok"),
           ("pen_proj_mm", "pen proj (mm)"), ("jerk", "jerk"), ("min_ade_m", "minADE (cm)"),
           ("coverage", "cov@3cm"), ("coverage_1cm", "cov@1cm"), ("apd_m", "APD (cm)"),
           ("end_obj_cm", "end obj (cm)"), ("s_per_sample", "s/sample")]


def markdown(results, n_cases, n_samples):
    out = []
    for goal, by_lam in results.items():
        out.append(f"\n**Goal: {goal}** ({n_cases} cases x {n_samples} samples, mean over cases)\n")
        out.append("| lambda | " + " | ".join(c for _, c in COLUMNS) + " |")
        out.append("|---" * (len(COLUMNS) + 1) + "|")
        for lam, mean in by_lam.items():
            cells = []
            for key, _ in COLUMNS:
                v = mean[key] * (100 if key.endswith("_m") else 1)
                cells.append(f"{v:.3g}" if key in ("jerk", "s_per_sample") else f"{v:.2f}")
            out.append(f"| {lam} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--episodes", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="output prefix: writes <out>.json and <out>.md")
    p.add_argument("--lams", default="0,0.25,0.5,1,2")
    p.add_argument("--goals", default="objects,full")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ds, cases, episodes = load_cases(args.episodes)
    model, dur_stats = load_model(args.ckpt, args.device)
    fk = PandaFK(episodes[0].meta.get("robot_uid", "panda_wristcam"), device=args.device)
    energy = PhysicsEnergy(fk, ds.n_objects, device=args.device)
    results = {}
    for goal in args.goals.split(","):
        results[goal] = {}
        for lam in [float(v) for v in args.lams.split(",")]:
            rows = [run_case(model, ds, idx, goal, lam, args.samples, args.steps, int(seed), energy, fk, dur_stats,
                             args.device) for seed, idx in cases.items()]
            mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
            results[goal][lam] = mean
            print(json.dumps({"goal": goal, "lambda": lam, **mean}), flush=True)
    md = markdown(results, len(cases), args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({"ckpt": str(args.ckpt), "episodes": str(args.episodes),
                                                         "samples": args.samples, "steps": args.steps,
                                                         "results": results}, indent=1))
    args.out.with_suffix(".md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
