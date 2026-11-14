"""C4/C6 diagnosis: where does the jitter of generated motions come from? (progress.md, C4 finding 1)

Generated motions have jerk 1e4-4e5 m^2/s^6 vs 2-8 for planner motions. Candidate causes, one experiment each:

  sampler     Euler integration error: vary the number of sampling steps (20, 40, 100, 200)
  precision   bf16 autocast in the network: sample in bf16 and in fp32
  weights     raw training weights vs their EMA (the C3 overfit models have no EMA; C4 checkpoints have both)
  network     the network's own prediction error at low noise: take a REAL motion x0, noise it to x_tau, read the
              network's clean-motion guess x0_hat = x_tau - tau * v (one step, no sampling at all). If x0_hat is
              already jittery at small tau, the jitter is in the trained network (undertrained / too little data),
              not in the sampler.

Jitter measure (duration-free, easy to read): hf = RMS over steps and points of |x_n - (x_{n-1} + x_{n+1}) / 2|,
the part of a track that a straight line through its neighbours does not explain. For TCP and object tracks in mm,
for the 7 arm joints in mrad. Planner references are the target. Jerk (m^2/s^6, at reference 0's duration) also.

usage (CHTC GPU job, chtc/diagnose_jitter.sub):
    python scripts/diagnose_jitter.py --episodes dev --out diag --model overfit=run/checkpoint.pt \\
        --model c4_ema=trial/ckpt/ckpt_0003000.pt:ema --model c4_raw=trial/ckpt/ckpt_0003000.pt:model
"""
import argparse
import json
from pathlib import Path

import torch

from eval_motion import load_cases  # scripts/ is on sys.path when this file is run
from gaussbridge.eval import metrics as M
from gaussbridge.model.dit import MotionDiT
from gaussbridge.model.flow import impose, sample
from gaussbridge.sim.kinematics import PandaFK
from gaussbridge.trajectory import decode_motion, known_mask

GOAL = "full"


def load_weights(spec, device):
    """'path' (EMA if present, else model) or 'path:ema' / 'path:model' -> (MotionDiT, dur_stats)."""
    path, _, which = spec.partition(":")
    s = torch.load(path, map_location=device, weights_only=False)
    which = which or ("ema" if "ema" in s else "model")
    model = MotionDiT(**(s.get("model_config") or s["config"])).to(device)
    model.load_state_dict(s[which])
    return model.eval().requires_grad_(False), s["dur_stats"]


def hf(t):
    """[..., N, P, D] -> RMS of the second-difference residual, same units as t."""
    r = t[..., 1:-1, :, :] - 0.5 * (t[..., :-2, :, :] + t[..., 2:, :, :])
    return r.square().sum(-1).mean().sqrt().item()


def jitter(x, n_objects, fk, duration):
    """Jitter numbers of motion grids x [B, N, E, F]."""
    tr = M.motion_tracks(x, n_objects, fk)
    q = decode_motion(x, 0.0)["qpos"][..., :7]
    return {"hf_tcp_mm": 1000 * hf(tr[..., :1, :]), "hf_obj_mm": 1000 * hf(tr[..., 1:, :]),
            "hf_joint_mrad": 1000 * hf(q[..., None, :]),
            "jerk": M.mean_squared_jerk(tr, duration).mean().item()}


def cond_of(ds, idx, n, device):
    x0 = ds.x0[idx].to(device)
    ent = ds.ent_mask.to(device).expand(n, -1)
    known = known_mask(ent, GOAL)
    x_known = torch.where(known[..., None], x0[:1].expand(n, -1, -1, -1), torch.zeros_like(x0[:1]))
    return x0, x_known, known, ent, ds.shapes.to(device).expand(n, *ds.shapes.shape)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episodes", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", action="append", required=True, help="name=checkpoint[:ema|:model], repeatable")
    p.add_argument("--steps", default="20,40,100,200")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device
    ds, cases, episodes = load_cases(args.episodes)
    fk = PandaFK(episodes[0].meta.get("robot_uid", "panda_wristcam"), device=dev)
    k = ds.n_objects
    durs = {seed: float(decode_motion(ds.x0[idx[:1]], ds.log_dur[idx[:1]], ds.dur_stats)["duration"][0])
            for seed, idx in cases.items()}
    rows = []

    def record(row):
        rows.append(row)
        print(json.dumps(row), flush=True)

    # planner references: the target
    ref = [jitter(ds.x0[idx].to(dev), k, fk, durs[s]) for s, idx in cases.items()]
    record({"exp": "refs", **{m: sum(r[m] for r in ref) / len(ref) for m in ref[0]}})

    for spec in args.model:
        name, _, path = spec.partition("=")
        model, _ = load_weights(path, dev)
        # sampler steps x precision
        for steps in [int(s) for s in args.steps.split(",")]:
            for bf16 in (True, False):
                if not bf16 and steps not in (40, 200):
                    continue                                         # fp32 only where it can tell something
                per = []
                for seed, idx in cases.items():
                    _, x_known, known, ent, shapes = cond_of(ds, idx, args.samples, dev)
                    g = torch.Generator(dev).manual_seed(int(seed))
                    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=bf16 and dev == "cuda"):
                        x, _ = sample(model, x_known, known, ent, shapes, steps=steps, generator=g)
                    per.append(jitter(x.float(), k, fk, durs[seed]))
                record({"exp": "sample", "model": name, "steps": steps, "bf16": bf16,
                        **{m: sum(r[m] for r in per) / len(per) for m in per[0]}})
        # one-step clean guesses from noised real motions (fp32)
        for tau in (0.02, 0.1, 0.3, 0.6):
            per, err = [], []
            for seed, idx in cases.items():
                x0, x_known, known, ent, shapes = cond_of(ds, idx, len(idx), dev)
                x_known = torch.where(known[..., None], x0, torch.zeros_like(x0))
                g = torch.Generator(dev).manual_seed(int(seed))
                eps = torch.randn(x0.shape, generator=g, device=dev)
                x_t = impose((1 - tau) * x0 + tau * eps, x_known, known, ent)
                t = torch.full((len(idx),), tau, device=dev)
                v, _ = model(x_t, (1 - tau) * ds.log_dur[idx].to(dev), t, x_known=x_known, known=known,
                             ent_mask=ent, shapes=shapes)
                x0_hat = impose(x_t - tau * v.float(), x_known, known, ent)
                per.append(jitter(x0_hat, k, fk, durs[seed]))
                err.append(1000 * M.ade(M.motion_tracks(x0_hat, k, fk), M.motion_tracks(x0, k, fk)).mean().item())
            record({"exp": "denoise", "model": name, "tau": tau, "err_mm": sum(err) / len(err),
                    **{m: sum(r[m] for r in per) / len(per) for m in per[0]}})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    cols = ["hf_tcp_mm", "hf_obj_mm", "hf_joint_mrad", "jerk"]
    md = ["| experiment | " + " | ".join(cols) + " | ADE to x0 (mm) |", "|---" * (len(cols) + 2) + "|"]
    for r in rows:
        label = r["exp"] + (f" {r['model']}" if "model" in r else "") + \
            (f" steps={r['steps']} {'bf16' if r['bf16'] else 'fp32'}" if r["exp"] == "sample" else "") + \
            (f" tau={r['tau']}" if r["exp"] == "denoise" else "")
        md.append(f"| {label} | " + " | ".join(f"{r[c]:.3g}" for c in cols) + f" | {r.get('err_mm', float('nan')):.3g} |")
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
