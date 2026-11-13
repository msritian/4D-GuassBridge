"""B4: reconstruct a task's asset model (robot links + objects + background Gaussians + EnvMap) from B2 captures.

usage: python scripts/train_recon.py --captures data/assets/StackCube-v1/captures.pt
       [--out <dir of captures>] [--results results/b4] [--iters 30000] [--color-mode lit] [--set key=value ...]
Writes to --out: scene.splats.pt, background.pt (EnvMap), <object>.points.npy (512 x [xyz, rgb], CLAUDE.md §6.4),
recon.json (config, loss history, held-out metrics, git commit). Writes to --results: <task>.recon.json (same
without the history) and <task>.recon.png (held-out policy + ring views: capture, render, labels).
"""
import argparse
import dataclasses
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gaussbridge.gs.recon import IGNORE, ReconConfig, evaluate, prepare_views, reconstruct, render_view, summarize
from gaussbridge.gs.render import surface_points


def _parse_set(items, cfg):
    fields = {f.name: f for f in dataclasses.fields(cfg)}
    for item in items:
        k, v = item.split("=", 1)
        if k not in fields:
            raise SystemExit(f"unknown config key {k!r}; known: {sorted(fields)}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            val = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, tuple):
            val = tuple(type(cur[0])(x) for x in v.split(","))
        else:
            val = type(cur)(v)
        setattr(cfg, k, val)
    return cfg


def figure(splats, env, cap, cfg, configs, device, path, ring_cams=("wide_e45_a00", "close_e60_a02")):
    cams = [c for c in cap["cameras"] if c in ("base_camera", "hand_camera") or c in ring_cams]
    views = prepare_views(cap, configs, device, cfg.ground_radius, cfg.ground_center, cams)
    pal = torch.rand(len(cap["part_names"]) + 1, 3, generator=torch.Generator().manual_seed(1))
    fig, axes = plt.subplots(len(views), 4, figsize=(12, 3 * len(views)), squeeze=False)
    part_T = cap["part_T"].float().to(device)
    for ax, v in zip(axes, views):
        W, H = v.size
        o = render_view(splats, env, part_T[v.config], v.viewmat, v.K, W, H)
        gt = v.rgb.float() / 255.0
        p = float(-10 * torch.log10(((o["rgb"] - gt) ** 2).mean()))
        fg = v.label != IGNORE
        acc = float((o["label"][fg] == v.label[fg]).float().mean()) if fg.any() else float("nan")
        for a, img, title in zip(ax, [gt, o["rgb"], pal[v.label.cpu() + 1], pal[o["label"].cpu() + 1]],
                                 [f"cfg {v.config} {v.camera} capture", f"render {p:.1f} dB", "capture parts",
                                  f"rendered parts acc {acc:.3f}"]):
            a.imshow(img.cpu().numpy())
            a.set_title(title, fontsize=8)
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--out", default=None, help="asset directory (default: next to the captures file)")
    ap.add_argument("--results", default="results/b4")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--color-mode", default=None, choices=["sh", "lit"])
    ap.add_argument("--set", nargs="*", default=[], help="ReconConfig overrides, key=value")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = ReconConfig()
    if args.iters is not None:
        cfg.iters = args.iters
    if args.color_mode is not None:
        cfg.color_mode = args.color_mode
    cfg = _parse_set(args.set, cfg)
    cap = torch.load(args.captures, weights_only=True)
    task = cap["meta"]["task"]
    out = Path(args.out) if args.out else Path(args.captures).parent
    out.mkdir(parents=True, exist_ok=True)
    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()

    splats, env, info = reconstruct(cap, cfg, args.device, log=lambda m: print(m, flush=True))  # live logs on CHTC
    heldout = [r for r, s in enumerate(cap["split"]) if s == "heldout"]
    rows = evaluate(splats, env, cap, heldout, args.device, cfg)
    summary = summarize(rows)
    print(json.dumps(summary, indent=1))

    splats.meta.update({"git": git, "captures": str(args.captures)})
    splats.save(out / "scene.splats.pt")
    env.save(out / "background.pt")
    objects = [n for n, k in zip(splats.part_names, splats.part_kinds) if k == "object"]
    for name in objects:
        np.save(out / f"{name}.points.npy", surface_points(splats, name).cpu().numpy().astype(np.float32))
    report = {"task": task, "git": git, "config": dataclasses.asdict(cfg), "num_gaussians": len(splats),
              "counts": splats.counts(), "seconds": round(info["seconds"], 1), "num_train_views": info["num_views"],
              "train_configs": info["train_configs"], "heldout_configs": heldout, "heldout": summary,
              "lights": {k: v.detach().cpu().tolist() for k, v in splats.lights.items()}, "views": rows}
    (res / f"{task}.recon.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "recon.json").write_text(json.dumps({**report, "history": info["history"]}, indent=1), encoding="utf-8")
    figure(splats, env, cap, cfg, heldout, args.device, res / f"{task}.recon.png")
    print(f"saved {out / 'scene.splats.pt'} ({len(splats)} Gaussians), objects {objects}; report {res}")


if __name__ == "__main__":
    main()
