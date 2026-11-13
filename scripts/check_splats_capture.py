"""B3 check on real captures: back-project one asset configuration into per-part Gaussians, then pose them with
OTHER configurations' part poses and render the policy cameras. No optimisation: this tests frames, composition
and the renderer end to end on ManiSkill data (geometry must line up; colours are only a rough init).

usage: python scripts/check_splats_capture.py --captures data/assets/StackCube-v1/captures.pt --out results/b3
       [--init-config 0] [--max-points 20000] [--color-mode sh] [--opacity 0.9] [--device auto]
Writes <out>/<task>.splats_check.json (mask IoU, label accuracy, PSNR per config x camera) and a .png grid.
"""
import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from gaussbridge import geometry as G
from gaussbridge.gs.render import render
from gaussbridge.gs.splats import init_from_points
from gaussbridge.sim.capture import IGNORE, load_captures

MAX_GROUND_DEPTH = 3.0  # m, CLAUDE.md §7.2


def backproject_config(cap, r, max_points, seed=0):
    """Canonical-frame points, colours and part ids from every valid camera of configuration r."""
    part_T = cap["part_T"][r]
    ground = cap["part_names"].index("ground") if "ground" in cap["part_names"] else -2
    pts, cols, pids = [], [], []
    for cam in cap["cameras"].values():
        if not cam["valid"][r]:
            continue
        depth = cam["depth_mm"][r].float() / 1000.0
        seg = cam["seg"][r].long()
        ok = (seg != IGNORE) & (depth > 0) & ~((seg == ground) & (depth > MAX_GROUND_DEPTH))
        rows, colsi = ok.nonzero(as_tuple=True)
        xw = G.backproject(depth, cam["viewmat"][r], cam["K"], rows, colsi)
        pid = seg[rows, colsi]
        Tinv = G.se3_inverse(part_T[pid])
        pts.append((Tinv[:, :3, :3] @ xw[:, :, None]).squeeze(-1) + Tinv[:, :3, 3])
        cols.append(cam["rgb"][r][rows, colsi].float() / 255.0)
        pids.append(pid)
    pts, cols, pids = torch.cat(pts), torch.cat(cols), torch.cat(pids)
    g = torch.Generator().manual_seed(seed)
    keep = []
    for p in pids.unique().tolist():
        idx = (pids == p).nonzero().squeeze(1)
        keep.append(idx[torch.randperm(len(idx), generator=g)[:max_points]])
    keep = torch.cat(keep)
    return pts[keep], cols[keep], pids[keep]


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--out", default="results/b3")
    ap.add_argument("--init-config", type=int, default=0)
    ap.add_argument("--max-points", type=int, default=20000)
    ap.add_argument("--color-mode", default="sh", choices=["sh", "lit"])
    ap.add_argument("--opacity", type=float, default=0.9)
    ap.add_argument("--cameras", nargs="+", default=["base_camera", "hand_camera"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cap = load_captures(args.captures)
    task = cap["meta"]["task"]
    r0 = args.init_config
    pts, cols, pids = backproject_config(cap, r0, args.max_points)
    splats = init_from_points(pts.to(dev), cols.to(dev), pids.to(dev), cap["part_names"], cap["part_kinds"],
                              color_mode=args.color_mode, part_T=cap["part_T"][r0].to(dev),
                              init_opacity=args.opacity)
    configs = [r0] + [r for r, s in enumerate(cap["split"]) if s == "heldout"]
    cams = [c for c in args.cameras if c in cap["cameras"]]
    report = {"task": task, "init_config": r0, "num_gaussians": len(splats), "counts": splats.counts(),
              "color_mode": args.color_mode, "opacity": args.opacity, "views": [],
              "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
              "note": "no optimisation; colours are raw back-projected init, geometry must line up"}
    fig, axes = plt.subplots(len(configs) * len(cams), 4, figsize=(12, 3 * len(configs) * len(cams)), squeeze=False)
    row = 0
    for r in configs:
        for uid in cams:
            c = cap["cameras"][uid]
            gt = c["rgb"][r].float() / 255.0
            seg = c["seg"][r].long()
            bg = gt[seg == IGNORE].mean(0) if (seg == IGNORE).any() else torch.zeros(3)
            H, W = seg.shape
            with torch.no_grad():
                out = render(splats, cap["part_T"][r].to(dev), c["viewmat"][r][None].to(dev), c["K"][None].to(dev),
                             W, H, background=bg.to(dev), features=splats.part_onehot())
            rgb = out["rgb"][0].clamp(0, 1).cpu()
            fg_r, fg_g = out["alpha"][0, ..., 0].cpu() > 0.5, seg != IGNORE
            lab = out["features"][0].argmax(-1).cpu()
            both = fg_r & fg_g
            v = {"config": r, "split": cap["split"][r], "camera": uid, "valid": bool(c["valid"][r]),
                 "mask_iou": float((fg_r & fg_g).sum() / (fg_r | fg_g).sum().clamp_min(1)),
                 "label_acc": float((lab[both] == seg[both]).float().mean()) if both.any() else None,
                 "psnr": psnr(rgb, gt)}
            report["views"].append(v)
            print(json.dumps(v))
            pal = torch.rand(len(cap["part_names"]) + 1, 3, generator=torch.Generator().manual_seed(1))
            for ax, img, title in zip(axes[row], [gt, rgb, pal[seg.clamp_min(-1) + 1], pal[torch.where(fg_r, lab, -1) + 1]],
                                      ["ManiSkill", f"splats {v['psnr']:.1f} dB", "GT parts",
                                       f"rendered parts IoU {v['mask_iou']:.3f} acc {v['label_acc'] or 0:.3f}"]):
                ax.imshow(img.numpy())
                ax.set_title(f"cfg {r} ({cap['split'][r]}) {uid}\n{title}" if title == "ManiSkill" else title,
                             fontsize=8)
                ax.axis("off")
            row += 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}.splats_check.png", dpi=100)
    (out_dir / f"{task}.splats_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
