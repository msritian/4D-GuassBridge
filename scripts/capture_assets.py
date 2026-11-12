"""B2: asset-mode captures (CLAUDE.md §7.2). Random robot/object configurations, every capture camera, one file.

usage: python scripts/capture_assets.py --task StackCube-v1 --out data/assets/StackCube-v1/captures.pt \
           [--seed 0 --n-train 8 --n-heldout 2 --sheet results/b2/StackCube-v1_sheet.png]
Runs on a laptop or a CHTC job (CPU with lavapipe or GPU): sim_backend physx_cpu, render_backend sapien_cpu.
Training configurations use rng seed [seed, 0], held-out ones [seed, 1] (disjoint streams).
Writes <out> (torch file, see capture.capture_configs) and <out without suffix>.report.json (view counts per part,
timing, git commit); with --sheet also a contact sheet (training configurations x views).
"""
import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import matplotlib

import gaussbridge.sim.envs as E
from gaussbridge.sim.capture import capture_configs, part_table, random_configs, save_captures, view_counts

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHEET_VIEWS = ["base_camera", "hand_camera", "wide_e45_a00", "wide_e45_a04", "wide_e20_a08", "close_e30_a02",
               "close_e60_a05"]
MIN_VIEWS = 20  # B2 acceptance: every visual robot link in >= 20 training views


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def contact_sheet(cap, path, n_rows=4):
    views = [v for v in SHEET_VIEWS if v in cap["cameras"]]
    rows = [r for r, s in enumerate(cap["split"]) if s == "train"][:n_rows]
    fig, axes = plt.subplots(len(rows), len(views), figsize=(2.2 * len(views), 2.3 * len(rows)), squeeze=False)
    for i, r in enumerate(rows):
        for j, v in enumerate(views):
            ax = axes[i, j]
            ax.imshow(cap["cameras"][v]["rgb"][r].numpy())
            if not cap["cameras"][v]["valid"][r]:
                ax.text(0.5, 0.5, "invalid view\n(camera inside arm)", color="red", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(v, fontsize=8)
        n_float = int(cap["floating"][r].sum())
        axes[i, 0].set_ylabel(f"config {r}\n{n_float} floating", fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="StackCube-v1", choices=sorted(E.CAPTURE_ENV))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-heldout", type=int, default=2)
    ap.add_argument("--sheet", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    configs = random_configs(args.task, args.n_train, [args.seed, 0]) + \
        random_configs(args.task, args.n_heldout, [args.seed, 1])
    splits = ["train"] * args.n_train + ["heldout"] * args.n_heldout
    env = gym.make(E.CAPTURE_ENV[args.task], obs_mode="rgb+depth+segmentation", sim_backend="physx_cpu",
                   render_backend="sapien_cpu")
    env.reset(seed=args.seed)
    parts = part_table(env)
    t0 = time.time()
    cap = capture_configs(env, configs, splits, parts)
    seconds = time.time() - t0
    env.close()

    meta = {"task": args.task, "env_id": E.CAPTURE_ENV[args.task], "robot_uid": E.DEFAULT_ROBOT[args.task],
            "seed": args.seed, "n_train": args.n_train, "n_heldout": args.n_heldout, "git": git_commit(),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "rings": [vars(r) | {"target": list(r.target), "elevations_deg": list(r.elevations_deg)} for r in E.RINGS]}
    save_captures(out, cap, meta)

    counts = view_counts(cap)
    robot_visual = [n for n, k, v in zip(cap["part_names"], cap["part_kinds"], cap["part_visual"]) if k == "robot" and v]
    worst = min(robot_visual, key=lambda n: counts[n])
    report = {"meta": meta, "file_mb": round(out.stat().st_size / 2**20, 1), "capture_seconds": round(seconds, 2),
              "n_cameras": len(cap["cameras"]), "views_per_part_train": counts,
              "worst_visual_robot_link": [worst, counts[worst]], "acceptance_min_views": MIN_VIEWS,
              "acceptance_pass": counts[worst] >= MIN_VIEWS and all(
                  counts[n] >= MIN_VIEWS for n, k in zip(cap["part_names"], cap["part_kinds"]) if k == "object"),
              "floating": cap["floating"].tolist(),
              "invalid_views": [[uid, r] for uid, c in cap["cameras"].items() for r in range(len(cap["split"]))
                                if not c["valid"][r]]}
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    if args.sheet:
        contact_sheet(cap, args.sheet)
    print(json.dumps({k: report[k] for k in ("file_mb", "capture_seconds", "n_cameras", "worst_visual_robot_link",
                                              "acceptance_pass", "invalid_views")}))


if __name__ == "__main__":
    main()
