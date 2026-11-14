"""C5: motion-quality evaluation harness (CLAUDE.md §10.1). Scores motion generators on test cases, writes JSON + a
markdown table.

A test case = one start scene (seed) with its R planner reference motions. For each case and goal mode a method
generates S motions from the case's start + goal; they are compared with the references:

    goal "objects"  start + the objects' end (taken from reference 0; for v1 tasks all references share it)
    goal "full"     start + reference 0's complete end (robot included)

Methods available now:
    refs      the references themselves as "samples" (sanity check: coverage = precision = 1, minADE = 0; its APD is
              the diversity of the real data, the target for a good model)
    straight  the straight-line baseline (eval/baselines.py), one deterministic sample
A trained model plugs in as one more method returning [S, N, E, F] grids per case (task C4/C7).

usage:
    python scripts/eval_motion.py --episodes data/episodes/StackCube-v1/dev --out results/c5/StackCube-v1_dev
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from gaussbridge.eval import metrics as M
from gaussbridge.eval.baselines import straight_line
from gaussbridge.model.dataset import MotionDataset
from gaussbridge.sim.kinematics import PandaFK
from gaussbridge.trajectory import Episode, decode_motion, episode_files, known_mask

GOALS = ("objects", "full")


def load_cases(episode_dir):
    """Episodes grouped by start seed -> (dataset over all episodes, {seed: [indices]})."""
    files = episode_files(episode_dir)
    if not files:
        raise SystemExit(f"no .npz episodes in {episode_dir}")
    episodes = [Episode.load(f) for f in files]
    ds = MotionDataset(episodes, endpoint_noise=False)
    cases = defaultdict(list)
    for i, ep in enumerate(episodes):
        cases[ep.meta["seed"]].append(i)
    return ds, dict(sorted(cases.items())), episodes


def condition(ds, idx, goal):
    """Conditioning of a case: the start and goal of its reference 0, as a batch of one."""
    x0 = ds.x0[idx[0]][None]
    ent = ds.ent_mask[None]
    known = known_mask(ent, goal)
    return torch.where(known[..., None], x0, torch.zeros_like(x0)), known, ent


def method_refs(ds, idx, goal):
    return ds.x0[idx], ds.log_dur[idx]


def method_straight(ds, idx, goal):
    x_known, known, ent = condition(ds, idx, goal)
    return straight_line(x_known, known, ent), ds.log_dur[idx[:1]]  # duration: reference 0's (it has none)


METHODS = {"refs": method_refs, "straight": method_straight}


def evaluate(ds, cases, episodes, methods, fk):
    """{method: {goal: {"per_case": [...], "mean": {...}, "std": {...}}}}"""
    results = {}
    for name in methods:
        results[name] = {}
        for goal in GOALS:
            rows = []
            for seed, idx in cases.items():
                x, log_dur = METHODS[name](ds, idx, goal)
                dur = decode_motion(x, log_dur, ds.dur_stats)["duration"]
                samples = M.motion_tracks(x, ds.n_objects, fk)
                refs = M.motion_tracks(ds.x0[idx], ds.n_objects, fk)
                row = {"seed": seed, "n_refs": len(idx), "n_samples": len(x),
                       **M.case_metrics(samples, refs, dur)}
                end = M.endpoint_errors(x, ds.x0[idx[:1]].expand_as(x), ds.n_objects, fk)
                row |= {"end_obj_pos_cm": end["obj_pos_cm"].mean().item(),
                        "end_obj_rot_deg": end["obj_rot_deg"].mean().item(),
                        "end_tcp_cm": end["tcp_pos_cm"].mean().item()}
                rows.append(row)
            keys = [k for k in rows[0] if k not in ("seed", "n_refs", "n_samples")]
            table = torch.tensor([[r[k] for k in keys] for r in rows], dtype=torch.float64)
            results[name][goal] = {
                "per_case": rows,
                "mean": dict(zip(keys, table.mean(0).tolist())),
                "std": dict(zip(keys, (table.std(0) if len(rows) > 1 else torch.zeros(len(keys))).tolist())),
            }
    return results


COLUMNS = [("end_obj_pos_cm", "end obj (cm)"), ("end_obj_rot_deg", "end rot (deg)"), ("end_tcp_cm", "end TCP (cm)"),
           ("min_ade_m", "minADE (cm)"), ("coverage", "coverage@3cm"), ("precision", "precision@3cm"),
           ("apd_m", "APD (cm)"), ("jerk", "jerk (m²/s⁶)")]


def markdown(results, n_cases):
    """One table per goal mode, mean ± std over cases; distances in cm."""
    out = []
    for goal in GOALS:
        out.append(f"\n**Goal: {goal}** ({n_cases} cases, mean ± std over cases)\n")
        out.append("| method | " + " | ".join(c for _, c in COLUMNS) + " |")
        out.append("|---" * (len(COLUMNS) + 1) + "|")
        for name, res in results.items():
            cells = []
            for key, _ in COLUMNS:
                scale = 100 if key.endswith("_m") else 1
                m, s = res[goal]["mean"][key] * scale, res[goal]["std"][key] * scale
                cells.append(f"{m:.2f} ± {s:.2f}")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episodes", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="output prefix: writes <out>.json and <out>.md")
    p.add_argument("--methods", default="refs,straight")
    args = p.parse_args()
    ds, cases, episodes = load_cases(args.episodes)
    fk = PandaFK(episodes[0].meta.get("robot_uid", "panda_wristcam"))
    results = evaluate(ds, cases, episodes, args.methods.split(","), fk)
    md = markdown(results, len(cases))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"episodes": str(args.episodes), "cases": {str(k): len(v) for k, v in cases.items()}, "results": results},
        indent=1))
    args.out.with_suffix(".md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
