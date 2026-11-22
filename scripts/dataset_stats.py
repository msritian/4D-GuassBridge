"""A4: dataset statistics per task -> data_stats/<env>.json (counts per split, planner success, lengths, duration and
joint/object statistics of the train split).
usage: python scripts/dataset_stats.py --episodes data/episodes/StackCube-v1 --logs logs/a4 --out data_stats/StackCube-v1.json
--logs holds the CHTC job .out files; each ends with gen_demos.py's JSON summary line {"attempts", "saved", ...}.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gaussbridge.trajectory import Episode

PLAN = {  # CLAUDE.md §6.1: seeds x refs per split
    "StackCube-v1": {"train": 12000, "val": 400, "test": 1600, "policy_real": 100},
    "PickCube-v1": {"train": 8000, "val": 400, "test": 1600, "policy_real": 100},
}

p = argparse.ArgumentParser()
p.add_argument("--episodes", required=True, type=Path)
p.add_argument("--logs", type=Path)
p.add_argument("--out", required=True, type=Path)
a = p.parse_args()
env = a.episodes.name

stats = {"env_id": env, "splits": {}}
for split, planned in PLAN[env].items():
    files = sorted((a.episodes / split).glob("[0-9]*.npz"))  # glob skips macOS "._*" files
    lens = np.array([Episode.load(f).num_steps for f in files])
    stats["splits"][split] = {"episodes": len(files), "planned": planned, "ratio": round(len(files) / planned, 3),
                              "seeds": len({f.name[:5] for f in files}),
                              "len_min": int(lens.min()), "len_mean": float(lens.mean()), "len_max": int(lens.max())}

if a.logs:  # planner attempts from the job logs (the job list names the task in each shard)
    att = saved = 0
    for f in a.logs.glob("*.out"):
        lines = [ln for ln in f.read_text().splitlines() if ln.startswith('{"attempts"')]
        if lines and f"/{env}/" in f.read_text():
            s = json.loads(lines[-1])
            att, saved = att + s["attempts"], saved + s["saved"]
    stats["planner"] = {"attempts": att, "saved": saved, "success_per_attempt": round(saved / max(att, 1), 4)}

eps = [Episode.load(f) for f in sorted((a.episodes / "train").glob("[0-9]*.npz"))]
log_dur = np.log([e.num_steps / e.control_freq for e in eps])
q = np.concatenate([e.robot_qpos for e in eps])
pos = np.concatenate([e.obj_pose[..., :3].reshape(-1, 3) for e in eps])
stats["train"] = {
    "log_dur_mu": float(log_dur.mean()), "log_dur_sigma": float(log_dur.std()),  # as model.dataset.duration_stats
    "qpos_min": q.min(0).tolist(), "qpos_max": q.max(0).tolist(), "qpos_mean": q.mean(0).tolist(),
    "qpos_std": q.std(0).tolist(), "obj_pos_min": pos.min(0).tolist(), "obj_pos_max": pos.max(0).tolist(),
}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(stats, indent=1))
print(json.dumps(stats["splits"], indent=1), stats.get("planner"))
