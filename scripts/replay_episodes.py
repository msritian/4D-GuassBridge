"""A3/A4: replay Episodes in physics and report success and object end errors.
usage: python scripts/replay_episodes.py data/episodes/StackCube-v1/train/*.npz [--n 20] [--round-trip]
--round-trip first converts each episode to the 128-step model tensor and back (A3 acceptance).
"""
import argparse
import json
import random

import numpy as np

from gaussbridge.sim.replay import make_env, replay
from gaussbridge.trajectory import Episode, episode_to_motion, motion_to_episode

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+", help="Episode .npz files")
p.add_argument("--n", type=int, help="replay a random subset of this size")
p.add_argument("--seed", type=int, default=0, help="for the random subset")
p.add_argument("--round-trip", action="store_true")
a = p.parse_args()

files = sorted(a.files)
if a.n:
    files = sorted(random.Random(a.seed).sample(files, min(a.n, len(files))))
envs, ok, end_err, joint_err = {}, 0, [], []
for fn in files:
    ep = Episode.load(fn)
    if a.round_trip:
        m = episode_to_motion(ep)
        ep2 = motion_to_episode(m["x"], m["log_dur"], ep.obj_names, ep.base_pose, ep.meta)
        joint_err.append(float(np.abs(ep2.robot_qpos[:, :7] - ep.robot_qpos[:, :7]).max()))
        ep = ep2
    env_id = ep.meta["env_id"]
    if env_id not in envs:
        envs[env_id] = make_env(env_id)
    r = replay(ep, envs[env_id])
    ok += r["success"]
    end_err.append(float(r["obj_end_err"].max()))
    print(f"{fn}: success={r['success']} obj_end_err={1000 * end_err[-1]:.1f} mm")
summary = {"episodes": len(files), "success": ok, "max_obj_end_err_mm": round(1000 * max(end_err), 2),
           "worst": files[int(np.argmax(end_err))]}
if joint_err:
    summary["max_joint_path_err_rad"] = round(max(joint_err), 4)
print(json.dumps(summary))
