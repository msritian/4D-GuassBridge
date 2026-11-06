"""Acceptance plot for A1: gripper (TCP) xyz over time for every ref of one seed.
TCP is not stored in the h5, so set each saved state in the sim and read the TCP pose.
usage: python scripts/plot_refs.py <shard prefix> <env_id> <seed> <out.png>"""
import json
import sys

import gymnasium as gym
import h5py
import matplotlib
import mani_skill.envs  # noqa: F401
from mani_skill.trajectory.utils import dict_to_list_of_dicts

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

prefix, env_id, seed, png = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
knobs = json.load(open(f"{prefix}.knobs.json"))
env = gym.make(env_id, obs_mode="none", sim_backend="physx_cpu", render_backend="sapien_cpu")
env.reset(seed=seed)
u = env.unwrapped
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
with h5py.File(f"{prefix}.h5") as f:
    for eid, k in knobs.items():
        if k["seed"] != seed:
            continue
        tcp = []
        for s in dict_to_list_of_dicts(f[f"traj_{eid}"]["env_states"]):
            u.set_state_dict(s)
            tcp.append(u.agent.tcp.pose.p[0].numpy().copy())
        t = [i / 20 for i in range(len(tcp))]
        kn = k["knobs"]
        label = f"ref {k['ref_idx']}: yaw {kn['grasp_yaw_k']*90}°, speed {kn['speed']:.2f}, via {'y' if kn['via'] else 'n'}, rrt {'y' if kn['rrt_transport'] else 'n'}"
        for a, ax in enumerate(axes):
            ax.plot(t, [p[a] for p in tcp], label=label if a == 0 else None)
for ax, name in zip(axes, "xyz"):
    ax.set_title(f"TCP {name} (m)")
    ax.set_xlabel("time (s)")
fig.suptitle(f"{env_id}, seed {seed}: references from the same start")
fig.legend(loc="lower center", ncol=2, fontsize=8)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(png, dpi=110)
print("saved", png)
