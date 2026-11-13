"""A2: A1 planner shards (<prefix>.h5 / .json / .knobs.json) -> Episodes (CLAUDE.md §6.2) with the §5 attached rule."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from gaussbridge.sim.envs import OBJECTS
from gaussbridge.sim.kinematics import PandaFK
from gaussbridge.trajectory import Episode

GRASP_DIST, FINGER_OPEN = 0.02, 0.039  # m, thresholds of the §5 rule


def attached_mask(gripper_cmd, finger, obj_pos, tcp_pos):
    """gripper_cmd [T], finger joint [T], object centres [T, K, 3], TCP [T, 3] -> [T, K] bool:
    gripper commanded closed, fingers not fully open, object centre within 2 cm of the TCP."""
    near = np.linalg.norm(obj_pos - tcp_pos[:, None], axis=-1) < GRASP_DIST
    return near & ((gripper_cmd < 0) & (finger < FINGER_OPEN))[:, None]


def load_shard(prefix):
    """Every episode of one shard as an Episode, in episode_id order."""
    info = json.loads(Path(f"{prefix}.json").read_text())
    knobs = json.loads(Path(f"{prefix}.knobs.json").read_text())
    env_id = info["env_info"]["env_id"]
    names = OBJECTS[env_id]
    fk, episodes = {}, []
    with h5py.File(f"{prefix}.h5", "r") as f:
        for e in info["episodes"]:
            eid = e["episode_id"]
            t = f[f"traj_{eid}"]
            (uid, art), = t["env_states/articulations"].items()  # the one robot; key = robot uid
            art = art[:].astype(np.float32)
            qpos, base = art[:, 13:22], art[0, :7]
            g = t["actions"][:, -1].astype(np.float32)
            grip = np.append(g, g[-1])
            obj_pose = np.stack([t[f"env_states/actors/{n}"][:, :7] for n in names], 1).astype(np.float32)
            fk.setdefault(uid, PandaFK(uid))
            tcp = fk[uid].tcp_pose(torch.as_tensor(qpos), base_pose=base)[:, :3, 3].numpy()
            k = knobs[str(eid)]
            episodes.append(Episode(
                robot_qpos=qpos, gripper_cmd=grip, obj_pose=obj_pose,
                attached=attached_mask(grip, qpos[:, 7], obj_pose[..., :3], tcp), base_pose=base,
                obj_names=names,
                meta={"env_id": env_id, "robot_uid": uid, "seed": e["episode_seed"], "ref_idx": k["ref_idx"],
                      "attempt": k["attempt"], "knobs": k["knobs"], "success": bool(e["success"]),
                      "source": f"{Path(prefix).name}/traj_{eid}"}))
    return episodes
