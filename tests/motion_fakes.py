"""Synthetic motion Episodes for the C3 model tests (no simulator needed)."""
import math

import numpy as np

from gaussbridge.trajectory import Episode

HOME = np.array([0, math.pi / 8, 0, -5 * math.pi / 8, 0, 3 * math.pi / 4, -math.pi / 4, 0.04, 0.04])


def fake_episode(num_steps, obj_names=("cubeA", "cubeB"), seed=0):
    """A smooth, plausible-looking episode (no physics): arm drifts from HOME, gripper closes half way,
    the first object is lifted while 'attached', the others stay put. Enough to exercise the data pipeline."""
    rng = np.random.default_rng(seed)
    T = num_steps + 1
    s = np.linspace(0, 1, T)[:, None]
    qpos = HOME + s * np.concatenate([rng.uniform(-0.3, 0.3, 7), [0, 0]])
    qpos[T // 2:, 7:] = 0.018                                           # fingers closed on the cube
    gripper = np.where(np.arange(T) < T // 2, 1.0, -1.0)
    k = len(obj_names)
    yaw = rng.uniform(-np.pi, np.pi, k)
    quat = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], -1)
    pos = np.stack([rng.uniform(-0.1, 0.1, k), rng.uniform(-0.1, 0.1, k), np.full(k, 0.02)], -1)
    obj = np.repeat(np.concatenate([pos, quat], -1)[None], T, 0)
    obj[T // 2:, 0, 2] += np.linspace(0, 0.1, T - T // 2)               # lift object 0
    attached = np.zeros((T, k), bool)
    attached[T // 2:, 0] = True
    return Episode(robot_qpos=qpos.astype(np.float32), gripper_cmd=gripper.astype(np.float32),
                   obj_pose=obj.astype(np.float32), attached=attached, base_pose=np.zeros(7, np.float32),
                   obj_names=list(obj_names), meta={"seed": seed})
