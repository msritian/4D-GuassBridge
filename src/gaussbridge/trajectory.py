"""C1 trajectory core: the Episode on-disk format (CLAUDE.md §6.2) and the model's motion tensor (§6.3)."""
import dataclasses
import json
import math

import numpy as np
import torch

from gaussbridge import geometry as G

N_STEPS = 128           # normalised steps in model space
K_MAX = 3               # object slots; entity 0 is the robot
NUM_FEATURES = 10
CONTROL_FREQ = 20.0
Q_LO = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_HI = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
FINGER_MAX = 0.04
POS_CENTER = torch.tensor([0.0, 0.0, 0.1])
POS_SCALE = 0.3
# ponytail: expected values from §6.3; replace with the training-set stats A4 writes to data_stats/<env>.json
LOG_DUR_MU, LOG_DUR_SIGMA = math.log(8.0), 0.35


@dataclasses.dataclass
class Episode:
    robot_qpos: np.ndarray   # [L+1, 9] float32, ManiSkill joint order
    gripper_cmd: np.ndarray  # [L+1] float32, +1 open / -1 close; entry t is applied after state t (last = repeat)
    obj_pose: np.ndarray     # [L+1, K, 7] float32 pose7, order = obj_names
    attached: np.ndarray     # [L+1, K] bool
    base_pose: np.ndarray    # [7] float32 robot root pose
    obj_names: list
    meta: dict               # env_id, robot_uid, seed, ref_idx, knobs, success, source
    control_freq: float = CONTROL_FREQ

    @property
    def num_steps(self):
        return len(self.robot_qpos) - 1

    def actions(self):
        """pd_joint_pos actions [L, 8]: next arm joint targets + gripper command."""
        return np.concatenate([self.robot_qpos[1:, :7], self.gripper_cmd[:-1, None]], 1).astype(np.float32)

    def save(self, path):
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d["obj_names"], d["meta"] = json.dumps(self.obj_names), json.dumps(self.meta)
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        d["obj_names"], d["meta"] = json.loads(str(d["obj_names"])), json.loads(str(d["meta"]))
        d["control_freq"] = float(d["control_freq"])
        return cls(**d)


def resample(x, n, kind="linear"):
    """[T, ...] samples evenly spaced in time -> [n, ...]. kind: 'linear', 'slerp' (wxyz quaternions, last dim)
    or 'hold' (nearest sample; gripper / attached). First and last samples are kept exactly."""
    x = torch.as_tensor(x)
    s = torch.linspace(0, len(x) - 1, n, dtype=torch.float64)
    if kind == "hold":
        return x[s.round().long()]
    i0 = s.floor().long().clamp(max=len(x) - 2)
    w = (s - i0).to(x.dtype)
    if kind == "linear":
        out = torch.lerp(x[i0], x[i0 + 1], w.view(-1, *[1] * (x.ndim - 1)))
    else:
        out = G.slerp(x[i0], x[i0 + 1], w.view(-1, *[1] * (x.ndim - 2)))
    return torch.cat([x[:1], out[1:-1], x[-1:]])


def episode_to_motion(ep, n=N_STEPS):
    """Episode -> {"x": [n, 1+K_MAX, 10], "ent_mask": [1+K_MAX] bool, "log_dur": scalar}.
    Robot token: [q_norm(7), finger_norm, gripper_cmd, 0]; object token: [(p-c)/s (3), rot6d (6), attached*2-1]."""
    k = len(ep.obj_names)
    q = resample(ep.robot_qpos, n)
    pose = torch.as_tensor(ep.obj_pose)
    x = torch.zeros(n, 1 + K_MAX, NUM_FEATURES)
    x[:, 0, :7] = 2 * (q[:, :7] - Q_LO) / (Q_HI - Q_LO) - 1
    x[:, 0, 7] = 2 * q[:, 7] / FINGER_MAX - 1
    x[:, 0, 8] = resample(ep.gripper_cmd, n, "hold")
    x[:, 1:1 + k, :3] = (resample(pose[..., :3], n) - POS_CENTER) / POS_SCALE
    x[:, 1:1 + k, 3:9] = G.matrix_to_rot6d(G.quat_to_matrix(resample(pose[..., 3:], n, "slerp")))
    x[:, 1:1 + k, 9] = 2 * resample(ep.attached, n, "hold").float() - 1
    log_dur = (math.log(ep.num_steps / ep.control_freq) - LOG_DUR_MU) / LOG_DUR_SIGMA
    return {"x": x, "ent_mask": torch.arange(1 + K_MAX) <= k, "log_dur": torch.tensor(log_dur)}


def known_mask(ent_mask, goal="full", n=N_STEPS):
    """[..., n, E] bool of entries held fixed (inpainted). goal: 'full' = start + end of every entity,
    'objects' = robot end unknown, 'none' = start only (forward-only baseline)."""
    assert goal in ("full", "objects", "none"), goal
    known = torch.zeros(*ent_mask.shape[:-1], n, ent_mask.shape[-1], dtype=torch.bool)
    known[..., 0, :] = ent_mask
    if goal != "none":
        known[..., -1, :] = ent_mask
    if goal == "objects":
        known[..., -1, 0] = False
    return known


def decode_motion(x, log_dur):
    """Motion tensor [..., n, E, 10] -> physical quantities on the n-step grid (differentiable; for guidance).
    Object outputs cover all K_MAX slots; padded slots are meaningless."""
    r, o = x[..., 0, :], x[..., 1:, :]
    finger = (r[..., 7:8] + 1) / 2 * FINGER_MAX
    return {
        "qpos": torch.cat([(r[..., :7] + 1) / 2 * (Q_HI - Q_LO).to(x) + Q_LO.to(x), finger, finger], -1),
        "gripper": r[..., 8],                       # > 0 open
        "obj_pos": o[..., :3] * POS_SCALE + POS_CENTER.to(x),
        "obj_rot": G.rot6d_to_matrix(o[..., 3:9]),  # [..., n, K_MAX, 3, 3]
        "attached": o[..., 9],                      # > 0 attached
        "duration": torch.exp(torch.as_tensor(log_dur) * LOG_DUR_SIGMA + LOG_DUR_MU),  # seconds
    }


def motion_to_episode(x, log_dur, obj_names, base_pose, meta=None, control_freq=CONTROL_FREQ):
    """One motion tensor [n, E, 10] -> Episode at control_freq (for physics replay, rendering, DP export)."""
    d = decode_motion(x.detach().float().cpu(), torch.as_tensor(log_dur).detach().float().cpu())
    L = max(1, round(float(d["duration"]) * control_freq))
    k = len(obj_names)
    quat = G.matrix_to_quat(d["obj_rot"][:, :k])
    return Episode(
        robot_qpos=resample(d["qpos"], L + 1).numpy(),
        gripper_cmd=np.where(resample(d["gripper"], L + 1, "hold").numpy() > 0, 1.0, -1.0).astype(np.float32),
        obj_pose=torch.cat([resample(d["obj_pos"][:, :k], L + 1), resample(quat, L + 1, "slerp")], -1).numpy(),
        attached=(resample(d["attached"][:, :k], L + 1, "hold") > 0).numpy(),
        base_pose=np.asarray(base_pose, np.float32), obj_names=list(obj_names), meta=dict(meta or {}),
        control_freq=control_freq)
