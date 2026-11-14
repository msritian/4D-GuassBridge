"""C3/C4: compare generated motions with reference motions in physical units (for training-time monitoring).

The model works in normalised units; people think in radians, millimetres and degrees. `motion_errors` decodes
both grids (trajectory.decode_motion) and reports, per motion, the mean over all 128 steps of:

    joint_rad    |arm joint angle difference|, averaged over the 7 arm joints
    finger_mm    |finger opening difference|
    obj_pos_mm   distance between object centres, averaged over the task's objects
    obj_rot_deg  angle of the rotation between the two object orientations, averaged over objects
    dur_s        |duration difference| in seconds

These are per-motion errors against ONE reference. The full evaluation metrics (coverage, diversity, collisions,
cube symmetries, ...) are task C5 (eval/metrics.py).
"""
import torch

from gaussbridge.trajectory import decode_motion


def rotation_angle_deg(Ra, Rb):
    """Angle (degrees) of the rotation taking Rb to Ra, for [..., 3, 3] rotation matrices.
    Uses atan2(sin, cos) rather than acos(cos): acos loses precision for small angles (float32 gives ~0.02 deg
    of noise near 0), and small angles are exactly what we measure once the model is good."""
    D = Ra @ Rb.mT
    cos = (D.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    sin = torch.stack([D[..., 2, 1] - D[..., 1, 2], D[..., 0, 2] - D[..., 2, 0], D[..., 1, 0] - D[..., 0, 1]],
                      -1).norm(dim=-1) / 2
    return torch.rad2deg(torch.atan2(sin, cos))


def motion_errors(x, log_dur, x_ref, log_dur_ref, n_objects, dur_stats):
    """Errors of motions x [B, N, E, F] (log_dur [B]) against references of the same shape. Returns a dict of
    [B] tensors (see module docstring)."""
    a, b = decode_motion(x, log_dur, dur_stats), decode_motion(x_ref, log_dur_ref, dur_stats)
    k = n_objects
    return {
        "joint_rad": (a["qpos"][..., :7] - b["qpos"][..., :7]).abs().mean((-1, -2)),
        "finger_mm": 1000 * (a["qpos"][..., 7] - b["qpos"][..., 7]).abs().mean(-1),
        "obj_pos_mm": 1000 * (a["obj_pos"][..., :k, :] - b["obj_pos"][..., :k, :]).norm(dim=-1).mean((-1, -2)),
        "obj_rot_deg": rotation_angle_deg(a["obj_rot"][..., :k, :, :], b["obj_rot"][..., :k, :, :]).mean((-1, -2)),
        "dur_s": (a["duration"] - b["duration"]).abs(),
    }
