"""C1 geometry helpers (CLAUDE.md §4). Torch, any leading batch dims, differentiable except where noted.

Quaternions are (w, x, y, z). pose7 = (px, py, pz, qw, qx, qy, qz) = ManiSkill Pose.raw_pose.
T_a_b (4x4) maps frame-b coordinates to frame a. 6D rotation = first two columns of R.
Cameras use OpenCV axes (x right, y down, z forward); viewmat = world->camera 4x4 (what gsplat wants);
pixel j covers u in [j, j+1) (principal point = W/2, H/2, as SAPIEN).
"""
import itertools
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- rotations
def quat_to_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).unflatten(-1, (3, 3))


def matrix_to_quat(R):
    """Unit quaternion with w >= 0. Stable for all angles: builds q from its largest component (as pytorch3d)."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = R.flatten(-2).unbind(-1)
    q_abs = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                         1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1).clamp_min(0).sqrt()  # 2|q_i|
    cands = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1)], -2)
    cands = cands / (2 * q_abs[..., None].clamp_min(0.1))
    idx = q_abs.argmax(-1)[..., None, None].expand(*q_abs.shape[:-1], 1, 4)
    q = F.normalize(cands.gather(-2, idx).squeeze(-2), dim=-1)
    return q * (1 - 2 * (q[..., :1] < 0).to(q.dtype))


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def matrix_to_rot6d(R):
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def rot6d_to_matrix(d):
    """Gram-Schmidt (Zhou et al. 2019): any 6 numbers -> a valid rotation."""
    b1 = F.normalize(d[..., :3], dim=-1)
    a2 = d[..., 3:]
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -1)


def slerp(q0, q1, t):
    """Shortest-path interpolation of unit quaternions; t broadcasts against q0[..., 0]."""
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    theta = torch.acos(dot.abs().clamp(max=1.0))
    t = torch.as_tensor(t, dtype=q0.dtype, device=q0.device)[..., None]
    s = torch.sin(theta)
    small = s < 1e-6
    s = torch.where(small, torch.ones_like(s), s)
    w0 = torch.where(small, 1 - t, torch.sin((1 - t) * theta) / s)
    w1 = torch.where(small, t, torch.sin(t * theta) / s)
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)


def cube_symmetries(dtype=torch.float32):
    """The 24 proper rotations mapping a cube onto itself (signed permutation matrices, det = +1). [24, 3, 3]"""
    mats = [torch.eye(3, dtype=dtype)[list(p)] * torch.tensor(s, dtype=dtype)[:, None]
            for p in itertools.permutations(range(3)) for s in itertools.product([1, -1], repeat=3)]
    return torch.stack([m for m in mats if torch.det(m) > 0])


# ---------------------------------------------------------------- poses
def pose7_to_matrix(p):
    T = torch.zeros(*p.shape[:-1], 4, 4, dtype=p.dtype, device=p.device)
    T[..., :3, :3] = quat_to_matrix(p[..., 3:])
    T[..., :3, 3] = p[..., :3]
    T[..., 3, 3] = 1
    return T


def matrix_to_pose7(T):
    return torch.cat([T[..., :3, 3], matrix_to_quat(T[..., :3, :3])], -1)


def se3_inverse(T):
    Rt = T[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ T[..., :3, 3:]).squeeze(-1)
    out[..., 3, 3] = 1
    return out


def transform_gaussians(means, quats, T):
    """Move Gaussians [..., N, 3] / [..., N, 4] from part frame to world by T_world_part [..., 4, 4].
    Covariance becomes R Sigma R^T; scales and opacities are unchanged (CLAUDE.md §7.3)."""
    R, t = T[..., :3, :3], T[..., :3, 3]
    return means @ R.transpose(-1, -2) + t[..., None, :], quat_mul(matrix_to_quat(R)[..., None, :], quats)


# ---------------------------------------------------------------- cameras
R_CV_FROM_SAPIEN = torch.tensor([[0., -1, 0], [0, 0, -1], [1, 0, 0]])


def sapien_pose_to_viewmat(T_world_cam):
    """SAPIEN camera/link pose (x forward, y left, z up) -> OpenCV viewmat. Use for the hand camera (camera_link)."""
    C = torch.eye(4, dtype=T_world_cam.dtype, device=T_world_cam.device)
    C[:3, :3] = R_CV_FROM_SAPIEN.to(T_world_cam).T
    return se3_inverse(T_world_cam @ C)


def look_at_viewmat(eye, target, up=(0.0, 0.0, 1.0)):
    """OpenCV viewmat of a camera at eye looking at target (same camera as ManiSkill's sapien_utils.look_at)."""
    eye = torch.as_tensor(eye, dtype=torch.float32) if not torch.is_tensor(eye) else eye
    target, up = (torch.as_tensor(v, dtype=eye.dtype, device=eye.device) for v in (target, up))
    z = F.normalize(target - eye, dim=-1)
    x = F.normalize(torch.cross(z, up.expand_as(z), dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], -2)  # rows = camera axes in world = world->camera rotation
    V = torch.zeros(*eye.shape[:-1], 4, 4, dtype=eye.dtype, device=eye.device)
    V[..., :3, :3] = R
    V[..., :3, 3] = -(R @ eye[..., None]).squeeze(-1)
    V[..., 3, 3] = 1
    return V


def fov_to_K(fovy, width, height):
    """Intrinsics from the VERTICAL field of view (ManiSkill CameraConfig.fov), square pixels."""
    f = height / 2 / math.tan(fovy / 2)
    return torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=torch.float32)


def project(points, viewmat, K):
    """World points [..., N, 3] -> pixel coords uv [..., N, 2] and camera depth z [..., N]."""
    pc = points @ viewmat[..., :3, :3].transpose(-1, -2) + viewmat[..., None, :3, 3]
    uvw = pc @ K.transpose(-1, -2)
    return uvw[..., :2] / uvw[..., 2:3], pc[..., 2]


def backproject(depth, viewmat, K, rows, cols):
    """Pixels (rows, cols) [N] of a z-depth image [H, W] (metres, camera z as ManiSkill) -> world points [N, 3].
    Uses pixel centres (col + 0.5, row + 0.5), the inverse of project() + floor()."""
    z = depth[rows, cols]
    uv1 = torch.stack([cols + 0.5, rows + 0.5, torch.ones_like(z)], -1).to(K.dtype)
    pc = (uv1 @ torch.linalg.inv(K).T) * z[:, None]
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    return (pc - t) @ R  # camera -> world: R^T (pc - t)
