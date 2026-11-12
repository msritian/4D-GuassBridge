"""C1 geometry checks (pure torch). Camera functions are checked against ManiSkill in test_sim_integration.py."""
import math

import torch

from gaussbridge import geometry as G


def _random_quats(n, gen):
    q = torch.nn.functional.normalize(torch.randn(n, 4, generator=gen, dtype=torch.float64), dim=-1)
    # 180-degree turns are the classic failure case of matrix->quaternion conversion
    s = math.sqrt(0.5)
    edge = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, s, s, 0]], dtype=torch.float64)
    return torch.cat([q, edge])


def test_rotation_and_pose_round_trips():
    gen = torch.Generator().manual_seed(0)
    q = _random_quats(500, gen)
    R = G.quat_to_matrix(q)
    torch.testing.assert_close(R @ R.mT, torch.eye(3, dtype=torch.float64).expand_as(R))
    assert torch.allclose(torch.det(R), torch.ones(len(R), dtype=torch.float64))

    q2 = G.matrix_to_quat(R)  # same rotation, sign fixed to w >= 0
    assert ((q2 * q).sum(-1).abs() - 1).abs().max() < 1e-12 and (q2[:, 0] >= 0).all()
    torch.testing.assert_close(G.rot6d_to_matrix(G.matrix_to_rot6d(R)), R)
    # the model outputs arbitrary 6 numbers: decoding must still give a proper rotation
    R_noisy = G.rot6d_to_matrix(G.matrix_to_rot6d(R) + 0.3 * torch.randn(len(R), 6, generator=gen, dtype=torch.float64))
    torch.testing.assert_close(R_noisy @ R_noisy.mT, torch.eye(3, dtype=torch.float64).expand_as(R))

    p = torch.cat([torch.randn(len(q), 3, generator=gen, dtype=torch.float64), q], -1)
    T = G.pose7_to_matrix(p)
    torch.testing.assert_close(G.pose7_to_matrix(G.matrix_to_pose7(T)), T)
    torch.testing.assert_close(G.se3_inverse(T) @ T, torch.eye(4, dtype=torch.float64).expand_as(T))


def test_cube_symmetries_form_a_group():
    S = G.cube_symmetries()
    assert len(S) == 24 and len({tuple(m.flatten().tolist()) for m in S}) == 24
    products = {tuple(m.flatten().tolist()) for m in (S[:, None] @ S[None]).flatten(0, 1)}
    assert products == {tuple(m.flatten().tolist()) for m in S}  # closed under composition


def test_backproject_inverts_project():
    gen = torch.Generator().manual_seed(2)
    V = G.look_at_viewmat(torch.tensor([0.4, -0.3, 0.7], dtype=torch.float64), [0.0, 0.1, 0.05])
    K = G.fov_to_K(math.radians(60), 64, 48).double()
    rows = torch.randint(0, 48, (200,), generator=gen)
    cols = torch.randint(0, 64, (200,), generator=gen)
    depth = 0.3 + torch.rand(48, 64, generator=gen, dtype=torch.float64)
    pts = G.backproject(depth, V, K, rows, cols)
    uv, z = G.project(pts, V, K)
    torch.testing.assert_close(z, depth[rows, cols])
    torch.testing.assert_close(uv, torch.stack([cols, rows], -1).double() + 0.5)  # pixel centres
    assert torch.equal(uv.floor().long(), torch.stack([cols, rows], -1))


def test_transform_gaussians_rotates_covariance():
    gen = torch.Generator().manual_seed(1)
    means = torch.randn(50, 3, generator=gen, dtype=torch.float64)
    quats = _random_quats(45, gen)
    scales = torch.rand(50, 3, generator=gen, dtype=torch.float64) + 0.1
    T = G.pose7_to_matrix(torch.cat([torch.randn(3, generator=gen, dtype=torch.float64), _random_quats(1, gen)[0]]))

    def cov(q):
        Rq = G.quat_to_matrix(q)
        return Rq @ torch.diag_embed(scales ** 2) @ Rq.mT

    m_w, q_w = G.transform_gaussians(means, quats, T)
    R = T[:3, :3]
    torch.testing.assert_close(m_w, means @ R.T + T[:3, 3])
    torch.testing.assert_close(cov(q_w), R @ cov(quats) @ R.T)  # catches a wrong quaternion product order
