"""C5/C7: baselines for the motion model, in the same grid format, so they are scored by exactly the same code.

straight_line (CLAUDE.md §8.4): the simplest way to connect start and goal.
    robot joints + finger   linear from start to end
    gripper command         start value for the first half, end value for the second half
    objects                 position linear, rotation slerp (shortest arc), never "attached"
It reaches the goal exactly but ignores how things actually move: objects slide through the air without being
grasped, the arm takes the joint-space straight line. A good model must beat it on realism (coverage, precision,
physics replay), not on endpoint error.

What is "the end" depends on the goal mode (`known` of the last step):
    full     everything goes to its given end
    objects  objects go to their given end; the robot's end is unknown, so the robot stays at its start
    none     nothing has an end; everything stays at its start
"""
import torch

from gaussbridge import geometry as G


def straight_line(x_known, known, ent_mask):
    """x_known [..., N, E, F] (given values), known [..., N, E] -> straight-line motion grid [..., N, E, F].
    The start (step 0) must be known for every real entity."""
    N = x_known.shape[-3]
    start, end = x_known[..., 0, :, :], x_known[..., -1, :, :]                      # [..., E, F]
    end = torch.where(known[..., -1, :, None], end, start)                          # unknown end -> stay put
    t = torch.linspace(0, 1, N, dtype=x_known.dtype, device=x_known.device)
    tt = t.view(N, 1, 1)
    x = start[..., None, :, :] * (1 - tt) + end[..., None, :, :] * tt               # linear everything first

    # robot gripper command: hold the start value, switch to the end value half way
    half = (t >= 0.5).view(N, 1)
    x[..., 0, 8] = torch.where(half, end[..., None, 0, 8:9], start[..., None, 0, 8:9])[..., 0]
    # objects: rotation by slerp (a linear blend of 6D vectors is not a constant-speed rotation), never attached
    q0 = G.matrix_to_quat(G.rot6d_to_matrix(start[..., 1:, 3:9]))                   # [..., K, 4]
    q1 = G.matrix_to_quat(G.rot6d_to_matrix(end[..., 1:, 3:9]))
    q = G.slerp(q0[..., None, :, :], q1[..., None, :, :], t.view(N, 1).expand(N, q0.shape[-2]))
    x[..., 1:, 3:9] = G.matrix_to_rot6d(G.quat_to_matrix(q))
    x[..., 1:, 9] = -1.0
    # exact endpoints (slerp/re-encoding can differ in the last digits) and zero padding, like the model's sampler
    x = torch.where(known[..., None], x_known, x)
    return x * ent_mask[..., None, :, None]
