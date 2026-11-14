"""C5: motion-quality metrics (CLAUDE.md §10.1). Pure torch, CPU-friendly, no simulator.

Every metric works on TRACKS: the 3D positions of a few points that summarise a motion, sampled on the model's
128-step grid. As in §10.1 the tracked points are the gripper centre (TCP, from forward kinematics) and the centre
of every object:

    tracks [..., N, P, 3]   N = 128 steps, P = 1 + K points (TCP first, then the K objects), metres

A test case (one start scene + one goal) has R reference motions (planner solutions) and S generated samples:

    ADE(a, b)        average displacement error: mean over steps and points of the distance |a - b|  (m)
    minADE           for each reference, the distance to its closest sample; averaged over references
                     ("is every real solution matched by at least one sample?", in metres)
    coverage@thr     fraction of references that have a sample within ADE thr (3 cm)  -> realism + mode coverage
    precision@thr    fraction of samples within ADE thr of some reference              -> no invented nonsense
    APD              mean ADE between pairs of different samples                       -> diversity
    jerk             mean squared third time-derivative of the tracks, m^2/s^6       -> smoothness

Coverage and precision together separate "diverse" from "noisy": random noise has high APD but low precision;
a model that always produces the same motion has high precision but low coverage.

Endpoint errors compare the final poses with the goal: object position (cm), object rotation (degrees, the smallest
angle over the 24 rotations that map a cube onto itself, since a cube turned by 90 deg looks identical) and TCP
position (cm, only meaningful when the robot's end was given).

Not here yet: penetration (needs the sphere proxies of task C6) and physics-replay success (needs task A3).
"""
import torch

from gaussbridge import geometry as G
from gaussbridge.sim.kinematics import PandaFK
from gaussbridge.trajectory import decode_motion

COVERAGE_THRESHOLD = 0.03  # m (§10.1)


# ---------------------------------------------------------------------------------------------------- tracks
def motion_tracks(x, n_objects, fk: PandaFK, base_pose=None):
    """Motion grids x [..., N, E, F] (model units) -> tracks [..., N, 1 + n_objects, 3] in metres (TCP, objects)."""
    d = decode_motion(x, 0.0)  # the duration does not matter for positions
    tcp = fk.tcp_pose(d["qpos"].to(fk.dtype), base_pose)[..., :3, 3].to(x.dtype)       # [..., N, 3]
    return torch.cat([tcp[..., None, :], d["obj_pos"][..., :n_objects, :]], -2)


# ---------------------------------------------------------------------------------------------------- distances
def ade(a, b):
    """Average displacement error between tracks a, b [..., N, P, 3] (broadcasting) -> [...] metres."""
    return (a - b).norm(dim=-1).mean((-1, -2))


def pairwise_ade(A, B):
    """[S, N, P, 3] x [R, N, P, 3] -> [S, R] ADE of every pair."""
    return ade(A[:, None], B[None])


def min_ade(samples, refs):
    """Mean over references of the ADE to their closest sample (m)."""
    return pairwise_ade(samples, refs).min(0).values.mean()


def coverage(samples, refs, thr=COVERAGE_THRESHOLD):
    """Fraction of references with at least one sample within ADE thr."""
    return (pairwise_ade(samples, refs).min(0).values < thr).float().mean()


def precision(samples, refs, thr=COVERAGE_THRESHOLD):
    """Fraction of samples within ADE thr of at least one reference."""
    return (pairwise_ade(samples, refs).min(1).values < thr).float().mean()


def apd(samples):
    """Average pairwise ADE among different samples (m); 0 for a single sample."""
    S = samples.shape[0]
    if S < 2:
        return samples.new_zeros(())
    D = pairwise_ade(samples, samples)
    return D.sum() / (S * (S - 1))  # the diagonal is 0


# ---------------------------------------------------------------------------------------------------- smoothness
def mean_squared_jerk(tracks, duration):
    """Mean over steps and points of |d^3 x / dt^3|^2 (m^2/s^6), tracks [..., N, P, 3] on an evenly spaced grid
    covering `duration` seconds ([...] or scalar). Third finite difference divided by dt^3, in float64: dt^3 is
    ~1e-5. float32 input tracks still carry a rounding floor of ~1e-4 m^2/s^6, negligible next to real motions
    (1.2-8 m^2/s^6 for the StackCube dev episodes)."""
    tracks = tracks.double()
    dt = torch.as_tensor(duration, dtype=torch.float64, device=tracks.device) / (tracks.shape[-3] - 1)
    j = torch.diff(tracks, n=3, dim=-3) / dt[..., None, None, None] ** 3
    return j.square().sum(-1).mean((-1, -2))


# ---------------------------------------------------------------------------------------------------- endpoints
def rotation_error_deg(R_pred, R_goal, symmetric=True):
    """Angle (deg) between rotations [..., 3, 3]; with symmetric=True the smallest over the 24 cube symmetries
    (R_goal @ S looks identical to R_goal for a cube)."""
    if symmetric:
        S = G.cube_symmetries(R_goal.dtype).to(R_goal.device)                       # [24, 3, 3]
        R_goal = R_goal[..., None, :, :] @ S                                         # [..., 24, 3, 3]
        R_pred = R_pred[..., None, :, :]
    D = R_pred @ R_goal.mT
    cos = (D.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    sin = torch.stack([D[..., 2, 1] - D[..., 1, 2], D[..., 0, 2] - D[..., 2, 0], D[..., 1, 0] - D[..., 0, 1]],
                      -1).norm(dim=-1) / 2
    angle = torch.rad2deg(torch.atan2(sin, cos))
    return angle.min(-1).values if symmetric else angle


def endpoint_errors(x, x_goal, n_objects, fk: PandaFK, base_pose=None):
    """Final-step errors of motions x [..., N, E, F] against goal grids x_goal (only their last step is used).
    Returns {"obj_pos_cm" [..., K], "obj_rot_deg" [..., K] (cube-symmetric), "tcp_pos_cm" [...]}."""
    a, b = decode_motion(x[..., -1:, :, :], 0.0), decode_motion(x_goal[..., -1:, :, :], 0.0)
    k = n_objects
    tcp = [fk.tcp_pose(d["qpos"].to(fk.dtype), base_pose)[..., 0, :3, 3] for d in (a, b)]
    return {
        "obj_pos_cm": 100 * (a["obj_pos"][..., 0, :k, :] - b["obj_pos"][..., 0, :k, :]).norm(dim=-1),
        "obj_rot_deg": rotation_error_deg(a["obj_rot"][..., 0, :k, :, :], b["obj_rot"][..., 0, :k, :, :]),
        "tcp_pos_cm": 100 * (tcp[0] - tcp[1]).norm(dim=-1).to(x.dtype),
    }


# ---------------------------------------------------------------------------------------------------- one case
def case_metrics(sample_tracks, ref_tracks, sample_durations, thr=COVERAGE_THRESHOLD):
    """All track metrics for one test case: S samples vs R references -> dict of floats."""
    return {
        "min_ade_m": min_ade(sample_tracks, ref_tracks).item(),
        "coverage": coverage(sample_tracks, ref_tracks, thr).item(),
        "precision": precision(sample_tracks, ref_tracks, thr).item(),
        "apd_m": apd(sample_tracks).item(),
        "jerk": mean_squared_jerk(sample_tracks, sample_durations).mean().item(),
    }
