"""C6: grasp consistency, objects move only while held and then rigidly with the gripper (CLAUDE.md §8.3).

attach_energy (soft, used as sampling guidance)
------------------------------------------------
r_n = position of the object in the TCP frame, (T_tcp(n)^-1 T_obj(n)).t. While an object is held, r should not change:

    E = sum_n  sigmoid(4 att_n) * sigmoid(4 att_{n+1}) * |r_{n+1} - r_n|^2      (att = the grid's attached feature, +-1)

project_attachment (hard, after sampling)
-----------------------------------------
Per object, with [n0, n1] the longest run of steps marked attached (att > 0):
    G = T_tcp(ng)^-1 T_obj(ng), ng = min(n0 + 2, n1)  the grip, read once the fingers have settled (planner data:
                                                       the object moves up to 1.6 mm in the hand on the first held
                                                       step, then <= 0.3 mm for the rest of the carry)
    before n0     the object stays at its start pose
    n0 .. n1      T_obj(n) = T_tcp(n) G               rigid in the hand
    after n1      the object stays where it was released, T_obj(n1)
    last step     the given end (when the object's end is known)
and the attached feature is rewritten to +1 on [n0, n1], -1 elsewhere. The sample is REJECTED when
    * the release pose misses the given end by more than max_release_err (1 cm), or
    * an object whose end is more than 1 cm from its start was never held (objects only move when carried; true for
      every v1 task).
An object that is never held and does not need to move simply stays at its start (and its given end at the last step).

Everything is computed on decoded poses and written back into the motion grid, so the output is again a grid that
motion_to_episode, the metrics and the renderer consume as usual.
"""
import torch

from gaussbridge import geometry as G
from gaussbridge.trajectory import POS_CENTER, POS_SCALE, decode_motion

MAX_RELEASE_ERR = 0.01   # m (§8.3)
MOVE_THRESHOLD = 0.01    # m: an object whose end is farther than this from its start must have been carried
GRIP_SETTLE_STEPS = 2    # the grip is read this many steps after grasping (fingers still closing at the first step)


def _tcp_poses(qpos, fk):
    return fk.tcp_pose(qpos.to(fk.dtype)).to(qpos.dtype)                                   # [..., N, 4, 4]


def _object_poses(d, n_objects):
    T = torch.zeros(*d["obj_pos"].shape[:-2], n_objects, 4, 4, dtype=d["obj_pos"].dtype, device=d["obj_pos"].device)
    T[..., :3, :3] = d["obj_rot"][..., :n_objects, :, :]
    T[..., :3, 3] = d["obj_pos"][..., :n_objects, :]
    T[..., 3, 3] = 1
    return T                                                                                 # [..., N, K, 4, 4]


def attach_energy(x, n_objects, fk):
    """[B] attachment energy of motion grids x [B, N, E, F] (differentiable)."""
    d = decode_motion(x, 0.0)
    T_tcp = _tcp_poses(d["qpos"], fk)                                                        # [B, N, 4, 4]
    T_obj = _object_poses(d, n_objects)                                                      # [B, N, K, 4, 4]
    r = (G.se3_inverse(T_tcp)[:, :, None] @ T_obj)[..., :3, 3]                               # [B, N, K, 3]
    w = torch.sigmoid(4 * d["attached"][..., :n_objects])                                    # [B, N, K]
    step = (r[:, 1:] - r[:, :-1]).square().sum(-1)                                           # [B, N-1, K]
    return (w[:, 1:] * w[:, :-1] * step).sum((1, 2))


def longest_run(mask):
    """Longest run of True in a 1-D bool tensor -> (start, end) inclusive, or None if there is none."""
    best, start = None, None
    for i, v in enumerate(mask.tolist() + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - 1 - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    return best


@torch.no_grad()
def project_attachment(x, known, n_objects, fk, max_release_err=MAX_RELEASE_ERR):
    """Make grasps physically consistent (module docstring). x [B, N, E, F] motion grids (given cells already
    imposed, as the sampler returns them), known [B, N, E]. Returns (x_projected [B, N, E, F], ok [B] bool,
    info {"release_err_m" [B, K] (nan if never held), "held" [B, K] bool, "interval" [B][K] (n0, n1) or None})."""
    B, N = x.shape[:2]
    d = decode_motion(x, 0.0)
    T_tcp = _tcp_poses(d["qpos"], fk)                                                        # [B, N, 4, 4]
    T_obj = _object_poses(d, n_objects)                                                      # [B, N, K, 4, 4]
    out = x.clone()
    ok = torch.ones(B, dtype=torch.bool, device=x.device)
    release_err = torch.full((B, n_objects), float("nan"), device=x.device)
    held = torch.zeros(B, n_objects, dtype=torch.bool, device=x.device)
    intervals = [[None] * n_objects for _ in range(B)]
    for b in range(B):
        for k in range(n_objects):
            slot = 1 + k
            end_known = bool(known[b, -1, slot])
            T_start, T_end = T_obj[b, 0, k], T_obj[b, -1, k]
            run = longest_run(d["attached"][b, :, k] > 0)
            T_new = T_start.expand(N, 4, 4).clone()
            att = torch.full((N,), -1.0, device=x.device)
            if run is None:
                if end_known and (T_end[:3, 3] - T_start[:3, 3]).norm() > MOVE_THRESHOLD:
                    ok[b] = False                                      # it had to move but was never carried
            else:
                n0, n1 = run
                intervals[b][k], held[b, k] = run, True
                ng = min(n0 + GRIP_SETTLE_STEPS, n1)
                grip = G.se3_inverse(T_tcp[b, ng]) @ T_obj[b, ng, k]
                T_new[n0:n1 + 1] = T_tcp[b, n0:n1 + 1] @ grip
                T_new[n1 + 1:] = T_new[n1]
                att[n0:n1 + 1] = 1.0
                if end_known:
                    release_err[b, k] = (T_new[n1, :3, 3] - T_end[:3, 3]).norm()
                    if release_err[b, k] > max_release_err:
                        ok[b] = False
            if end_known:
                T_new[-1] = T_end                                      # the given end, exactly
            out[b, :, slot, :3] = (T_new[:, :3, 3] - POS_CENTER.to(x)) / POS_SCALE
            out[b, :, slot, 3:9] = G.matrix_to_rot6d(T_new[:, :3, :3])
            out[b, :, slot, 9] = att
            if end_known:                                              # keep the given end cell bit-exact
                out[b, -1, slot] = x[b, -1, slot]
            out[b, 0, slot] = x[b, 0, slot]                            # and the start
    return out, ok, {"release_err_m": release_err, "held": held, "interval": intervals}
