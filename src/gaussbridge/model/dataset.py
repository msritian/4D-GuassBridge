"""C3: turn motion Episodes into training samples for the motion model (CLAUDE.md §6.3, §6.4, §8.2).

What one training sample is
---------------------------
The model learns to fill in the middle of a motion given its start and end. One sample therefore holds:

    x0        [N, E, F]  the full, clean motion grid (the answer the model should learn to produce)
    known     [N, E]     which cells are given to the model (start row, and some or all of the end row)
    x_known   [N, E, F]  the values of the given cells (slightly noised, see "Endpoint noise"); 0 elsewhere
    ent_mask  [E]        which entity slots exist (robot + this task's objects; the rest are padding)
    log_dur   scalar     normalised log of the motion's duration in seconds (the model generates it too)
    shapes    [K, P, 6]  each object's shape as P surface points [x, y, z, r, g, b] in the object's own frame
    goal      int        which goal mode was drawn (GOAL_FULL / GOAL_OBJECTS / GOAL_NONE), for logging

with N = 128 time steps, E = 4 entity slots (slot 0 = robot, slots 1..3 = objects), F = 10 numbers per cell,
K = 3 object slots, P = 512 points. The grid layout itself is defined in trajectory.py (`episode_to_motion`).

Goal modes (drawn fresh for every sample, so one episode teaches all three)
----------------------------------------------------------------------------
    full     start + end of every entity known                       -> the "full goal" mode
    objects  start of everything + end of the objects only            -> the "object-only goal" mode (robot end free)
    none     start only                                               -> the forward-only baseline (§8.4)
Probabilities: none = 0.1, objects = 0.3, full = 0.6 (CLAUDE.md §8.2; read as mutually exclusive).

Endpoint noise (augmentation)
-----------------------------
At test time the start/end poses come from measurements and are slightly wrong. So during training the GIVEN
copy of the endpoints (x_known) gets small random errors, while the target x0 stays exact. The model learns to
produce a clean motion that starts/ends near, not exactly at, the given values. Sizes (CLAUDE.md §6.4), each a
Gaussian standard deviation: object position 5 mm per axis, object rotation 2 deg about a random axis, arm joints
0.01 rad. Finger opening, gripper command and the attached flag are left exact (they are read off directly).

Shapes
------
The model is told what each object looks like through surface points. Until the Gaussian reconstruction exists
(task B4) we use points sampled on the exact 4 cm cube (`box_shape_points`); afterwards `PartSplats.shape_points`
returns the same [P, 6] format from the reconstructed Gaussians, so only the source changes.
"""
import math

import torch

from gaussbridge import geometry as G
from gaussbridge.trajectory import K_MAX, POS_SCALE, Q_HI, Q_LO, episode_to_motion, known_mask

# ---------------------------------------------------------------------------------------------------- constants
GOAL_FULL, GOAL_OBJECTS, GOAL_NONE = 0, 1, 2
GOAL_NAMES = {GOAL_FULL: "full", GOAL_OBJECTS: "objects", GOAL_NONE: "none"}

SHAPE_POINTS = 512  # points per object shape (CLAUDE.md §6.4)

# Every v1 object is a 4 cm cube; colours are ManiSkill's (stack_cube.py / pick_cube.py), RGB in [0, 1].
CUBE_HALF_SIZE = 0.02
OBJECT_COLORS = {"cubeA": (1.0, 0.0, 0.0), "cubeB": (0.0, 1.0, 0.0), "cube": (1.0, 0.0, 0.0)}

# Endpoint-noise sizes (standard deviations), CLAUDE.md §6.4.
NOISE_OBJ_POS = 0.005               # m, per axis
NOISE_OBJ_ROT = math.radians(2.0)   # rad, about a random axis
NOISE_JOINT = 0.01                  # rad, each of the 7 arm joints


# ---------------------------------------------------------------------------------------------------- shapes
def box_shape_points(half_size=CUBE_HALF_SIZE, rgb=(1.0, 0.0, 0.0), n=SHAPE_POINTS, seed=0):
    """n points spread uniformly over the surface of a cube centred at the origin, as [n, 6] = [xyz, rgb].

    Same format and frame as `PartSplats.shape_points` (the object's own frame, colour in [0, 1]), so it is a
    drop-in placeholder until reconstructed Gaussians exist. All 6 faces have equal area, so picking a face
    uniformly and then a point uniformly on it gives a uniform distribution over the surface."""
    g = torch.Generator().manual_seed(seed)
    face = torch.randint(0, 6, (n,), generator=g)          # which of the 6 faces
    axis, sign = face // 2, (face % 2) * 2 - 1              # face = (axis x/y/z, side -/+)
    xyz = (torch.rand(n, 3, generator=g) * 2 - 1) * half_size
    xyz[torch.arange(n), axis] = sign.float() * half_size   # push the point onto that face
    return torch.cat([xyz, torch.tensor(rgb).expand(n, 3)], -1)


def task_shapes(obj_names, n=SHAPE_POINTS):
    """Shape points of every object slot, [K_MAX, n, 6]; unused slots are zeros (they are masked anyway)."""
    shapes = torch.zeros(K_MAX, n, 6)
    for k, name in enumerate(obj_names):
        shapes[k] = box_shape_points(CUBE_HALF_SIZE, OBJECT_COLORS[name], n, seed=k)
    return shapes


# ---------------------------------------------------------------------------------------------------- duration
def duration_stats(episodes):
    """Mean and standard deviation of log(duration in seconds) over a list of Episodes.

    The model sees the duration as (log_dur - mu) / sigma, so it is roughly in [-2, 2]. Compute these on the
    TRAINING split only and keep them with the model checkpoint: the sampler needs the same numbers to turn a
    generated duration back into seconds."""
    logs = torch.tensor([math.log(ep.num_steps / ep.control_freq) for ep in episodes], dtype=torch.float64)
    sigma = logs.std().item() if len(logs) > 1 else 1.0
    return {"mu": logs.mean().item(), "sigma": max(sigma, 1e-3)}  # floor: one repeated length must not divide by 0


# ---------------------------------------------------------------------------------------------------- noise
def _random_small_rotations(shape, std, generator=None, device=None):
    """Rotation matrices [*shape, 3, 3]: random axis, angle ~ N(0, std) (axis-angle -> matrix, Rodrigues)."""
    axis = torch.nn.functional.normalize(torch.randn(*shape, 3, generator=generator, device=device), dim=-1)
    angle = torch.randn(*shape, 1, 1, generator=generator, device=device) * std
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    Kx = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).unflatten(-1, (3, 3))  # cross-product matrix
    return torch.eye(3, device=device) + torch.sin(angle) * Kx + (1 - torch.cos(angle)) * Kx @ Kx


def add_endpoint_noise(x, n_objects, generator=None):
    """Return a noised copy of motion grids x [..., N, E, F] (model units, any device). Applied to every row; the
    caller keeps only the known cells. Robot slot: the 7 arm joints. Object slots 1..n_objects: position, rotation."""
    x = x.clone()
    dev = x.device
    # Arm joints. The grid stores q_norm = 2 (q - lo) / (hi - lo) - 1, so a change of d rad is 2 d / (hi - lo).
    joint_scale = (2 * NOISE_JOINT / (Q_HI - Q_LO)).to(dev)
    x[..., 0, :7] += torch.randn(*x.shape[:-2], 7, generator=generator, device=dev) * joint_scale
    if n_objects:
        objs = x[..., 1:1 + n_objects, :]
        # Position. The grid stores (p - c) / POS_SCALE, so a change of d metres is d / POS_SCALE.
        objs[..., :3] += torch.randn(*objs.shape[:-1], 3, generator=generator, device=dev) * (NOISE_OBJ_POS / POS_SCALE)
        # Rotation: decode the 6D rotation, turn it slightly (in the world frame), encode again.
        R = G.rot6d_to_matrix(objs[..., 3:9])
        dR = _random_small_rotations(objs.shape[:-1], NOISE_OBJ_ROT, generator, dev)
        objs[..., 3:9] = G.matrix_to_rot6d(dR @ R)
    return x


# ---------------------------------------------------------------------------------------------------- dataset
class MotionDataset(torch.utils.data.Dataset):
    """Training samples from a list of Episodes of ONE task (all with the same objects).

    Args:
        episodes:        list of trajectory.Episode.
        dur_stats:       {"mu", "sigma"} for normalising log-duration. Default: computed from `episodes`
                         (correct for the training split; pass the training stats for val/test).
        p_none:          probability of the "none" goal mode (start only).
        p_objects:       probability of the "objects" goal mode (robot end unknown).
        endpoint_noise:  add noise to the given endpoints (training: True; evaluation: False).
        shapes:          [K_MAX, P, 6] object shape points. Default: exact cube surface points.

    The motion grids are computed once here (the slow part: resampling to N steps), so __getitem__ only draws a
    goal mode and the noise. Randomness uses torch's global RNG, which DataLoader seeds differently per worker.
    For training at scale use `make_batch`: the same samples, built for a whole batch at once (optionally on the
    GPU) from an explicit random generator, so a resumed run draws exactly the same batches.
    """

    def __init__(self, episodes, dur_stats=None, p_none=0.1, p_objects=0.3, endpoint_noise=True, shapes=None):
        if not episodes:
            raise ValueError("no episodes")
        names = {tuple(ep.obj_names) for ep in episodes}
        if len(names) != 1:
            raise ValueError(f"episodes of different tasks mixed: object lists {names}")
        self.obj_names = list(names.pop())
        self.n_objects = len(self.obj_names)
        self.p_none, self.p_objects, self.endpoint_noise = p_none, p_objects, endpoint_noise
        self.dur_stats = dur_stats or duration_stats(episodes)

        motions = [episode_to_motion(ep) for ep in episodes]
        self.x0 = torch.stack([m["x"] for m in motions])          # [B, N, E, F]
        self.ent_mask = motions[0]["ent_mask"]                     # [E], the same for every episode of a task
        log_seconds = torch.tensor([math.log(ep.num_steps / ep.control_freq) for ep in episodes])
        self.log_dur = (log_seconds - self.dur_stats["mu"]) / self.dur_stats["sigma"]
        self.shapes = task_shapes(self.obj_names) if shapes is None else shapes
        self.meta = [ep.meta for ep in episodes]

    def __len__(self):
        return len(self.x0)

    def draw_goal(self):
        """Draw a goal mode: none with p_none, objects with p_objects, otherwise full."""
        u = torch.rand(()).item()
        if u < self.p_none:
            return GOAL_NONE
        return GOAL_OBJECTS if u < self.p_none + self.p_objects else GOAL_FULL

    def to(self, device):
        """Move the stored grids to `device` (they are small: 12k episodes are about 250 MB) for fast make_batch."""
        self.x0, self.ent_mask = self.x0.to(device), self.ent_mask.to(device)
        self.log_dur, self.shapes = self.log_dur.to(device), self.shapes.to(device)
        return self

    def make_batch(self, idx, generator=None):
        """A batch for episodes idx [B] (same content as collating __getitem__ for each): goal modes, known masks
        and endpoint noise are drawn for the whole batch at once with `generator` (on the data's device)."""
        dev = self.x0.device
        idx = torch.as_tensor(idx, device=dev)
        B = len(idx)
        x0 = self.x0[idx]
        u = torch.rand(B, generator=generator, device=dev)
        goal = torch.full((B,), GOAL_FULL, device=dev)
        goal[u < self.p_none + self.p_objects] = GOAL_OBJECTS
        goal[u < self.p_none] = GOAL_NONE
        ent = self.ent_mask.expand(B, -1)
        known = torch.zeros(*x0.shape[:-1], dtype=torch.bool, device=dev)    # [B, N, E]
        known[:, 0] = ent
        known[:, -1] = ent & (goal != GOAL_NONE)[:, None]
        known[:, -1, 0] &= goal == GOAL_FULL                                   # robot end only in the full mode
        given = add_endpoint_noise(x0, self.n_objects, generator) if self.endpoint_noise else x0
        x_known = torch.where(known[..., None], given, torch.zeros_like(given))
        return {"x0": x0, "x_known": x_known, "known": known, "ent_mask": ent, "log_dur": self.log_dur[idx],
                "shapes": self.shapes.expand(B, *self.shapes.shape), "goal": goal}

    def __getitem__(self, i):
        x0 = self.x0[i]
        goal = self.draw_goal()
        # [N, E] bool: start row of every real entity + the end row per goal mode; padded slots never known.
        known = known_mask(self.ent_mask, GOAL_NAMES[goal])
        given = add_endpoint_noise(x0, self.n_objects) if self.endpoint_noise else x0
        x_known = torch.where(known[..., None], given, torch.zeros_like(given))  # zero outside the known cells
        return {"x0": x0, "x_known": x_known, "known": known, "ent_mask": self.ent_mask,
                "log_dur": self.log_dur[i], "shapes": self.shapes, "goal": goal}
