"""B1/B2: parts, multi-view captures and asset-mode configurations (CLAUDE.md §7.1-7.2).

A part = one rigid thing that gets its own Gaussians: every robot link, every task object and each static
background entity (table, ground). The part table is built from ManiSkill's segmentation_id_map, never hard-coded.
"""
import dataclasses

import numpy as np
import sapien
import torch
from mani_skill.utils.structs.pose import Pose

from gaussbridge import geometry as G
from gaussbridge.sim.envs import CUBE_HALF_SIZE, DEFAULT_ROBOT, HOME_QPOS, OBJECTS, POLICY_CAMERAS
from gaussbridge.sim.kinematics import PandaFK
from gaussbridge.trajectory import Q_HI, Q_LO

IGNORE = -1  # part index for pixels that belong to no part (sky / empty)


def _task_id(u):
    """'StackCube-v1' for both StackCube-v1 and StackCubeCapture-v1 envs."""
    return getattr(u, "task_id", None) or u.spec.id


@dataclasses.dataclass(frozen=True)
class Part:
    name: str
    kind: str      # "robot" | "object" | "background"
    seg_id: int    # ManiSkill segmentation id
    visual: bool = True  # has render geometry (False: panda_hand_tcp, panda_link8, camera_base_link, finger pads)


def _has_visual(ent):
    for obj in ent._objs:
        e = obj if isinstance(obj, sapien.Entity) else obj.entity  # actors hold entities, links hold components
        body = e.find_component_by_type(sapien.render.RenderBodyComponent)
        if body is not None and len(body.render_shapes) > 0:
            return True
    return False


def part_table(env):
    """Parts of the scene, ordered by segmentation id. Hidden helpers (PickCube's goal_site) are excluded.
    Robot links without visual geometry (panda_hand_tcp, ...) are kept so link names stay complete; they simply
    have no pixels (`visual=False`)."""
    u = env.unwrapped
    hidden = {getattr(o, "name", None) for o in getattr(u, "_hidden_objects", [])}
    robot_links = {link.name for link in u.agent.robot.links}
    objects = set(OBJECTS[_task_id(u)])
    parts = []
    for seg_id, ent in sorted(u.segmentation_id_map.items()):
        name = getattr(ent, "name", None)
        if seg_id == 0 or name is None or name in hidden:
            continue
        kind = "robot" if name in robot_links else "object" if name in objects else "background"
        parts.append(Part(name, kind, int(seg_id), _has_visual(ent)))
    return parts


def seg_lut(parts):
    """Lookup table: ManiSkill segmentation id -> part index (IGNORE for anything else)."""
    lut = torch.full((max(p.seg_id for p in parts) + 1,), IGNORE, dtype=torch.int16)
    for i, p in enumerate(parts):
        lut[p.seg_id] = i
    return lut


def part_poses(env, parts):
    """T_world_part [P, 4, 4] (float32) of every part in the current simulator state."""
    ents = env.unwrapped.segmentation_id_map
    return torch.stack([G.pose7_to_matrix(ents[p.seg_id].pose.raw_pose[0].float()) for p in parts])


def reprojection_agreement(state_a, cam_a, state_b, cam_b, tol=0.005):
    """Geometric consistency check of captures (B1 acceptance, also useful on B2 captures).

    Back-project every part's pixels of (state_a, cam_a), move them rigidly with the part from state_a to state_b
    (T_b T_a^-1), project into (state_b, cam_b) and, where visible there (depth within `tol` metres), check that
    they land on that part's pixels. Catches errors in depth units, intrinsics, extrinsics and part poses.
    Returns {part name: (agreeing points, visible points)}."""
    A, B = state_a["cameras"][cam_a], state_b["cameras"][cam_b]
    H, W = B["depth"].shape
    out = {}
    for p, name in enumerate(state_a["part_names"]):
        rows, cols = torch.nonzero((A["seg"] == p) & (A["depth"] > 0), as_tuple=True)
        if len(rows) == 0:
            continue
        X = G.backproject(A["depth"].double(), A["viewmat"].double(), A["K"].double(), rows, cols)
        M = state_b["part_T"][p].double() @ G.se3_inverse(state_a["part_T"][p].double())
        uv, z = G.project(X @ M[:3, :3].T + M[:3, 3], B["viewmat"].double(), B["K"].double())
        px = uv.floor().long()
        inside = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H) & (z > 0)
        px, z = px[inside], z[inside]
        d = B["depth"][px[:, 1], px[:, 0]].double()
        visible = (d > 0) & ((z - d).abs() < tol)
        labels = B["seg"][px[visible, 1], px[visible, 0]]
        out[name] = (int((labels == p).sum()), int(visible.sum()))
    return out


def capture_state(env, parts=None, cameras=None):
    """Render every sensor (or `cameras`) in the current state. The env needs obs_mode with rgb+depth+segmentation.

    Returns {"cameras": {uid: {rgb [H,W,3] uint8, depth [H,W] float32 metres (0 = invalid),
                               seg [H,W] int16 part index (-1 = none), viewmat [4,4] (OpenCV world->camera), K [3,3]}},
             "part_names": [P], "part_kinds": [P], "part_T": [P,4,4], "qpos": [9]}
    Cameras keep the env's order (policy cameras first). Cameras differ in resolution, so they are not stacked."""
    u = env.unwrapped
    parts = parts or part_table(env)
    lut = seg_lut(parts)
    obs = u.get_obs()
    policy = POLICY_CAMERAS[_task_id(u)]  # ManiSkill lists agent-mounted cameras (hand_camera) last; put them first
    order = [c for c in policy if c in obs["sensor_data"]] + [c for c in obs["sensor_data"] if c not in policy]
    out = {}
    for uid in order:
        data = obs["sensor_data"][uid]
        if cameras is not None and uid not in cameras:
            continue
        seg_raw = data["segmentation"][0, ..., 0].long().cpu()
        seg = torch.where(seg_raw < len(lut), lut[seg_raw.clamp(max=len(lut) - 1)], IGNORE).to(torch.int16)
        param = obs["sensor_param"][uid]
        viewmat = torch.eye(4)
        viewmat[:3] = param["extrinsic_cv"][0].float().cpu()
        out[uid] = {
            "rgb": data["rgb"][0].cpu(),
            "depth": data["depth"][0, ..., 0].cpu().float() / 1000.0,  # ManiSkill: int16 millimetres, 0 = invalid
            "seg": seg,
            "viewmat": viewmat,
            "K": param["intrinsic_cv"][0].float().cpu(),
        }
    return {
        "cameras": out,
        "part_names": [p.name for p in parts],
        "part_kinds": [p.kind for p in parts],
        "part_T": part_poses(env, parts),
        "qpos": u.agent.robot.get_qpos()[0].float().cpu(),
    }


# ---------------------------------------------------------------------------------------------------------------
# B2: asset-mode configurations (CLAUDE.md §7.2 "Asset mode")

ARM_NOISE = 0.6            # rad, uniform per arm joint around HOME_QPOS
LIMIT_MARGIN = 0.05        # rad inside the joint limits
FINGER_MAX = 0.04          # m
MIN_LINK_Z = 0.02          # m, reject if any link origin (incl. the TCP, excl. the fixed base) is lower
MIN_OBJ_ROBOT_DIST = 0.12  # m, object centre to the robot's link-origin chain
MIN_OBJ_OBJ_DIST = 0.08    # m, between object centres (> two half-diagonals, 0.07)
OBJ_X, OBJ_Y = (-0.2, 0.15), (-0.2, 0.2)  # m, inside every ring's view and the base camera's
FLOAT_Z = (0.08, 0.25)     # m, centre height of floating objects
P_FLOAT = 0.5
CHAIN = [f"panda_link{i}" for i in range(8)] + ["panda_hand", "panda_hand_tcp"]  # base -> fingertips


def _segment_dist(p, a, b):
    """Distance from points p [..., 3] to segments a->b [S, 3] -> [..., S]."""
    ab = b - a
    t = (((p[..., None, :] - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return (p[..., None, :] - (a + t[..., None] * ab)).norm(dim=-1)


def _uniform_quat(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def random_configs(task, n, seed, robot_uid=None, max_tries=10000):
    """n random asset-mode configurations: [{"qpos": [9], "obj_pose": [K, 7] (order OBJECTS[task]),
    "floating": [K] bool}]. Deterministic in `seed` (an int or a list, e.g. [0, 1] for a held-out stream).
    Robot: HOME_QPOS + U(±ARM_NOISE) per arm joint, clipped to the limits - LIMIT_MARGIN, both fingers U[0, 0.04],
    rejected if a link origin is below MIN_LINK_Z. Objects: resting on the table with random yaw, or (p = P_FLOAT)
    floating with a uniformly random rotation; >= MIN_OBJ_ROBOT_DIST from the robot chain, apart from each other."""
    robot_uid = robot_uid or DEFAULT_ROBOT[task]
    rng = np.random.default_rng(seed)
    fk = PandaFK(robot_uid, dtype=torch.float64)
    home = np.asarray(HOME_QPOS[robot_uid])
    lo, hi = Q_LO.double().numpy() + LIMIT_MARGIN, Q_HI.double().numpy() - LIMIT_MARGIN
    configs = []
    for _ in range(n):
        for _ in range(max_tries):
            q = home.copy()
            q[:7] = np.clip(home[:7] + rng.uniform(-ARM_NOISE, ARM_NOISE, 7), lo, hi)
            q[7:] = rng.uniform(0, FINGER_MAX)
            links = fk.link_poses(torch.from_numpy(q), links=CHAIN)
            chain = torch.stack([links[k][:3, 3] for k in CHAIN])
            if chain[1:, 2].min() >= MIN_LINK_Z:
                break
        else:
            raise RuntimeError("no valid robot configuration found")
        poses, floating = [], []
        for _ in range(len(OBJECTS[task])):
            for _ in range(max_tries):
                fl = bool(rng.random() < P_FLOAT)
                xy = (rng.uniform(*OBJ_X), rng.uniform(*OBJ_Y))
                if fl:
                    p, quat = (*xy, rng.uniform(*FLOAT_Z)), _uniform_quat(rng)
                else:
                    yaw = rng.uniform(-np.pi, np.pi)
                    p, quat = (*xy, CUBE_HALF_SIZE), np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
                pt = torch.tensor(p, dtype=torch.float64)
                if _segment_dist(pt, chain[:-1], chain[1:]).min() < MIN_OBJ_ROBOT_DIST:
                    continue
                if any(np.linalg.norm(np.subtract(p, o[:3])) < MIN_OBJ_OBJ_DIST for o in poses):
                    continue
                poses.append(np.concatenate([p, quat]))
                floating.append(fl)
                break
            else:
                raise RuntimeError("no valid object placement found")
        configs.append({"qpos": torch.tensor(q, dtype=torch.float32),
                        "obj_pose": torch.tensor(np.stack(poses), dtype=torch.float32),
                        "floating": torch.tensor(floating)})
    return configs


def apply_config(env, cfg):
    """Put the simulator into configuration `cfg` (no physics step: floating objects stay where they are)."""
    u = env.unwrapped
    u.agent.robot.set_qpos(cfg["qpos"][None])
    for name, pose in zip(OBJECTS[_task_id(u)], cfg["obj_pose"]):
        getattr(u, name).set_pose(Pose.create_from_pq(p=pose[None, :3], q=pose[None, 3:]))


INSIDE_FILL = 0.9      # a single part fills >= 90% of the view ...
INSIDE_DEPTH = 0.10    # ... with median depth < 10 cm: the camera is inside or touching it (arm swung into a ring
                       # camera). Such views show back faces / one blurry surface, so reconstruction skips them.


def view_valid(depth_mm, seg):
    """[R] bool for stacked views depth_mm [R,H,W] int16, seg [R,H,W] int8 (see INSIDE_FILL, INSIDE_DEPTH)."""
    s = seg.flatten(1).long() + 1  # shift IGNORE to 0
    counts = torch.stack([torch.bincount(x, minlength=128) for x in s])
    counts[:, 0] = 0
    top = counts.argmax(1)
    fill = counts.max(1).values / s.shape[1]
    d = depth_mm.flatten(1).float() / 1000.0
    near = torch.stack([d[r][s[r] == top[r]].median() if fill[r] > 0 else d.new_tensor(1e9) for r in range(len(d))])
    return ~((fill >= INSIDE_FILL) & (near < INSIDE_DEPTH))


def capture_configs(env, configs, splits, parts=None, cameras=None):
    """Capture every configuration and stack per camera (compact dtypes; see `load_captures` / `config_state`):
    {"part_names", "part_kinds", "part_visual" [P], "split" [R], "qpos" [R,9], "obj_pose" [R,K,7], "floating" [R,K],
     "part_T" [R,P,4,4], "cameras": {uid: {rgb [R,H,W,3] uint8, depth_mm [R,H,W] int16 (0 invalid),
     seg [R,H,W] int8 (-1 none), viewmat [R,4,4], K [3,3], valid [R] bool (see `view_valid`)}}}"""
    parts = parts or part_table(env)
    if len(parts) > 127:
        raise ValueError("seg is stored as int8")
    states = []
    for cfg in configs:
        apply_config(env, cfg)
        states.append(capture_state(env, parts, cameras))
    cams = {}
    for uid in states[0]["cameras"]:
        v = [s["cameras"][uid] for s in states]
        cams[uid] = {"rgb": torch.stack([c["rgb"] for c in v]),
                     "depth_mm": torch.stack([(c["depth"] * 1000).round().to(torch.int16) for c in v]),
                     "seg": torch.stack([c["seg"].to(torch.int8) for c in v]),
                     "viewmat": torch.stack([c["viewmat"] for c in v]),
                     "K": v[0]["K"]}
        cams[uid]["valid"] = view_valid(cams[uid]["depth_mm"], cams[uid]["seg"])
    return {"part_names": [p.name for p in parts], "part_kinds": [p.kind for p in parts],
            "part_visual": [p.visual for p in parts], "split": list(splits),
            "qpos": torch.stack([s["qpos"] for s in states]),
            "obj_pose": torch.stack([c["obj_pose"] for c in configs]),
            "floating": torch.stack([c["floating"] for c in configs]),
            "part_T": torch.stack([s["part_T"] for s in states]), "cameras": cams}


def save_captures(path, captures, meta):
    torch.save({**captures, "meta": meta}, path)


def load_captures(path):
    return torch.load(path, weights_only=True)


def config_state(captures, r, cameras=None):
    """Configuration r of stacked captures in the capture_state format (depth float32 m, seg int16)."""
    out = {}
    for uid, c in captures["cameras"].items():
        if cameras is None or uid in cameras:
            out[uid] = {"rgb": c["rgb"][r], "depth": c["depth_mm"][r].float() / 1000.0,
                        "seg": c["seg"][r].to(torch.int16), "viewmat": c["viewmat"][r], "K": c["K"],
                        "valid": bool(c["valid"][r])}
    return {"cameras": out, "part_names": captures["part_names"], "part_kinds": captures["part_kinds"],
            "part_T": captures["part_T"][r], "qpos": captures["qpos"][r]}


def view_counts(captures, min_pixels=50, split="train"):
    """{part name: number of valid (configuration, camera) views where it covers >= min_pixels} (B2 acceptance)."""
    rs = [r for r, s in enumerate(captures["split"]) if s == split]
    counts = dict.fromkeys(captures["part_names"], 0)
    for c in captures["cameras"].values():
        rv = [r for r in rs if c["valid"][r]]
        seg = c["seg"][rv].long()
        for p, name in enumerate(captures["part_names"]):
            counts[name] += int(((seg == p).flatten(1).sum(1) >= min_pixels).sum())
    return counts
