"""B1: parts and multi-view captures (CLAUDE.md §7.1-7.2).

A part = one rigid thing that gets its own Gaussians: every robot link, every task object and each static
background entity (table, ground). The part table is built from ManiSkill's segmentation_id_map, never hard-coded.
"""
import dataclasses

import torch

from gaussbridge import geometry as G
from gaussbridge.sim.envs import OBJECTS, POLICY_CAMERAS

IGNORE = -1  # part index for pixels that belong to no part (sky / empty)


def _task_id(u):
    """'StackCube-v1' for both StackCube-v1 and StackCubeCapture-v1 envs."""
    return getattr(u, "task_id", None) or u.spec.id


@dataclasses.dataclass(frozen=True)
class Part:
    name: str
    kind: str      # "robot" | "object" | "background"
    seg_id: int    # ManiSkill segmentation id


def part_table(env):
    """Parts of the scene, ordered by segmentation id. Hidden helpers (PickCube's goal_site) are excluded.
    Robot links without visual geometry (panda_hand_tcp, ...) are kept so link names stay complete; they simply
    have no pixels."""
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
        parts.append(Part(name, kind, int(seg_id)))
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
