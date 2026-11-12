"""B1 acceptance: capture environments, part table and geometric consistency of captures.
Needs ManiSkill + a Vulkan device (lavapipe works)."""
import math

import numpy as np
import pytest
import torch

pytest.importorskip("mani_skill")
import gymnasium as gym
from mani_skill.utils.structs.pose import Pose

from gaussbridge.sim.capture import IGNORE, capture_state, part_table, reprojection_agreement
from gaussbridge.sim.envs import OBJECTS, POLICY_CAMERAS, RINGS, capture_camera_names
from gaussbridge.sim.kinematics import PandaFK

pytestmark = pytest.mark.sim


def _env(env_id):
    env = gym.make(env_id, obs_mode="rgb+depth+segmentation", sim_backend="physx_cpu", render_backend="sapien_cpu")
    env.reset(seed=0)
    return env


@pytest.fixture(scope="module")
def stack_env():
    env = _env("StackCubeCapture-v1")
    yield env
    env.close()


def _total(results):
    agree = sum(a for r in results for a, _ in r.values())
    visible = sum(v for r in results for _, v in r.values())
    per_part = {}
    for r in results:
        for k, (a, v) in r.items():
            pa, pv = per_part.get(k, (0, 0))
            per_part[k] = (pa + a, pv + v)
    return agree / visible, {k: a / v for k, (a, v) in per_part.items() if v > 200}


@pytest.mark.parametrize("env_id,task", [("StackCubeCapture-v1", "StackCube-v1"), ("PickCubeCapture-v1", "PickCube-v1")])
def test_cameras_and_part_table(env_id, task):
    env = _env(env_id)
    parts = part_table(env)
    names = [p.name for p in parts]
    kinds = {p.name: p.kind for p in parts}
    assert all(kinds[o] == "object" for o in OBJECTS[task])
    assert kinds["table-workspace"] == kinds["ground"] == "background"
    assert {link.name for link in env.unwrapped.agent.robot.links} <= set(names)
    assert "goal_site" not in names  # hidden helper, never rendered

    state = capture_state(env, parts)
    cams = list(state["cameras"])
    assert cams[:len(POLICY_CAMERAS[task])] == POLICY_CAMERAS[task]  # policy cameras first
    assert set(cams[len(POLICY_CAMERAS[task]):]) == set(capture_camera_names(RINGS))
    wide = state["cameras"][capture_camera_names(RINGS)[0]]
    assert wide["rgb"].shape == (256, 256, 3) and wide["rgb"].dtype == torch.uint8
    assert wide["depth"].dtype == torch.float32 and wide["seg"].dtype == torch.int16
    seen = set(torch.unique(torch.cat([c["seg"].flatten() for c in state["cameras"].values()])).tolist())
    for o in OBJECTS[task]:  # every object is visible somewhere
        assert names.index(o) in seen
    assert seen <= set(range(len(parts))) | {IGNORE}
    env.close()


def test_reprojection_within_one_capture(stack_env):
    state = capture_state(stack_env)
    cams = list(state["cameras"])
    rng = np.random.default_rng(0)
    pairs = [(cams[i], cams[j]) for i, j in rng.integers(0, len(cams), (40, 2)) if i != j]
    total, per_part = _total([reprojection_agreement(state, a, state, b) for a, b in pairs])
    assert total > 0.995, total  # measured 0.9994
    assert min(per_part.values()) > 0.95, per_part  # a frame bug in one part would show here


def test_reprojection_after_moving_robot_and_cube(stack_env):
    """Points carried with each part's pose change must land on that part in the new configuration:
    this is the assumption articulated reconstruction (B4) relies on."""
    u = stack_env.unwrapped
    stack_env.reset(seed=1)
    s1 = capture_state(stack_env)
    rng = np.random.default_rng(1)
    q = u.agent.robot.get_qpos().clone()
    q[0, :7] += torch.tensor(rng.uniform(-0.5, 0.5, 7), dtype=q.dtype)
    u.agent.robot.set_qpos(q)
    yaw = 0.7
    u.cubeA.set_pose(Pose.create_from_pq(p=torch.tensor([[0.05, -0.08, 0.15]]),
                                         q=torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])))
    s2 = capture_state(stack_env)
    moved = (s2["part_T"][:, :3, 3] - s1["part_T"][:, :3, 3]).norm(dim=-1)
    assert moved[s1["part_names"].index("cubeA")] > 0.05 and moved[s1["part_names"].index("panda_hand")] > 0.05
    cams = list(s1["cameras"])
    pairs = [(cams[i], cams[j]) for i, j in rng.integers(0, len(cams), (40, 2))]
    total, per_part = _total([reprojection_agreement(s1, a, s2, b) for a, b in pairs])
    assert total > 0.995, total  # measured 0.9996
    assert min(per_part.values()) > 0.95, per_part


def test_hand_camera_from_fk_matches_capture(stack_env):
    state = capture_state(stack_env, cameras=["hand_camera"])
    V = PandaFK("panda_wristcam", dtype=torch.float64).hand_camera_viewmat(state["qpos"].double())
    torch.testing.assert_close(V.float(), state["cameras"]["hand_camera"]["viewmat"], atol=1e-5, rtol=0)
