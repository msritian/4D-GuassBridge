"""B2: random asset-mode configurations (sampling only; no rendering, no Vulkan needed)."""
import numpy as np
import pytest
import torch

pytest.importorskip("mani_skill")

from gaussbridge.sim.capture import (
    CHAIN,
    CUBE_HALF_SIZE,
    FINGER_MAX,
    FLOAT_Z,
    LIMIT_MARGIN,
    MIN_LINK_Z,
    MIN_OBJ_OBJ_DIST,
    MIN_OBJ_ROBOT_DIST,
    _segment_dist,
    random_configs,
    view_valid,
)
from gaussbridge.sim.envs import HOME_QPOS, OBJECTS
from gaussbridge.sim.kinematics import PandaFK
from gaussbridge.trajectory import Q_HI, Q_LO


@pytest.mark.parametrize("task,robot", [("StackCube-v1", "panda_wristcam"), ("PickCube-v1", "panda")])
def test_random_configs_valid(task, robot):
    cfgs = random_configs(task, 40, 0)
    fk = PandaFK(robot, dtype=torch.float64)
    home = torch.tensor(HOME_QPOS[robot], dtype=torch.float32)
    for c in cfgs:
        q = c["qpos"]
        assert ((q[:7] >= Q_LO + LIMIT_MARGIN - 1e-6) & (q[:7] <= Q_HI - LIMIT_MARGIN + 1e-6)).all()
        assert ((q[:7] - home[:7]).abs() <= 0.6 + 1e-6).all()
        assert 0 <= q[7] <= FINGER_MAX and q[7] == q[8]
        links = fk.link_poses(q.double(), links=CHAIN)
        chain = torch.stack([links[k][:3, 3] for k in CHAIN])
        assert chain[1:, 2].min() >= MIN_LINK_Z
        assert c["obj_pose"].shape == (len(OBJECTS[task]), 7)
        torch.testing.assert_close(c["obj_pose"][:, 3:].norm(dim=-1), torch.ones(len(OBJECTS[task])))
        for p, fl in zip(c["obj_pose"], c["floating"]):
            assert _segment_dist(p[:3].double(), chain[:-1], chain[1:]).min() >= MIN_OBJ_ROBOT_DIST
            if fl:
                assert FLOAT_Z[0] <= p[2] <= FLOAT_Z[1]
            else:  # resting: on the table, rotated about z only
                assert abs(p[2] - CUBE_HALF_SIZE) < 1e-6 and p[4].abs() < 1e-6 and p[5].abs() < 1e-6
        if len(c["obj_pose"]) > 1:
            assert (c["obj_pose"][0, :3] - c["obj_pose"][1, :3]).norm() >= MIN_OBJ_OBJ_DIST
    floating = torch.cat([c["floating"] for c in cfgs]).float().mean()
    assert 0.3 < floating < 0.7


def test_random_configs_deterministic_and_streams_differ():
    a, b = random_configs("StackCube-v1", 3, [0, 0]), random_configs("StackCube-v1", 3, [0, 0])
    held = random_configs("StackCube-v1", 3, [0, 1])
    for x, y in zip(a, b):
        assert torch.equal(x["qpos"], y["qpos"]) and torch.equal(x["obj_pose"], y["obj_pose"])
    assert not any(np.allclose(x["qpos"], h["qpos"]) for x in a for h in held)


def test_view_valid():
    depth = torch.full((3, 8, 8), 500, dtype=torch.int16)  # 0.5 m
    seg = torch.full((3, 8, 8), -1, dtype=torch.int8)
    seg[0] = 4                      # one part fills the view at 0.5 m: valid (just a close-up)
    seg[1] = 4
    depth[1] = 30                   # one part fills the view at 3 cm: camera inside it
    seg[2, :4] = 4
    depth[2] = 30                   # half the view at 3 cm: valid (e.g. fingers in the hand camera)
    assert view_valid(depth, seg).tolist() == [True, False, True]
