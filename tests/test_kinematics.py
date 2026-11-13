"""C2 acceptance: PandaFK matches ManiSkill's link poses (< 1e-5 m) for random configurations of both robots,
and the hand-camera viewmat matches ManiSkill's extrinsic. Needs ManiSkill + a Vulkan device (lavapipe works)."""
import numpy as np
import pytest
import torch

from gaussbridge import geometry as G
from gaussbridge.sim.kinematics import BASE_POSE, PandaFK

pytest.importorskip("mani_skill")
import gymnasium as gym
import mani_skill.envs  # noqa: F401

Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0])
Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04])
ROBOT_OF_TASK = {"StackCube-v1": "panda_wristcam", "PickCube-v1": "panda"}


def _random_qpos(n, seed):
    return np.random.default_rng(seed).uniform(Q_LO, Q_HI, size=(n, 9)).astype(np.float32)


@pytest.mark.sim
@pytest.mark.parametrize("env_id", ["StackCube-v1", "PickCube-v1"])
def test_fk_matches_simulator_links(env_id):
    env = gym.make(env_id, obs_mode="none", sim_backend="physx_cpu", render_backend="sapien_cpu")
    env.reset(seed=0)
    robot = env.unwrapped.agent.robot
    assert env.unwrapped.agent.uid == ROBOT_OF_TASK[env_id]
    fk = PandaFK(ROBOT_OF_TASK[env_id], dtype=torch.float64)
    torch.testing.assert_close(robot.pose.raw_pose[0].double(), torch.tensor(BASE_POSE, dtype=torch.float64))

    qs = _random_qpos(50, seed=1)
    pred = fk.link_poses(torch.from_numpy(qs).double())  # all 50 configurations in one batch
    names = [link.name for link in robot.links]
    assert set(names) <= set(pred), set(names) - set(pred)
    pos_err = rot_err = 0.0
    for i, q in enumerate(qs):
        robot.set_qpos(torch.from_numpy(q)[None])
        for link in robot.links:
            T_sim = G.pose7_to_matrix(link.pose.raw_pose[0].double())
            T_fk = pred[link.name][i]
            pos_err = max(pos_err, (T_sim[:3, 3] - T_fk[:3, 3]).norm().item())
            rot_err = max(rot_err, (T_sim[:3, :3] - T_fk[:3, :3]).abs().max().item())
    env.close()
    assert pos_err < 1e-5 and rot_err < 1e-5, (pos_err, rot_err)


@pytest.mark.sim
def test_hand_camera_viewmat_matches_maniskill_extrinsic():
    env = gym.make("StackCube-v1", obs_mode="rgb", sim_backend="physx_cpu", render_backend="sapien_cpu")
    env.reset(seed=0)
    u = env.unwrapped
    fk = PandaFK("panda_wristcam", dtype=torch.float64)
    qs = _random_qpos(5, seed=2)
    V = fk.hand_camera_viewmat(torch.from_numpy(qs).double())
    for i, q in enumerate(qs):
        u.agent.robot.set_qpos(torch.from_numpy(q)[None])
        extrinsic = u.get_obs()["sensor_param"]["hand_camera"]["extrinsic_cv"][0].double()  # [3, 4]
        torch.testing.assert_close(V[i, :3], extrinsic, atol=1e-5, rtol=0)
    env.close()


def test_batch_shapes_base_pose_and_gradients():
    fk = PandaFK("panda_wristcam")
    q = torch.from_numpy(_random_qpos(6, seed=3)).reshape(2, 3, 9)
    assert fk.tcp_pose(q).shape == (2, 3, 4, 4)
    assert fk.hand_camera_viewmat(q).shape == (2, 3, 4, 4)
    poses = fk.link_poses(q, links=["panda_hand", "panda_leftfinger"])
    assert set(poses) == {"panda_hand", "panda_leftfinger"}

    shifted = torch.tensor([0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0])  # moving the base moves every link with it
    d = fk.tcp_pose(q, base_pose=shifted)[..., :3, 3] - fk.tcp_pose(q)[..., :3, 3]
    torch.testing.assert_close(d, (shifted[:3] - torch.tensor(BASE_POSE[:3])).expand_as(d))

    q.requires_grad_(True)  # guidance (C6) differentiates link poses w.r.t. joint angles
    fk.tcp_pose(q)[..., :3, 3].sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad[..., :7].abs().sum() > 0

    with pytest.raises(ValueError):
        PandaFK("panda").hand_camera_viewmat(q.detach())
