"""C1, A2, A3 checked against ManiSkill and the A1 planner. Needs ManiSkill + mplib (Linux): run in docker/Dockerfile.cpu."""
import dataclasses
import math

import numpy as np
import pytest
import torch

pytest.importorskip("mani_skill")
import gymnasium as gym  # noqa: E402

from gaussbridge import geometry as G  # noqa: E402
from gaussbridge.sim.extract import load_shard  # noqa: E402
from gaussbridge.sim.planner_gen import generate  # noqa: E402
from gaussbridge.sim.replay import make_env, replay  # noqa: E402
from gaussbridge.trajectory import Episode, episode_to_motion, motion_to_episode  # noqa: E402

OBJECTS = {"StackCube-v1": ["cubeA", "cubeB"], "PickCube-v1": ["cube"]}


def _make_env(env_id, **kw):
    return gym.make(env_id, sim_backend="physx_cpu", render_backend="sapien_cpu", **kw)


def test_camera_helpers_match_maniskill():
    env = _make_env("StackCube-v1", obs_mode="rgb+segmentation")
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    param = obs["sensor_param"]
    V = G.look_at_viewmat([0.3, 0, 0.6], [-0.1, 0, 0.1])  # base_camera as in CLAUDE.md §5
    K = G.fov_to_K(math.pi / 2, 128, 128)
    torch.testing.assert_close(V[:3], param["base_camera"]["extrinsic_cv"][0], atol=1e-5, rtol=0)
    torch.testing.assert_close(K, param["base_camera"]["intrinsic_cv"][0])

    cam_link = next(l for l in u.agent.robot.links if l.name == "camera_link")
    V_hand = G.sapien_pose_to_viewmat(G.pose7_to_matrix(cam_link.pose.raw_pose[0]))
    torch.testing.assert_close(V_hand[:3], param["hand_camera"]["extrinsic_cv"][0], atol=1e-5, rtol=0)

    seg = obs["sensor_data"]["base_camera"]["segmentation"][0, ..., 0]
    for name in OBJECTS["StackCube-v1"]:
        seg_id = next(i for i, o in u.segmentation_id_map.items() if getattr(o, "name", None) == name)
        uv, _ = G.project(getattr(u, name).pose.p, V, K)
        col, row = uv[0].floor().long().tolist()
        assert seg[row, col] == seg_id, name  # cube centre projects onto the cube's own pixels
    env.close()


@pytest.mark.parametrize("env_id", ["StackCube-v1", "PickCube-v1"])
def test_planner_demo_survives_motion_round_trip_and_replays(tmp_path, env_id):
    generate(env_id, seed_start=0, num_seeds=1, refs_per_seed=1, out=tmp_path / "shard")
    load_shard(tmp_path / "shard")[0].save(tmp_path / "ep.npz")  # A2 extraction
    ep = Episode.load(tmp_path / "ep.npz")

    m = episode_to_motion(ep)
    assert m["x"].shape == (128, 4, 10) and m["ent_mask"].sum() == 1 + len(ep.obj_names)
    ep2 = motion_to_episode(m["x"], m["log_dur"], ep.obj_names, ep.base_pose, ep.meta)
    assert ep2.num_steps == ep.num_steps
    for i in (0, -1):  # start and end states survive exactly (up to float32)
        np.testing.assert_allclose(ep2.robot_qpos[i, :8], ep.robot_qpos[i, :8], atol=1e-5)
        np.testing.assert_allclose(ep2.obj_pose[i, :, :3], ep.obj_pose[i, :, :3], atol=1e-5)
        np.testing.assert_allclose(G.quat_to_matrix(torch.as_tensor(ep2.obj_pose[i, :, 3:])),
                                   G.quat_to_matrix(torch.as_tensor(ep.obj_pose[i, :, 3:])), atol=1e-5)
    assert np.abs(ep2.robot_qpos[:, :7] - ep.robot_qpos[:, :7]).max() < 0.01  # whole path, rad

    env = make_env(env_id)
    r = replay(ep2, env)  # A3
    assert r["success"] and r["obj_end_err"].max() < 0.01, r  # 1 cm: replay settles cubes up to ~6 mm apart
    n = ep.num_steps // 2 + 1  # the check must also be able to fail: stopping halfway solves nothing
    half = dataclasses.replace(ep, robot_qpos=ep.robot_qpos[:n], gripper_cmd=ep.gripper_cmd[:n],
                               obj_pose=ep.obj_pose[:n], attached=ep.attached[:n])
    assert not replay(half, env)["success"]
    env.close()
