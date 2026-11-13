"""C3: checks of model/dataset.py. Pure torch on synthetic episodes (no simulator), runs anywhere in seconds.

Each test checks one promise from the module docstring: sample layout, goal-mode masks and frequencies, endpoint
noise (sizes, only on the given copy, target untouched), placeholder cube shapes, duration normalisation."""
import math

import numpy as np
import pytest
import torch

from gaussbridge.model.dataset import (
    CUBE_HALF_SIZE,
    GOAL_FULL,
    GOAL_NONE,
    GOAL_OBJECTS,
    NOISE_JOINT,
    NOISE_OBJ_POS,
    NOISE_OBJ_ROT,
    MotionDataset,
    box_shape_points,
    duration_stats,
)
from gaussbridge.trajectory import K_MAX, N_STEPS, NUM_FEATURES, decode_motion, episode_to_motion
from motion_fakes import fake_episode


@pytest.fixture
def stack_episodes():
    return [fake_episode(n, seed=i) for i, n in enumerate([140, 165, 190, 210, 250])]


def test_sample_layout(stack_episodes):
    ds = MotionDataset(stack_episodes)
    s = ds[2]
    assert s["x0"].shape == s["x_known"].shape == (N_STEPS, 1 + K_MAX, NUM_FEATURES)
    assert s["known"].shape == (N_STEPS, 1 + K_MAX) and s["known"].dtype == torch.bool
    assert s["ent_mask"].tolist() == [True, True, True, False]          # robot, cubeA, cubeB, padding
    assert s["shapes"].shape == (K_MAX, 512, 6)
    torch.testing.assert_close(s["x0"], episode_to_motion(stack_episodes[2])["x"])  # target = the C1 grid


def test_goal_modes_mark_the_right_cells(stack_episodes):
    ds = MotionDataset(stack_episodes)
    want = {  # which slots are known on the end row; the start row is always [robot, cubeA, cubeB]
        GOAL_FULL: [True, True, True, False],
        GOAL_OBJECTS: [False, True, True, False],
        GOAL_NONE: [False, False, False, False],
    }
    seen = set()
    torch.manual_seed(0)
    for _ in range(200):
        s = ds[0]
        seen.add(s["goal"])
        known = s["known"]
        assert known[0].tolist() == [True, True, True, False]
        assert known[-1].tolist() == want[s["goal"]]
        assert not known[1:-1].any()                                     # the middle is never given
        assert not known[:, 3].any()                                     # padding slot is never given
    assert seen == {GOAL_FULL, GOAL_OBJECTS, GOAL_NONE}


def test_goal_mode_frequencies(stack_episodes):
    ds = MotionDataset(stack_episodes)
    torch.manual_seed(1)
    goals = torch.tensor([ds.draw_goal() for _ in range(20000)])
    freq = [(goals == g).float().mean().item() for g in (GOAL_NONE, GOAL_OBJECTS, GOAL_FULL)]
    assert np.allclose(freq, [0.1, 0.3, 0.6], atol=0.015), freq


def test_x_known_is_zero_outside_known_and_exact_without_noise(stack_episodes):
    ds = MotionDataset(stack_episodes, endpoint_noise=False)
    for i in range(len(ds)):
        s = ds[i]
        k = s["known"][..., None].expand_as(s["x0"])
        assert (s["x_known"][~k] == 0).all()
        assert torch.equal(s["x_known"][k], s["x0"][k])


def test_endpoint_noise_sizes_and_target_untouched(stack_episodes):
    """Noise goes only into the given copy, with the documented sizes; x0 (the target) never changes."""
    ds = MotionDataset(stack_episodes)
    clean = ds.x0.clone()
    torch.manual_seed(2)
    d_joint, d_pos, d_rot = [], [], []
    for _ in range(300):
        s = ds[1]
        x0, xk, known = s["x0"], s["x_known"], s["known"]
        assert torch.equal(x0, clean[1])
        a, b = decode_motion(x0, 0.0), decode_motion(xk, 0.0)            # model units -> radians / metres
        robot_rows = known[:, 0]
        d_joint.append((b["qpos"][robot_rows, :7] - a["qpos"][robot_rows, :7]).flatten())
        # finger, gripper command and attached flag are given exactly
        assert torch.equal(xk[robot_rows, 0, 7:9], x0[robot_rows, 0, 7:9])
        assert torch.equal(xk[known[:, 1], 1, 9], x0[known[:, 1], 1, 9])
        for slot in (1, 2):
            rows = known[:, slot]
            d_pos.append((b["obj_pos"][rows, slot - 1] - a["obj_pos"][rows, slot - 1]).flatten())
            R_rel = b["obj_rot"][rows, slot - 1] @ a["obj_rot"][rows, slot - 1].mT
            cos = ((R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
            d_rot.append(torch.acos(cos))
    d_joint, d_pos, d_rot = torch.cat(d_joint), torch.cat(d_pos), torch.cat(d_rot)
    assert abs(d_joint.std().item() - NOISE_JOINT) < 0.1 * NOISE_JOINT
    assert abs(d_pos.std().item() - NOISE_OBJ_POS) < 0.1 * NOISE_OBJ_POS
    # angle of a rotation with angle ~ N(0, s) about a random axis: |angle|, whose RMS is s
    assert abs(d_rot.pow(2).mean().sqrt().item() - NOISE_OBJ_ROT) < 0.1 * NOISE_OBJ_ROT


def test_box_shape_points_lie_on_the_cube_surface():
    pts = box_shape_points(rgb=(0.0, 1.0, 0.0), n=6000)
    xyz, rgb = pts[:, :3], pts[:, 3:]
    assert torch.allclose(xyz.abs().max(-1).values, torch.full((6000,), CUBE_HALF_SIZE))  # on a face
    assert (xyz.abs() <= CUBE_HALF_SIZE + 1e-7).all()                                      # not outside
    assert (rgb == torch.tensor([0.0, 1.0, 0.0])).all()
    face = xyz.abs().argmax(-1) * 2 + (xyz.gather(-1, xyz.abs().argmax(-1, keepdim=True))[:, 0] > 0)
    counts = torch.bincount(face, minlength=6).float()
    assert (counts / 6000 - 1 / 6).abs().max() < 0.02                                       # faces equally covered


def test_shapes_follow_object_slots(stack_episodes):
    shapes = MotionDataset(stack_episodes)[0]["shapes"]
    assert (shapes[0, :, 3:] == torch.tensor([1.0, 0, 0])).all()        # cubeA red
    assert (shapes[1, :, 3:] == torch.tensor([0, 1.0, 0])).all()        # cubeB green
    assert (shapes[2] == 0).all()                                        # unused slot


def test_pick_cube_has_one_object_and_padding():
    ds = MotionDataset([fake_episode(150, obj_names=("cube",), seed=s) for s in range(3)])
    s = ds[0]
    assert s["ent_mask"].tolist() == [True, True, False, False]
    assert not s["known"][:, 2:].any()
    assert (s["x0"][:, 2:] == 0).all()                                   # padded slots hold zeros


def test_duration_normalisation(stack_episodes):
    ds = MotionDataset(stack_episodes)
    assert abs(ds.log_dur.mean().item()) < 1e-5 and abs(ds.log_dur.std().item() - 1) < 1e-5
    stats = duration_stats(stack_episodes)
    assert math.isclose(stats["mu"], np.mean([math.log(e.num_steps / 20) for e in stack_episodes]))
    # validation data must reuse the training stats, so the same duration maps to the same number
    val = MotionDataset([stack_episodes[0]], dur_stats=stats)
    assert torch.isclose(val.log_dur[0], ds.log_dur[0])


def test_mixing_tasks_is_rejected(stack_episodes):
    with pytest.raises(ValueError):
        MotionDataset(stack_episodes + [fake_episode(150, obj_names=("cube",))])


def test_tensors_follow_the_input_device():
    """Masks built inside helpers must live on the input's device (a CPU-only mask crashed the first GPU memory
    check, job 8539267). The 'meta' device has shapes but no data, so this catches it without a GPU."""
    from gaussbridge.trajectory import known_mask

    ent_mask = torch.tensor([True, True, True, False], device="meta").expand(2, 4)
    for goal in ("full", "objects", "none"):
        assert known_mask(ent_mask, goal).device.type == "meta"


def test_make_batch_matches_getitem_and_is_reproducible(stack_episodes):
    """make_batch (vectorised, used for training at scale) builds the same samples as __getitem__: same masks per
    goal mode, same goal frequencies, noise only on the given copy, and identical batches from identical seeds."""
    from gaussbridge.trajectory import known_mask

    ds = MotionDataset(stack_episodes)
    idx = torch.randint(len(ds), (4000,), generator=torch.Generator().manual_seed(0))
    b = ds.make_batch(idx, torch.Generator().manual_seed(1))
    for g, name in ((GOAL_FULL, "full"), (GOAL_OBJECTS, "objects"), (GOAL_NONE, "none")):
        sel = b["goal"] == g
        assert torch.equal(b["known"][sel], known_mask(ds.ent_mask, name).expand(int(sel.sum()), -1, -1))
    freq = [(b["goal"] == g).float().mean().item() for g in (GOAL_NONE, GOAL_OBJECTS, GOAL_FULL)]
    assert np.allclose(freq, [0.1, 0.3, 0.6], atol=0.025), freq
    assert torch.equal(b["x0"], ds.x0[idx]) and torch.equal(b["log_dur"], ds.log_dur[idx])
    assert (b["x_known"][~b["known"]] == 0).all()
    k = b["known"][..., None].expand_as(b["x0"])
    assert not torch.equal(b["x_known"][k], b["x0"][k])                    # noise is on ...
    assert (b["x_known"][..., 0, 7:9][b["known"][..., 0]] == b["x0"][..., 0, 7:9][b["known"][..., 0]]).all()
    quiet = MotionDataset(stack_episodes, endpoint_noise=False).make_batch(idx[:50], torch.Generator().manual_seed(1))
    assert torch.equal(quiet["x_known"][quiet["known"]], quiet["x0"][quiet["known"]])  # ... and can be off
    again = ds.make_batch(idx, torch.Generator().manual_seed(1))
    assert all(torch.equal(b[key], again[key]) for key in b)               # same seed -> same batch
    assert b["shapes"].shape == (4000, K_MAX, 512, 6)
