"""C6: checks of guidance/attach.py. Real planner motions must pass the projection almost unchanged (they are physically
consistent); hand-made violations must be repaired or rejected exactly as the rules say."""
from pathlib import Path

import pytest
import torch

from gaussbridge import geometry as G
from gaussbridge.guidance.attach import attach_energy, longest_run, project_attachment
from gaussbridge.model.dataset import MotionDataset
from gaussbridge.trajectory import POS_SCALE, decode_motion, known_mask

pytest.importorskip("pytorch_kinematics")
pytest.importorskip("h5py")
from gaussbridge.sim.extract import load_shard
from gaussbridge.sim.kinematics import PandaFK

REPO = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def stack():
    ds = MotionDataset(load_shard(REPO / "artifacts/raw/StackCube-v1/dev"))
    return ds, PandaFK("panda_wristcam")


def _known(ds, goal="full"):
    return known_mask(ds.ent_mask.expand(len(ds.x0), -1), goal)


def _tcp_frame_positions(x, fk, k=0):
    from gaussbridge.guidance.attach import _object_poses, _tcp_poses

    d = decode_motion(x, 0.0)
    return (G.se3_inverse(_tcp_poses(d["qpos"], fk))[:, :, None] @ _object_poses(d, 2))[:, :, k, :3, 3]


def test_longest_run():
    t = torch.tensor
    assert longest_run(t([False, True, True, False, True, True, True])) == (4, 6)
    assert longest_run(t([True, False])) == (0, 0) and longest_run(t([False, False])) is None


@pytest.mark.parametrize("task,uid", [("StackCube-v1", "panda_wristcam"), ("PickCube-v1", "panda")])
def test_planner_motions_pass_almost_unchanged(task, uid):
    """Calibration (2026-11-13): all 24 dev motions accepted; projection moves objects by < 0.2 mm on average;
    release error <= 7.9 mm (StackCube: cubeA is nudged by cubeB while being placed), < 0.02 mm (PickCube)."""
    ds = MotionDataset(load_shard(REPO / f"artifacts/raw/{task}/dev"))
    fk = PandaFK(uid)
    for goal in ("full", "objects"):
        xp, ok, info = project_attachment(ds.x0, _known(ds, goal), ds.n_objects, fk)
        assert ok.all(), goal
        a, b = decode_motion(ds.x0, 0.0), decode_motion(xp, 0.0)
        moved = (a["obj_pos"][..., :ds.n_objects, :] - b["obj_pos"][..., :ds.n_objects, :]).norm(dim=-1)
        assert moved.mean() < 0.0005 and moved.max() < 0.01, (moved.mean(), moved.max())
        err = info["release_err_m"][info["held"]]
        assert (err < 0.01).all()
        assert torch.equal(xp[:, 0], ds.x0[:, 0])                                          # start untouched
        known = _known(ds, goal)
        assert torch.equal(xp[:, -1][known[:, -1]], ds.x0[:, -1][known[:, -1]])            # given end bit-exact


def test_projection_makes_the_carry_rigid(stack):
    """Cube A drifts 2 cm in the hand during the carry: after projection its position in the TCP frame is constant
    over the hold, and the attachment energy drops."""
    ds, fk = stack
    x = ds.x0[:1].clone()
    n0, n1 = longest_run(decode_motion(x, 0.0)["attached"][0, :, 0] > 0)
    drift = torch.linspace(0, 0.02, n1 - n0 - 10)                                          # end before release
    x[0, n0 + 5:n1 - 5, 1, 0] += drift / POS_SCALE
    e_before = attach_energy(x, 2, fk)
    xp, _, _ = project_attachment(x, _known(ds)[:1], 2, fk)
    r = _tcp_frame_positions(xp, fk)[0, n0 + 2:n1 + 1]
    assert (r - r[0]).norm(dim=-1).max() < 1e-5                                            # rigid in the hand
    assert attach_energy(xp, 2, fk) < e_before / 10
    flags = decode_motion(xp, 0.0)["attached"][0, :, 0]
    assert (flags[n0:n1 + 1] == 1).all() and (flags[:n0] == -1).all()


def test_reject_when_release_misses_the_goal(stack):
    ds, fk = stack
    x = ds.x0[:1].clone()
    goal = x[0, -1, 1, :3].clone()
    x[0, -1, 1, 0] += 0.02 / POS_SCALE                                 # the given goal is 2 cm from where it is released
    _, ok, info = project_attachment(x, _known(ds)[:1], 2, fk)
    assert not ok[0] and info["release_err_m"][0, 0] > 0.015
    x[0, -1, 1, :3] = goal
    assert project_attachment(x, _known(ds)[:1], 2, fk)[1][0]                             # back on goal: accepted


def test_object_that_must_move_but_is_never_held_is_rejected(stack):
    ds, fk = stack
    x = ds.x0[:1].clone()
    x[0, :, 1, 9] = -1.0                                                                   # model forgot the grasp
    xp, ok, info = project_attachment(x, _known(ds)[:1], 2, fk)
    assert not ok[0] and not info["held"][0, 0]
    # with an unknown end (forward-only mode) there is nothing to check: the cube just stays at its start
    xp, ok, _ = project_attachment(x, _known(ds, "none")[:1], 2, fk)
    assert ok[0]
    pos = decode_motion(xp, 0.0)["obj_pos"][0, :, 0]
    assert (pos - pos[0]).norm(dim=-1).max() < 1e-6


def test_unheld_resting_object_stays_put(stack):
    """cubeB is never held in StackCube: after projection it is exactly at its start until the given end."""
    ds, fk = stack
    xp, ok, info = project_attachment(ds.x0, _known(ds), 2, fk)
    pos = decode_motion(xp, 0.0)["obj_pos"][:, :-1, 1]
    assert ok.all() and not info["held"][:, 1].any()
    assert (pos - pos[:, :1]).norm(dim=-1).max() < 1e-6


def test_attach_energy_is_differentiable(stack):
    ds, fk = stack
    x = ds.x0[:2].clone().requires_grad_(True)
    e = attach_energy(x, 2, fk)
    e.sum().backward()
    assert e.shape == (2,) and torch.isfinite(x.grad).all() and x.grad.norm() > 0
