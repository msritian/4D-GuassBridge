"""C6: checks of guidance/collision.py. Hand-made cases with known answers, plus the calibration on real planner
motions (physically valid, so their excess penetration must stay under the 5 mm reject threshold)."""
import itertools
from pathlib import Path

import pytest
import torch

from gaussbridge.guidance.collision import (
    SPHERE_DIR,
    SceneSpheres,
    box_grid,
    box_signed_distance,
    fit_spheres,
    load_spheres,
)
from gaussbridge.model.dataset import MotionDataset
from gaussbridge.trajectory import POS_CENTER, POS_SCALE
from motion_fakes import fake_episode

pytest.importorskip("pytorch_kinematics")
from gaussbridge.sim.kinematics import PandaFK

REPO = Path(__file__).parents[1]
H = torch.full((3,), 0.02)


def test_box_signed_distance():
    q = torch.tensor([[0.0, 0, 0], [0.0, 0, 0.03], [0.0, 0, 0.015], [0.03, 0.03, 0.0], [0.03, 0.03, 0.03]])
    expected = torch.tensor([-0.02, 0.01, -0.005, 2 ** 0.5 * 0.01, 3 ** 0.5 * 0.01])
    torch.testing.assert_close(box_signed_distance(q, H), expected)
    grid = box_grid(H)
    sd = box_signed_distance(grid, H)
    assert sd.max().abs() < 1e-7 and sd.min() == pytest.approx(-0.02)            # surface at 0, centre at -half


def test_fit_spheres_cover_their_points():
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(3000, 3, generator=g) * torch.tensor([0.1, 0.04, 0.02])
    c, r = fit_spheres(pts, 8)
    assert c.shape == (8, 3) and r.shape == (8,)
    assert ((torch.cdist(pts, c) - r).min(1).values <= 1e-6).all()            # every point inside some sphere


def test_sphere_files_exist_for_both_robots():
    for uid in ("panda", "panda_wristcam"):
        spheres = load_spheres(SPHERE_DIR / f"{uid}.json")
        assert "panda_hand" in spheres and "panda_leftfinger" in spheres
        assert all(c.shape == (32, 3) and (r > 0).all() for c, r in spheres.values())
    assert "camera_link" in load_spheres(SPHERE_DIR / "panda_wristcam.json")


@pytest.fixture(scope="module")
def stack():
    fk = PandaFK("panda_wristcam")
    return SceneSpheres.for_robot(fk, 2)


def _grid_with_objects(pos_a, pos_b):
    """A StackCube motion grid whose robot is the fake episode's and whose cubes follow the given paths [N, 3]."""
    x = MotionDataset([fake_episode(150)]).x0.clone()                            # [1, N, E, F]
    for slot, pos in ((1, pos_a), (2, pos_b)):
        x[0, :, slot, :3] = (pos - POS_CENTER) / POS_SCALE
        x[0, :, slot, 3:9] = torch.tensor([1.0, 0, 0, 0, 1, 0])                  # no rotation
    return x


def test_cube_passing_through_cube_and_table(stack):
    N = 128
    a = torch.tensor([0.10, 0.10, 0.02]).repeat(N, 1)
    b = torch.tensor([0.10, -0.10, 0.02]).repeat(N, 1)
    a[60:68, 1] = -0.09                                       # A jumps 1 cm into B for a few steps (overlap 3 cm in y)
    a[90:95, 2] = 0.01                                        # and later sinks 1 cm into the table
    pen = stack.penetration(_grid_with_objects(a, b))
    # 3 cm overlap: deeper than the half size (2 cm), where the grid gives 2 cm within half a grid spacing (2.5 mm)
    assert 0.0175 <= pen["obj_obj"][0, 63, 0, 1].item() <= 0.02 + 1e-6
    assert pen["obj_obj"][0, 30].max() == 0                                      # apart elsewhere
    assert pen["obj_table"][0, 92, 0].item() == pytest.approx(0.01, abs=1e-6)
    assert pen["obj_table"][0, 0].max() == 0                                     # resting at the start is fine


def test_overlap_measure_grows_with_overlap_and_catches_full_overlap(stack):
    N = 128
    b = torch.tensor([0.0, -0.10, 0.02]).repeat(N, 1)
    depths = []
    for overlap in (0.001, 0.005, 0.02, 0.03, 0.04):                            # 0.04 = same place
        a = torch.tensor([0.0, 0.10, 0.02]).repeat(N, 1)
        a[64, 1] = -0.10 + 0.04 - overlap
        depths.append(stack.penetration(_grid_with_objects(a, b))["obj_obj"][0, 64, 0, 1].item())
    assert depths[0] == pytest.approx(0.001, abs=1e-5) and depths[1] == pytest.approx(0.005, abs=1e-5)  # exact
    assert depths[2] == pytest.approx(0.02, abs=1e-5)
    assert all(d1 <= d2 + 0.0025 for d1, d2 in itertools.pairwise(depths))       # grows, up to half a grid step
    assert depths[-1] == pytest.approx(0.02, abs=1e-6)                           # full overlap caught


def test_resting_contact_at_the_endpoints_is_not_penetration(stack):
    """cubeA ends stacked on cubeB (touching): the endpoint correction keeps that at 0 excess."""
    N = 128
    b = torch.tensor([0.0, 0.0, 0.02]).repeat(N, 1)
    a = torch.tensor([0.0, 0.0, 0.0599]).repeat(N, 1)                          # 0.1 mm into B the whole time
    raw = stack.raw_penetration(_grid_with_objects(a, b))["obj_obj"]
    exc = stack.penetration(_grid_with_objects(a, b))["obj_obj"]
    assert raw.max() > 0 and exc.max() == 0


def test_energy_is_differentiable_and_points_apart(stack):
    """A shallow overlap (the case guidance acts on): the gradient is non-zero and a small step along it lowers the
    energy. (At a perfectly symmetric deep overlap, e.g. a grid point exactly at the other cube's centre, the gradient
    of that point is 0; samples never line up that exactly.)"""
    N = 128
    a = torch.tensor([0.10, 0.10, 0.02]).repeat(N, 1)
    b = torch.tensor([0.10, -0.10, 0.02]).repeat(N, 1)
    a[60:68] = torch.tensor([0.1037, -0.0640, 0.0213])        # 4 mm into B along y, slightly off-axis
    x = _grid_with_objects(a, b).requires_grad_(True)
    e = stack.energy(x)
    e.sum().backward()
    assert e.item() > 0 and torch.isfinite(x.grad).all() and x.grad.norm() > 0
    x2 = (x - 1e-4 * x.grad / x.grad.norm()).detach()                            # small step down the gradient
    assert stack.energy(x2).item() < e.item()


def test_planner_motions_score_under_the_threshold():
    """Calibration: the 24 dev planner motions (physically valid) must have max excess penetration < 5 mm
    (measured 2026-11-13: StackCube max 2.5 mm, finger vs cubeB while stacking; PickCube 0)."""
    pytest.importorskip("h5py")
    from gaussbridge.sim.extract import load_shard

    for task, uid in (("StackCube-v1", "panda_wristcam"), ("PickCube-v1", "panda")):
        episodes = load_shard(REPO / f"artifacts/raw/{task}/dev")
        ds = MotionDataset(episodes)
        scene = SceneSpheres.for_robot(PandaFK(uid), ds.n_objects)
        with torch.no_grad():
            worst = scene.max_penetration(ds.x0)
        assert (worst < 0.005).all(), (task, (worst * 1000).tolist())


def test_link_sphere_vs_cube_exact_depth():
    """One 1 cm sphere at panda_link5's origin; cubeA placed so the sphere sinks exactly 3 mm into its top face (and
    1 mm further at step 70). Checks the link-object geometry (FK placement, box frame, radius) to 1e-6 m."""
    fk = PandaFK("panda_wristcam")
    scene = SceneSpheres(fk, {"panda_link5": (torch.zeros(1, 3), torch.tensor([0.01]))}, n_objects=2)
    x = MotionDataset([fake_episode(150)]).x0.clone()
    from gaussbridge.trajectory import decode_motion
    link5 = fk.link_poses(decode_motion(x, 0.0)["qpos"], links=["panda_link5"])["panda_link5"][0, :, :3, 3]  # [N, 3]
    a = link5 - torch.tensor([0.0, 0.0, 0.02 + 0.01 - 0.003])     # cube top 3 mm above the sphere's bottom
    a[70, 2] += 0.001                                             # step 70: cube 1 mm higher -> 4 mm
    b = torch.tensor([0.3, 0.3, 0.02]).repeat(128, 1)
    for slot, pos in ((1, a), (2, b)):
        x[0, :, slot, :3] = (pos - POS_CENTER) / POS_SCALE
        x[0, :, slot, 3:9] = torch.tensor([1.0, 0, 0, 0, 1, 0])
    raw = scene.raw_penetration(x)["link_obj"][0, :, 0]           # [N, K]
    torch.testing.assert_close(raw[10, 0], torch.tensor(0.003), atol=1e-6, rtol=0)
    torch.testing.assert_close(raw[70, 0], torch.tensor(0.004), atol=1e-6, rtol=0)
    assert raw[:, 1].max() == 0                                   # cubeB far away
    assert scene.penetration(x)["link_obj"][0, 70, 0, 0].item() == pytest.approx(0.001, abs=1e-6)  # excess
