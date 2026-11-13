"""C5: checks of eval/metrics.py and eval/baselines.py. Each metric gets an input whose answer is known by hand."""
import math

import pytest
import torch

from gaussbridge import geometry as G
from gaussbridge.eval import metrics as M
from gaussbridge.eval.baselines import straight_line
from gaussbridge.model.dataset import MotionDataset
from gaussbridge.trajectory import decode_motion, known_mask
from motion_fakes import fake_episode

pytest.importorskip("pytorch_kinematics")
from gaussbridge.sim.kinematics import PandaFK

N, P = 128, 3


def line_tracks(offset=0.0, n=N, p=P):
    """Points moving 10 cm along x, shifted by `offset` metres along y."""
    t = torch.linspace(0, 0.1, n)
    tr = torch.zeros(n, p, 3)
    tr[..., 0] = t[:, None]
    tr[..., 1] = offset
    return tr


def test_ade_and_set_metrics_on_hand_made_tracks():
    a, b = line_tracks(), line_tracks(0.02)
    assert M.ade(a, a).item() == 0 and M.ade(a, b).item() == pytest.approx(0.02)
    refs = torch.stack([line_tracks(0.0), line_tracks(0.10)])                # two real solutions, 10 cm apart
    # references against themselves: everything matched, nothing invented (the C5 sanity check)
    assert M.coverage(refs, refs).item() == 1 and M.precision(refs, refs).item() == 1
    assert M.min_ade(refs, refs).item() == 0
    # samples near only one solution: precise but half the modes are missed
    one_mode = torch.stack([line_tracks(0.005), line_tracks(-0.005)])
    assert M.coverage(one_mode, refs).item() == 0.5 and M.precision(one_mode, refs).item() == 1
    # samples that are spread out but far from both: diverse yet useless
    noise = torch.stack([line_tracks(0.05), line_tracks(-0.06)])
    assert M.precision(noise, refs).item() == 0 and M.apd(noise).item() == pytest.approx(0.11)
    assert M.apd(refs[:1]).item() == 0


def test_jerk_of_known_curves():
    # float64 input: the formula is checked exactly. (float32 tracks carry a rounding floor of ~1e-4 m^2/s^6, about
    # 10^4 times below real planner motions, whose jerk is 1.2-8 m^2/s^6 on the StackCube dev episodes.)
    t = torch.linspace(0, 2.0, N, dtype=torch.float64)                        # 2 seconds
    tr = torch.zeros(N, 1, 3, dtype=torch.float64)
    tr[:, 0, 0] = 0.3 * t                                                     # constant velocity: no jerk
    assert M.mean_squared_jerk(tr, 2.0).item() == pytest.approx(0, abs=1e-6)
    tr[:, 0, 0] = t ** 3                                                      # x = t^3: third derivative is 6
    assert M.mean_squared_jerk(tr, 2.0).item() == pytest.approx(36, rel=1e-6)


def test_rotation_error_respects_cube_symmetry():
    R = G.quat_to_matrix(torch.nn.functional.normalize(torch.randn(5, 4), dim=-1))
    c, s = math.cos(math.radians(90)), math.sin(math.radians(90))
    Rz90 = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    assert (M.rotation_error_deg(R @ Rz90, R) < 1e-3).all()                   # turned 90 deg: looks identical
    torch.testing.assert_close(M.rotation_error_deg(R @ Rz90, R, symmetric=False), torch.full((5,), 90.0))
    c, s = math.cos(math.radians(10)), math.sin(math.radians(10))
    Rz10 = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    torch.testing.assert_close(M.rotation_error_deg(R @ Rz10, R), torch.full((5,), 10.0), atol=1e-3, rtol=0)


@pytest.fixture(scope="module")
def fk():
    return PandaFK("panda_wristcam")


def test_tracks_and_endpoints_on_an_episode(fk):
    ep = fake_episode(160)
    x = MotionDataset([ep]).x0                                                # [1, N, E, F]
    tr = M.motion_tracks(x, 2, fk)
    assert tr.shape == (1, N, 3, 3)
    d = decode_motion(x, 0.0)
    torch.testing.assert_close(tr[0, :, 0], fk.tcp_pose(d["qpos"][0])[:, :3, 3])  # point 0 = TCP
    torch.testing.assert_close(tr[0, :, 1:], d["obj_pos"][0, :, :2])              # points 1.. = objects
    err = M.endpoint_errors(x, x, 2, fk)
    assert all((v.abs() < 1e-3).all() for v in err.values()), err
    moved = x.clone()
    moved[:, -1, 1, 0] += 0.05 / 0.3                                          # cubeA end 5 cm further along x
    assert M.endpoint_errors(moved, x, 2, fk)["obj_pos_cm"][0, 0].item() == pytest.approx(5.0, abs=1e-3)


@pytest.mark.parametrize("goal", ["full", "objects", "none"])
def test_straight_line_baseline(goal):
    ds = MotionDataset([fake_episode(150, seed=s) for s in range(2)], endpoint_noise=False)
    x0, ent = ds.x0, ds.ent_mask.expand(2, 4)
    known = known_mask(ent, goal)
    x_known = torch.where(known[..., None], x0, torch.zeros_like(x0))
    x = straight_line(x_known, known, ent)
    assert torch.equal(x[known], x0[known])                                    # given cells exact
    assert (x[:, :, 3] == 0).all()                                             # padding
    free = ~known[:, :, 1:3]
    assert (x[:, :, 1:3, 9][free] == -1).all()                                 # never attached (given cells aside)
    pos = decode_motion(x, 0.0)["obj_pos"][:, :, :2]
    end = pos[:, -1] if goal != "none" else pos[:, 0]
    mid = pos[:, 64]
    torch.testing.assert_close(mid, (pos[:, 0] + end) / 2, atol=2e-3, rtol=0)  # half way at the middle step
    if goal != "full":                                                         # robot end unknown: stays at start
        torch.testing.assert_close(x[:, :, 0, :8], x0[:, :1, 0, :8].expand(-1, N, -1))
    R = G.rot6d_to_matrix(x[:, :, 1:3, 3:9])
    torch.testing.assert_close(R @ R.mT, torch.eye(3).expand_as(R), atol=1e-5, rtol=0)  # valid rotations


def test_harness_sanity_on_references(tmp_path):
    """C5 acceptance on synthetic data: references scored against themselves give coverage = precision = 1 and
    minADE = 0; the straight line reaches the objects' goal exactly but covers nothing in object-only mode."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("eval_motion", Path(__file__).parents[1] / "scripts/eval_motion.py")
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    for seed in (0, 1):
        for ref in range(3):
            ep = fake_episode(140 + 20 * ref, seed=10 * seed + ref)
            ep.robot_qpos[0] = fake_episode(140, seed=10 * seed).robot_qpos[0]   # references share the start ...
            ep.obj_pose[0] = fake_episode(140, seed=10 * seed).obj_pose[0]
            ep.meta = {"seed": seed, "robot_uid": "panda_wristcam"}
            ep.save(tmp_path / f"{seed:05d}_{ref:02d}.npz")
    ds, cases, episodes = em.load_cases(tmp_path)
    assert {k: len(v) for k, v in cases.items()} == {0: 3, 1: 3}
    res = em.evaluate(ds, cases, episodes, ["refs", "straight"], PandaFK("panda_wristcam"))
    for goal in em.GOALS:
        m = res["refs"][goal]["mean"]
        assert m["coverage"] == 1 and m["precision"] == 1 and m["min_ade_m"] == 0
    s = res["straight"]["objects"]["mean"]
    assert s["end_obj_pos_cm"] < 1e-3 and s["apd_m"] == 0
