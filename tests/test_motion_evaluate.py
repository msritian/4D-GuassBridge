"""C3: model/evaluate.py reports errors in the right physical units (known offsets in, known numbers out)."""
import math

import torch

from gaussbridge import geometry as G
from gaussbridge.model.dataset import MotionDataset
from gaussbridge.model.evaluate import motion_errors
from gaussbridge.trajectory import POS_SCALE, Q_HI, Q_LO
from motion_fakes import fake_episode


def test_known_offsets_give_known_errors():
    ds = MotionDataset([fake_episode(150, seed=0), fake_episode(180, seed=1)])
    x0, dur = ds.x0, ds.log_dur
    same = motion_errors(x0, dur, x0, dur, ds.n_objects, ds.dur_stats)
    assert all((v.abs() < 1e-4).all() for v in same.values()), same

    x = x0.clone()
    x[:, :, 0, :7] += 2 * 0.1 / (Q_HI - Q_LO)               # every arm joint +0.1 rad
    x[:, :, 1:3, 0] += 0.010 / POS_SCALE                    # both cubes +10 mm along x
    R = G.rot6d_to_matrix(x[:, :, 1:3, 3:9])
    c, s = math.cos(math.radians(5)), math.sin(math.radians(5))
    Rz = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    x[:, :, 1:3, 3:9] = G.matrix_to_rot6d(Rz @ R)            # both cubes turned 5 degrees
    longer = torch.log(torch.exp(dur * ds.dur_stats["sigma"] + ds.dur_stats["mu"]) + 0.5)
    err = motion_errors(x, (longer - ds.dur_stats["mu"]) / ds.dur_stats["sigma"], x0, dur, ds.n_objects,
                        ds.dur_stats)
    torch.testing.assert_close(err["joint_rad"], torch.full((2,), 0.1), atol=1e-4, rtol=0)
    torch.testing.assert_close(err["obj_pos_mm"], torch.full((2,), 10.0), atol=1e-3, rtol=0)
    torch.testing.assert_close(err["obj_rot_deg"], torch.full((2,), 5.0), atol=1e-2, rtol=0)
    torch.testing.assert_close(err["dur_s"], torch.full((2,), 0.5), atol=1e-4, rtol=0)
    assert (err["finger_mm"] < 1e-4).all()
