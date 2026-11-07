"""C1 trajectory checks. The full planner-demo round trip (with physics replay) is in test_sim_integration.py."""
import math

import torch

from gaussbridge import geometry as G
from gaussbridge.trajectory import resample


def test_resample_endpoints_rotation_sign_flip_and_gripper_hold():
    # object yawing 0 -> 170 deg at constant rate; the simulator may report q or -q for the same rotation
    yaw = torch.linspace(0, math.radians(170), 37, dtype=torch.float64)
    q = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], -1)
    q[18:] *= -1
    out = resample(q, 128, "slerp")
    assert torch.equal(out[0], q[0]) and torch.equal(out[-1], q[-1])
    R = G.quat_to_matrix(out)
    torch.testing.assert_close(torch.atan2(R[:, 1, 0], R[:, 0, 0]),
                               torch.linspace(0, math.radians(170), 128, dtype=torch.float64))

    grip = torch.tensor([1.0] * 20 + [-1.0] * 17)  # close command at step 20 of 36
    g = resample(grip, 128, "hold")
    switch = int((g < 0).nonzero()[0])
    assert set(g.tolist()) == {1.0, -1.0} and (g[switch:] < 0).all() and abs(switch / 127 - 19.5 / 36) < 0.01
