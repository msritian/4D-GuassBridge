"""A1 checks. Needs ManiSkill + mplib (Linux): run inside docker/Dockerfile.cpu."""
import json

import h5py
import numpy as np
import pytest

from gaussbridge.sim.planner_gen import generate, sample_knobs


def test_knobs_reproducible_and_in_range():
    ks = [sample_knobs(np.random.default_rng([0, r, 0])) for r in range(300)]
    assert sample_knobs(np.random.default_rng([0, 0, 0])) == ks[0]
    assert {k.grasp_yaw_k for k in ks} == {0, 1, 2, 3}
    assert all(0.03 <= k.reach_h <= 0.10 and 0.06 <= k.lift_h <= 0.20 and 0 <= k.place_h <= 0.08
               and 0.4 <= k.speed <= 0.9 for k in ks)
    assert {k.via is None for k in ks} == {True, False} and {k.place_h == 0 for k in ks} == {True, False}
    assert any(k.rrt_transport for k in ks)


def _start_state(traj):
    parts = []
    traj["env_states"].visititems(lambda _, d: parts.append(d[0]) if isinstance(d, h5py.Dataset) else None)
    return np.concatenate(parts)


@pytest.mark.parametrize("env_id", ["StackCube-v1", "PickCube-v1"])
def test_two_refs_share_start_and_differ(tmp_path, env_id):
    out = tmp_path / "shard"
    stats = generate(env_id, seed_start=0, num_seeds=1, refs_per_seed=2, out=out)
    episodes = json.loads(open(f"{out}.json").read())["episodes"]
    knobs = json.loads(open(f"{out}.knobs.json").read())
    assert stats["saved"] == len(episodes) == len(knobs) == 2
    assert all(e["success"] and e["episode_seed"] == 0 for e in episodes)
    assert [knobs[str(e["episode_id"])]["num_steps"] for e in episodes] == [e["elapsed_steps"] for e in episodes]
    with h5py.File(f"{out}.h5") as f:
        t0, t1 = f["traj_0"], f["traj_1"]
        np.testing.assert_allclose(_start_state(t0), _start_state(t1), atol=1e-6)  # same start scene
        n = min(len(t0["actions"]), len(t1["actions"]))
        assert not np.allclose(t0["actions"][:n], t1["actions"][:n])  # different motions
