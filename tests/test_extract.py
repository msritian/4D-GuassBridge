"""A2: planner shards -> Episodes with the attached rule. Needs ManiSkill + mplib (Linux): run in docker/Dockerfile.cpu."""
import numpy as np
import pytest

from gaussbridge.sim.extract import load_shard
from gaussbridge.sim.planner_gen import generate

# task: (carried object, objects that are never held, still held at the last step?)
CASES = {"StackCube-v1": ("cubeA", ["cubeB"], False), "PickCube-v1": ("cube", [], True)}


@pytest.mark.parametrize("env_id", CASES)
def test_attached_is_one_grasp_interval(tmp_path, env_id):
    generate(env_id, seed_start=0, num_seeds=1, refs_per_seed=2, out=tmp_path / "shard")
    episodes = load_shard(tmp_path / "shard")
    carried, never, held_at_end = CASES[env_id]
    assert [(ep.meta["seed"], ep.meta["ref_idx"]) for ep in episodes] == [(0, 0), (0, 1)]  # knobs sidecar matched
    for ep in episodes:
        a = ep.attached[:, ep.obj_names.index(carried)]
        assert np.sum(np.diff(a.astype(int)) == 1) + a[0] == 1  # exactly one contiguous interval
        finger = ep.robot_qpos[a, 7]
        assert finger.min() > 0.017 and finger.max() < 0.021  # closed on the 4 cm cube (≈ 18.3 mm), never open
        assert a[-1] == held_at_end  # StackCube releases cubeA on cubeB; PickCube keeps holding at the goal
        for name in never:
            assert not ep.attached[:, ep.obj_names.index(name)].any()
