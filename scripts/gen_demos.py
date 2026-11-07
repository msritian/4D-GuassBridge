"""Generate randomised planner demos (A1). Example:
python scripts/gen_demos.py --env-id StackCube-v1 --seed-start 0 --num-seeds 50 --refs-per-seed 8 \
    --out data/raw/StackCube-v1/shard_000
"""
import argparse
import json

from gaussbridge.sim.planner_gen import generate

p = argparse.ArgumentParser()
p.add_argument("--env-id", required=True, choices=["StackCube-v1", "PickCube-v1"])
p.add_argument("--seed-start", type=int, required=True)
p.add_argument("--num-seeds", type=int, required=True)
p.add_argument("--refs-per-seed", type=int, default=8)
p.add_argument("--max-attempts", type=int, default=3)
p.add_argument("--out", required=True, help="path prefix; writes <out>.h5, <out>.json, <out>.knobs.json")
a = p.parse_args()
print(json.dumps(generate(a.env_id, a.seed_start, a.num_seeds, a.refs_per_seed, a.out, a.max_attempts)))
