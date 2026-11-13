"""A2: planner shards -> Episode npz files (<out>/<split>/<seed>_<ref>.npz) packed as <out>/<split>.tar.
usage: python scripts/extract_episodes.py --split train --out data/episodes/StackCube-v1 data/raw/StackCube-v1/shard_*.h5
"""
import argparse
import subprocess
import tarfile
from pathlib import Path

from gaussbridge.sim.extract import load_shard

p = argparse.ArgumentParser()
p.add_argument("shards", nargs="+", help="shard .h5 files (the .json and .knobs.json must sit next to them)")
p.add_argument("--split", required=True)
p.add_argument("--out", required=True, type=Path)
a = p.parse_args()

commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
d = a.out / a.split
d.mkdir(parents=True, exist_ok=True)
n = 0
for shard in a.shards:
    for ep in load_shard(shard.removesuffix(".h5")):
        ep.meta["git_commit"] = commit
        ep.save(d / f"{ep.meta['seed']:05d}_{ep.meta['ref_idx']:02d}.npz")
        n += 1
with tarfile.open(a.out / f"{a.split}.tar", "w") as tar:  # one big file for CHTC staging
    tar.add(d, arcname=a.split)
print(f"{n} episodes -> {a.out / a.split}.tar")
