#!/bin/bash
# One planner shard per CHTC CPU job (CLAUDE.md §6.1): generate demos (A1), then extract Episodes (A2) into ./out.
# usage (via run.sh): gen_shard.sh <task> <split> <seed_start> <num_seeds> <refs_per_seed>
set -euo pipefail
task=$1 split=$2 start=$3 n=$4 refs=$5
name=${split}_$(printf %05d "$start")
python code/scripts/gen_demos.py --env-id "$task" --seed-start "$start" --num-seeds "$n" --refs-per-seed "$refs" \
    --out "out/raw/$task/$name"
python code/scripts/extract_episodes.py --split "$split" --out "out/episodes/$task" "out/raw/$task/$name.h5"
rm "out/episodes/$task/$split.tar"  # the per-split tar is built once all shards are back (A4)
