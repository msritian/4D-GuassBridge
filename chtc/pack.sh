#!/bin/bash
# Build the upload bundle for chtc/train.sub: code (git HEAD) + the Panda URDFs + train/val episode tars.
# usage: bash chtc/pack.sh <train episodes dir> <val episodes dir> <out dir>
# Run from the repo root on a machine with ManiSkill installed (only to copy its two URDF files). Uses committed code
# only (git archive HEAD), so commit first.
set -euo pipefail
train=$1
val=$2
out=$3
mkdir -p "$out"
tmp=$(mktemp -d)
git archive HEAD src scripts configs pyproject.toml chtc | tar -x -C "$tmp"
assets=$(python -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('mani_skill').submodule_search_locations[0]) / 'assets')")
mkdir -p "$tmp/urdf/robots/panda"
cp "$assets/robots/panda/panda_v2.urdf" "$assets/robots/panda/panda_v3.urdf" "$tmp/urdf/robots/panda/"
tar -czf "$out/gb_code.tar.gz" -C "$tmp" .
tar -cf "$out/episodes_train.tar" -C "$(dirname "$train")" -s "|^$(basename "$train")|train|" "$(basename "$train")" 2>/dev/null \
  || tar -cf "$out/episodes_train.tar" -C "$(dirname "$train")" --transform "s|^$(basename "$train")|train|" "$(basename "$train")"
tar -cf "$out/episodes_val.tar" -C "$(dirname "$val")" -s "|^$(basename "$val")|val|" "$(basename "$val")" 2>/dev/null \
  || tar -cf "$out/episodes_val.tar" -C "$(dirname "$val")" --transform "s|^$(basename "$val")|val|" "$(basename "$val")"
cp chtc/train.sub chtc/train.sh "$out/"
rm -rf "$tmp"
echo "bundle in $out:"; ls -la "$out"
echo "git HEAD: $(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo ' (uncommitted changes NOT included)')"
