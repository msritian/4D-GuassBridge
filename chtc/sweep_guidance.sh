#!/bin/bash
# C6 guidance-scale sweep job (see chtc/sweep_guidance.sub).
# usage: sweep_guidance.sh <episodes tar> <overfit run tar> <run name> [extra args for scripts/sweep_guidance.py]
# Inputs: gb_code.tar.gz (code + urdf/, as chtc/pack.sh builds it), an episodes tar (dev/*.npz) and an overfit run
# tar (<run>/checkpoint.pt, from chtc/overfit.sub). Output: <run name>.tar.gz with <run name>.{json,md} and the log.
set -uo pipefail
episodes_tar=$1
model_tar=$2
run=$3
shift 3
echo "host $(hostname)  start $(date)  run $run  episodes $episodes_tar  model $model_tar  args $*"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
mkdir -p code && tar -xzf gb_code.tar.gz -C code
tar -xf "$episodes_tar"                                 # -> dev/
mkdir -p model && tar -xzf "$model_tar" -C model        # -> model/<overfit run>/checkpoint.pt
ckpt=$(find model -name checkpoint.pt | head -1)
export PYTHONPATH="$PWD/code/src"
export GB_URDF_DIR="$PWD/code/urdf"
python -c "import pytorch_kinematics" 2>/dev/null || pip install --quiet --user pytorch_kinematics==0.7.6
status=0
mkdir -p "$run"
python code/scripts/sweep_guidance.py --ckpt "$ckpt" --episodes dev --out "$run/$run" "$@" 2>&1 | tee "$run/log.txt"
status=${PIPESTATUS[0]}
tar -czf "$run.tar.gz" "$run"
echo "end $(date)  exit $status"
exit $status
