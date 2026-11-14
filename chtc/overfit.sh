#!/bin/bash
# C3 overfit job (runs inside the container on the CHTC execute node; see chtc/overfit.sub).
# usage: overfit.sh <episodes tar> <run name> [extra args for scripts/overfit_motion.py]
# Inputs in the job's working directory: gb_code.tar.gz (git archive of the repo) and the episodes tar (Episode .npz
# files under dev/, from scripts/extract_episodes.py). Output: <run name>.tar.gz.
# Training needs only torch + numpy, so the public PyTorch image is enough (no ManiSkill / Vulkan / gsplat).
set -uo pipefail
episodes_tar=$1
run=$2
shift 2
echo "host $(hostname)  start $(date)  run $run  episodes $episodes_tar  args $*"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

mkdir -p code && tar -xzf gb_code.tar.gz -C code
tar -xf "$episodes_tar"                       # -> dev/*.npz
export PYTHONPATH="$PWD/code/src"
python -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available())"

# Always pack and return whatever exists (logs of a crashed run are the most useful thing to get back);
# the job still reports the script's exit code.
status=0
python code/scripts/overfit_motion.py --episodes dev --out "$run" "$@" || status=$?
mkdir -p "$run" && tar -czf "$run.tar.gz" "$run"
echo "end $(date)  exit $status"
exit $status
