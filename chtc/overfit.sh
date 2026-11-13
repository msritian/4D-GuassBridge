#!/bin/bash
# C3 overfit job (runs inside the container on the CHTC execute node; see chtc/overfit.sub).
# Inputs transferred into the job's working directory: gb_code.tar.gz (git archive of the repo) and
# episodes_dev.tar (Episode .npz files, from scripts/extract_episodes.py). Output: overfit_out.tar.gz.
# Training needs only torch + numpy, so the public PyTorch image is enough (no ManiSkill / Vulkan / gsplat).
set -euo pipefail
echo "host $(hostname)  start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

mkdir -p code && tar -xzf gb_code.tar.gz -C code
tar -xf episodes_dev.tar                       # -> dev/*.npz
export PYTHONPATH="$PWD/code/src"
python -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available())"

python code/scripts/overfit_motion.py --episodes dev --out overfit_out "$@"

tar -czf overfit_out.tar.gz overfit_out
echo "end $(date)"
