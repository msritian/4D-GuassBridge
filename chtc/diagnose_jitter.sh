#!/bin/bash
# Jitter diagnosis job (see chtc/diagnose_jitter.sub and scripts/diagnose_jitter.py).
# Inputs: gb_code.tar.gz (code + urdf/), episodes_stack_dev.tar (dev/), stack_allgoals.tar.gz (C3 overfit run, no
# EMA), trial_stackcube_dev.tar.gz (C4 trial run: checkpoints with raw + EMA weights). Output: diag_stack.tar.gz.
set -uo pipefail
echo "host $(hostname)  start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
mkdir -p code && tar -xzf gb_code.tar.gz -C code
tar -xf episodes_stack_dev.tar
tar -xzf stack_allgoals.tar.gz
tar -xzf trial_stackcube_dev.tar.gz
export PYTHONPATH="$PWD/code/src" GB_URDF_DIR="$PWD/code/urdf"
python -c "import pytorch_kinematics" 2>/dev/null || pip install --quiet --user pytorch_kinematics==0.7.6
overfit=$(find . -path ./code -prune -o -name checkpoint.pt -print | head -1)
c4=trial_stackcube_dev/ckpt/ckpt_0003000.pt
mkdir -p diag_stack
python code/scripts/diagnose_jitter.py --episodes dev --out diag_stack/diag_stack \
  --model "overfit10k=$overfit" --model "c4_3k_ema=$c4:ema" --model "c4_3k_raw=$c4:model" 2>&1 | tee diag_stack/log.txt
status=${PIPESTATUS[0]}
tar -czf diag_stack.tar.gz diag_stack
echo "end $(date)  exit $status"
exit $status
