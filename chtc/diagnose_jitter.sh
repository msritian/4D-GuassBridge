#!/bin/bash
# Jitter diagnosis job (see chtc/diagnose_jitter.sub and scripts/diagnose_jitter.py).
# usage: diagnose_jitter.sh <out name> <overfit run tar>...   (C3 overfit runs from chtc/overfit.sub)
# Inputs: gb_code.tar.gz (code + urdf/), episodes_stack_dev.tar (dev/), the run tars. Output: <out name>.tar.gz.
# (Job 8591071 also compared the C4 trial run's raw vs EMA weights; see progress.md.)
set -uo pipefail
out=$1
shift
echo "host $(hostname)  start $(date)  runs $*"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
mkdir -p code && tar -xzf gb_code.tar.gz -C code
tar -xf episodes_stack_dev.tar
export PYTHONPATH="$PWD/code/src" GB_URDF_DIR="$PWD/code/urdf"
python -c "import pytorch_kinematics" 2>/dev/null || pip install --quiet --user pytorch_kinematics==0.7.6
models=()
for t in "$@"; do
  name=${t%.tar.gz}
  mkdir -p "m_$name" && tar -xzf "$t" -C "m_$name"
  models+=(--model "$name=$(find "m_$name" -name checkpoint.pt | head -1)")
done
mkdir -p "$out"
python code/scripts/diagnose_jitter.py --episodes dev --out "$out/$out" --steps 40,100 "${models[@]}" 2>&1 | tee "$out/log.txt"
status=${PIPESTATUS[0]}
tar -czf "$out.tar.gz" "$out"
echo "end $(date)  exit $status"
exit $status
