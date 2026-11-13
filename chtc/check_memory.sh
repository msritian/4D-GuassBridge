#!/bin/bash
# C3 memory check job (see chtc/check_memory.sub). Input: gb_code.tar.gz. Output: memory.json.
set -uo pipefail
echo "host $(hostname)  start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
mkdir -p code && tar -xzf gb_code.tar.gz -C code
export PYTHONPATH="$PWD/code/src"
status=0
python code/scripts/check_memory.py --out memory.json "$@" || status=$?
[ -f memory.json ] || echo '{"error": "check_memory.py failed, see the .err log"}' > memory.json
echo "end $(date)  exit $status"
exit $status
