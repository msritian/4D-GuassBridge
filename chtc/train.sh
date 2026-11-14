#!/bin/bash
# C4 training job with HTCondor self-checkpointing (see chtc/train.sub).
# usage: train.sh <config yaml inside the code> <run name> [extra args for scripts/train_motion.py]
# Inputs: gb_code.tar.gz (code + urdf/, from chtc/pack.sh), episodes_train.tar and episodes_val.tar (Episode .npz
# under train/ and val/). The run directory <run name>/ is the checkpoint: HTCondor saves it whenever this script
# exits with 85 and brings it back when the job restarts (possibly on another machine), so training resumes.
set -uo pipefail
config=$1
run=$2
shift 2
echo "host $(hostname)  start $(date)  run $run  config $config  args $*"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

if [ ! -d code ]; then mkdir -p code && tar -xzf gb_code.tar.gz -C code; fi
[ -d train ] || tar -xf episodes_train.tar
[ -d val ] || tar -xf episodes_val.tar
export PYTHONPATH="$PWD/code/src"
export GB_URDF_DIR="$PWD/code/urdf"              # FK for validation metrics without ManiSkill (kinematics.urdf_path)
# pytorch_kinematics (FK) is not in the PyTorch image; ManiSkill pins 0.7.6 (CLAUDE.md §3.1)
python -c "import pytorch_kinematics" 2>/dev/null || pip install --quiet --user pytorch_kinematics==0.7.6
python -c "import torch, pytorch_kinematics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

status=0
python code/scripts/train_motion.py "code/$config" --run "$run" --set data.train=train data.val=val "$@" || status=$?
echo "end $(date)  exit $status"
if [ $status -eq 0 ]; then tar -czf "$run.tar.gz" "$run"; fi   # final results; while paused the run dir is the checkpoint
exit $status
