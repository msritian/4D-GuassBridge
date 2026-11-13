#!/bin/bash
# CHTC job wrapper (CLAUDE.md §3.5): unpack the code, run a command, pack ./out into one tar.
# The code tarball is made on the submit node with `git archive -o chtc/code.tar HEAD`, so every job runs the same
# commit, recorded as $GIT_COMMIT.  usage (submit file): executable = chtc/run.sh, arguments = <out.tar> <command...>
set -euo pipefail
out_tar=$1
shift
mkdir -p code out
tar -xf code.tar -C code
GIT_COMMIT=$(git get-tar-commit-id < code.tar || true)
export GIT_COMMIT PYTHONPATH=$PWD/code/src HOME=$PWD  # HOME: caches must land in the job's scratch dir
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/lvp_icd.json} MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
"$@"
tar -cf "$out_tar" out
