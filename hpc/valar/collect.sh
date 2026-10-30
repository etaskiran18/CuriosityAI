#!/usr/bin/env bash
# Gather the diaries, research maps, reports and a comparison table of a batch of runs, ready to send on.
#
#   bash hpc/valar/collect.sh              # the newest batch
#   bash hpc/valar/collect.sh pilot        # the batch runs/valar/pilot
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

batch=$RESULTS/${1:-latest}
if [ ! -d "$batch" ]; then
    echo "There is no batch at $batch." >&2
    exit 1
fi
batch=$(cd "$batch" && pwd -P)   # "latest" is a link to the newest batch
python=$PY
if [ ! -x "$python" ]; then python=python3; fi
"$python" "$VALAR_DIR/collect.py" "$batch"
echo "Copy it to your computer with, for example:  scp <user>@<VALAR login address>:$batch/$(basename "$batch")-share.tar.gz ."
