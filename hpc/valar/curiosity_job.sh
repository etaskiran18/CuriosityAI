#!/usr/bin/env bash
#SBATCH --job-name=curiosity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#
# One life of the curiosity organism on a GPU node, with an Ollama server of its own.
# submit_ladder.sh submits it with the partition, the GPUs and the settings filled in (hpc/valar/README.md).
#
# Settings come from the environment:
#   MODEL        the organism's model, e.g. qwen2.5:7b (required)
#   JUDGE        the judge's model; keep it the same for every run you compare (default: MODEL)
#   MINUTES      how long it lives (default 60)
#   HEARTBEATS   also stop after this many heartbeats, whichever comes first (default: no limit)
#   SEED         random seed (default 1)
#   ARCH         records (v10, default) or drive (v9)
#   ABLATE       parts of the records architecture to switch off, e.g. decision,attention
#   EXAM         1 = the exam on the philosophy of curiosity before and after (not counted in MINUTES)
#   MODE         life (default) or traps (the four trap tests, TRAP_HEARTBEATS heartbeats each, default 8)
#   WORK_MINUTES, REST_MINUTES   its rhythm (default: a rest of 1 minute after every 15 minutes of thinking;
#                after a rest it first goes back to a text that made it curious)
#   LABEL        a name for the run (default: from the model)
#   RUN_DIR      where everything is written (default: runs/valar/single/LABEL)
#   ALLOW_CPU    1 = go on even if a model does not fit on the GPU (many times slower)
set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$REPO/hpc/valar/common.sh" ]; then
    echo "The CuriosityAI folder is not at REPO=$REPO: submit from that folder or set REPO." >&2
    exit 2
fi
# shellcheck source=common.sh
source "$REPO/hpc/valar/common.sh"
cd "$REPO"
if [ ! -x "$PY" ] || [ ! -x "$OLLAMA" ]; then
    echo "No Python environment or Ollama in $STORE: run  bash hpc/valar/setup.sh  on the login node first." >&2
    exit 2
fi
if [ -f "$STORE/python_module" ]; then
    load_module "$(cat "$STORE/python_module")" || true
fi

MODEL=${MODEL:?set MODEL, for example MODEL=qwen2.5:7b}
JUDGE=${JUDGE:-$MODEL}
MINUTES=${MINUTES:-60}
HEARTBEATS=${HEARTBEATS:-}
SEED=${SEED:-1}
ARCH=${ARCH:-records}
ABLATE=${ABLATE:-}
EXAM=${EXAM:-0}
MODE=${MODE:-life}
WORK_MINUTES=${WORK_MINUTES:-15}
REST_MINUTES=${REST_MINUTES:-1}
TRAP_HEARTBEATS=${TRAP_HEARTBEATS:-8}
LABEL=${LABEL:-$(slug "$MODEL")}
RUN_DIR=${RUN_DIR:-$RESULTS/single/$LABEL}
mkdir -p "$RUN_DIR"
INFO=$RUN_DIR/run_info.txt

# Its own server on a free port, so that jobs sharing a node do not meet; models stay loaded for the whole job.
PORT=$(free_port)
URL=http://127.0.0.1:$PORT
OLLAMA_HOST=127.0.0.1:$PORT OLLAMA_MODELS=$STORE/models OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_FLASH_ATTENTION=1 OLLAMA_LOAD_TIMEOUT=30m \
    "$OLLAMA" serve >>"$RUN_DIR/ollama.log" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true' EXIT
stopped=""
trap 'stopped="stopped by a signal from SLURM (time limit or scancel)"' INT TERM
export PYTHONUNBUFFERED=1   # the job's log shows each line as it comes

gpus=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';' -) || true
commit=$(git rev-parse --short HEAD 2>/dev/null) || commit=unknown
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || branch=unknown
{
    echo "run: $LABEL ($MODE)"
    echo "model: $MODEL"
    echo "judge: $JUDGE"
    echo "architecture: $ARCH${ABLATE:+, without $ABLATE}"
    if [ "$MODE" = traps ]; then
        echo "traps: $TRAP_HEARTBEATS heartbeats each"
    else
        echo "time: $MINUTES minutes${HEARTBEATS:+ or $HEARTBEATS heartbeats, whichever comes first}"
        echo "rhythm: a rest of $REST_MINUTES min after every $WORK_MINUTES min of thinking"
    fi
    echo "seed: $SEED"
    echo "library: frozen (no --web)"
    echo "node: $(hostname), job ${SLURM_JOB_ID:-none} on ${SLURM_JOB_PARTITION:-no partition}"
    echo "gpus: ${gpus:-none}"
    echo "code: $commit ($branch)"
    echo "started: $(date '+%Y-%m-%d %H:%M:%S')"
} >"$INFO"
cat "$INFO"

# The usual configuration with a rhythm for a cluster: no breaths, no temperature or battery guard, short rests.
"$PY" - "$RUN_DIR/config.yaml" "$WORK_MINUTES" "$REST_MINUTES" <<'PYCONF'
import sys

import yaml

out, work, rest = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
body = config.setdefault("organism", {}).setdefault("body", {})
body.update(enabled=True, breath_seconds=0, work_minutes=work, rest_minutes=rest,
            gpu_temperature_guard=False, pause_on_battery=False)
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
PYCONF

# Wait for the server, load both models (not counted in the run's time) and make sure they sit on the GPU.
NUM_CTX=$("$PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["llm"]["num_ctx"])' "$RUN_DIR/config.yaml")
PREPARE=(--url "$URL" --models "$MODEL" "$JUDGE" --log "$RUN_DIR/ollama.log" --num-ctx "$NUM_CTX")
if [ "${ALLOW_CPU:-0}" = 1 ]; then PREPARE+=(--allow-cpu); fi
set +e
"$PY" "$VALAR_DIR/prepare_server.py" "${PREPARE[@]}" | tee -a "$INFO"
status=${PIPESTATUS[0]}
set -e
if [ "$status" != 0 ]; then
    echo "not started: the server or the models were not ready (exit code $status); see $RUN_DIR/ollama.log" | tee -a "$INFO"
    exit "$status"
fi

COMMON=(--config "$RUN_DIR/config.yaml" --ollama-url "$URL" --model "$MODEL" --judge-model "$JUDGE" --seed "$SEED")
set +e
if [ "$MODE" = traps ]; then
    TRAPS=("${COMMON[@]}" --home "$RUN_DIR/traps" --heartbeats "$TRAP_HEARTBEATS")
    if [ -n "$ABLATE" ]; then TRAPS+=(--ablate "$ABLATE"); fi
    "$PY" scripts/trap_tests.py "${TRAPS[@]}"
    status=$?
else
    "$PY" live.py "${COMMON[@]}" --home "$RUN_DIR/life" --check
    LIFE=("${COMMON[@]}" --home "$RUN_DIR/life" --architecture "$ARCH" --minutes "$MINUTES" --label "$LABEL")
    if [ -n "$HEARTBEATS" ]; then LIFE+=(--heartbeats "$HEARTBEATS"); fi
    if [ -n "$ABLATE" ]; then LIFE+=(--ablate "$ABLATE"); fi
    if [ "$EXAM" = 1 ]; then LIFE+=(--exam); fi
    "$PY" live.py "${LIFE[@]}"
    status=$?
fi
set -e

{
    echo "finished: $(date '+%Y-%m-%d %H:%M:%S'), exit code $status${stopped:+, $stopped}"
    if [ -d "$RUN_DIR/life/sessions" ]; then
        "$PY" - "$RUN_DIR/life/sessions" <<'PYLIVED' || true
import json
import sys
from pathlib import Path

sessions = sorted(p for p in Path(sys.argv[1]).iterdir() if (p / "metrics.json").is_file())
if sessions:
    m = json.loads((sessions[-1] / "metrics.json").read_text(encoding="utf-8"))
    print(f"lived: {m.get('heartbeats')} heartbeats in {m.get('minutes')} minutes "
          f"({m.get('heartbeats_per_hour')} per hour), session {sessions[-1].name}")
PYLIVED
    fi
} | tee -a "$INFO"
exit "$status"
