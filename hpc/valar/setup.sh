#!/usr/bin/env bash
# One-time setup on the VALAR login node: a Python environment, Ollama and the models.
#
#   bash hpc/valar/setup.sh                        # everything (the models are about 90 GB: it takes a while)
#   bash hpc/valar/setup.sh models qwen2.5:72b     # download more models later
#   bash hpc/valar/setup.sh python                 # only the Python environment
#   bash hpc/valar/setup.sh ollama                 # only Ollama
#
# Run it again at any time: what is already there is kept. Only downloads happen here; no model runs on
# the login node. Settings (where things go, which models) are in common.sh and your my_settings.sh.
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

VENV=$STORE/venv

setup_python() {
    mkdir -p "$STORE"
    if [ -x "$PY" ] && python_ok "$PY"; then
        echo "Python environment: $VENV (already there)"
    else
        local base="" candidate
        for candidate in "${PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3; do
            if [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
                base=$(command -v "$candidate")
                break
            fi
        done
        if [ -z "$base" ] && load_module "$PYTHON_MODULE"; then
            echo "$PYTHON_MODULE" > "$STORE/python_module"   # the jobs load it too
            for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
                if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
                    base=$(command -v "$candidate")
                    break
                fi
            done
        fi
        if [ -z "$base" ]; then
            echo "No Python 3.10 or newer found. Load one (see: module avail) and run this again, or put" >&2
            echo "PYTHON=/path/to/python3 in hpc/valar/my_settings.sh." >&2
            exit 1
        fi
        echo "Making a Python environment in $VENV with $base ($("$base" --version 2>&1))"
        "$base" -m venv "$VENV"
    fi
    "$PY" -m pip install --quiet --upgrade pip
    # chromadb serves only the old pipeline (run.py); the organism reads its library without it.
    grep -v -i '^chromadb' "$REPO/requirements.txt" > "$STORE/requirements-hpc.txt"
    "$PY" -m pip install --quiet -r "$STORE/requirements-hpc.txt"
    "$PY" -c 'import pydantic, pypdf, requests, rich, yaml' && echo "Python environment ready: $("$PY" --version 2>&1)"
}

setup_ollama() {
    if [ -x "$OLLAMA" ]; then
        echo "Ollama: already in $STORE/ollama"
        return
    fi
    mkdir -p "$STORE/ollama"
    # The official Linux build (with its GPU libraries). OLLAMA_VERSION=0.x.y in my_settings.sh fixes a version.
    local url="https://ollama.com/download/ollama-linux-amd64.tar.zst${OLLAMA_VERSION:+?version=$OLLAMA_VERSION}"
    echo "Downloading Ollama (about 1.5 GB) into $STORE/ollama ..."
    if command -v zstd >/dev/null 2>&1; then
        curl -fL --progress-bar "$url" | zstd -d | tar -xf - -C "$STORE/ollama"
    else
        if [ ! -x "$PY" ]; then setup_python; fi
        "$PY" -m pip install --quiet zstandard
        curl -fL --progress-bar "$url" \
            | "$PY" -c 'import sys, zstandard; zstandard.ZstdDecompressor().copy_stream(sys.stdin.buffer, sys.stdout.buffer)' \
            | tar -xf - -C "$STORE/ollama"
    fi
    if [ ! -x "$OLLAMA" ]; then
        echo "The download did not contain bin/ollama; look in $STORE/ollama." >&2
        exit 1
    fi
    echo "Ollama installed: $("$OLLAMA" --version 2>&1 | tail -1 | sed 's/^Warning: //')"
}

setup_models() {
    if [ ! -x "$PY" ]; then setup_python; fi
    if [ ! -x "$OLLAMA" ]; then setup_ollama; fi
    mkdir -p "$STORE/models" "$STORE/logs"
    local port pid model seen=" " unique=()
    for model in "$@"; do
        case "$seen" in
            *" $model "*) ;;
            *) unique+=("$model"); seen="$seen$model " ;;
        esac
    done
    set -- "${unique[@]}"
    port=$(free_port)
    echo "Downloading into $STORE/models: $*"
    # A server only for downloading, on a free port; CUDA_VISIBLE_DEVICES=-1 keeps it off the login node's GPU.
    CUDA_VISIBLE_DEVICES=-1 OLLAMA_HOST="127.0.0.1:$port" OLLAMA_MODELS="$STORE/models" \
        "$OLLAMA" serve >"$STORE/logs/setup-ollama.log" 2>&1 &
    pid=$!
    # shellcheck disable=SC2064
    trap "kill $pid 2>/dev/null || true" EXIT
    "$PY" "$VALAR_DIR/prepare_server.py" --url "http://127.0.0.1:$port"
    for model in "$@"; do
        OLLAMA_HOST="127.0.0.1:$port" "$OLLAMA" pull "$model"
    done
    OLLAMA_HOST="127.0.0.1:$port" "$OLLAMA" list
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    trap - EXIT
    echo "Models take $(du -sh "$STORE/models/" 2>/dev/null | cut -f1) in $STORE/models."
}

step=${1:-all}
if [ $# -gt 0 ]; then shift; fi
case "$step" in
    all)
        setup_python
        setup_ollama
        # shellcheck disable=SC2086
        setup_models $LADDER $JUDGE_MODEL
        echo
        echo "Ready. A 10-minute pilot:  bash hpc/valar/submit_ladder.sh --models qwen2.5:7b --minutes 10 --name pilot"
        ;;
    python) setup_python ;;
    ollama) setup_ollama ;;
    models)
        # shellcheck disable=SC2086
        if [ $# -eq 0 ]; then set -- $LADDER $JUDGE_MODEL; fi
        setup_models "$@"
        ;;
    *)
        echo "usage: bash hpc/valar/setup.sh [all | python | ollama | models [MODEL ...]]" >&2
        exit 1
        ;;
esac
