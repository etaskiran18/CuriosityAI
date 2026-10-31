#!/usr/bin/env bash
# On your computer: copy this project's code to VALAR's scratch, then check that both sides have the same code.
# Your lives (memory/), results (runs/), Python environments and secrets (.env) are not copied.
#
#   bash hpc/valar/sync_to_valar.sh
#
# `ssh valar` must work without a password (the lab's VALAR guide sets it up). VALAR_HOST names another
# host, VALAR_REPO another folder there (default /scratch/<your VALAR username>/CuriosityAI).
# Paths and arguments are quoted here with printf %q and sent ready to run: expanding them here is intended.
# shellcheck disable=SC2029,SC2087
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
host=${VALAR_HOST:-valar}
# shellcheck disable=SC2016
remote_user=$(ssh "$host" 'echo $USER')
dest=${VALAR_REPO:-/scratch/$remote_user/CuriosityAI}

ssh "$host" "mkdir -p $(printf %q "$dest")"
rsync -a --delete \
    --exclude '/memory/' --exclude '/runs/' --exclude '.venv/' --exclude 'venv/' --exclude '.env' \
    --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '*.pyc' \
    "$repo/" "$host:$dest/"

# The same code on both sides? Every file git knows, except the lives and results that stay here.
checksums() {
    (cd "$repo" && git ls-files -z -- . ':(exclude)memory' ':(exclude)runs') | "$@" 2>/dev/null || true
}
here=$(checksums bash -c "cd $(printf %q "$repo") && xargs -0 md5sum")
there=$(checksums ssh "$host" "cd $(printf %q "$dest") && xargs -0 md5sum")
if [ "$here" != "$there" ]; then
    echo "The code on VALAR differs from the code here:" >&2
    diff <(echo "$here") <(echo "$there") | head -20 >&2
    exit 1
fi
echo "Code copied to $host:$dest ($(echo "$here" | wc -l) files, the same on both sides)."
