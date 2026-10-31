#!/usr/bin/env bash
# On your computer: bring home the archive of a batch (collect.sh makes it on VALAR) and unpack it.
#
#   bash hpc/valar/fetch_from_valar.sh pilot       # a batch by name; without one, the newest batch
# Paths and arguments are quoted here with printf %q and sent ready to run: expanding them here is intended.
# shellcheck disable=SC2029,SC2087
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
host=${VALAR_HOST:-valar}
batch=${1:-latest}
# shellcheck disable=SC2016
remote_user=$(ssh "$host" 'echo $USER')
dest=${VALAR_REPO:-/scratch/$remote_user/CuriosityAI}

there=$(ssh "$host" "readlink -f $(printf %q "$dest/runs/valar/$batch")")
name=$(basename "$there")
archive=$there/$name-share.tar.gz
if ! ssh "$host" "test -f $(printf %q "$archive")"; then
    echo "No archive at $host:$archive yet: run  bash hpc/valar/remote.sh collect.sh $name  first." >&2
    exit 1
fi
here=$repo/runs/valar/$name
mkdir -p "$here"
ssh "$host" "cat $(printf %q "$archive")" >"$here/$name-share.tar.gz"
tar -xzf "$here/$name-share.tar.gz" -C "$here"
echo "Here: $here/$name-share (start with README.md there)"
