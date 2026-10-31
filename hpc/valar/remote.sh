#!/usr/bin/env bash
# On your computer: run one of the scripts of hpc/valar on VALAR's login node, in the copy sync_to_valar.sh made.
#
#   bash hpc/valar/remote.sh submit_ladder.sh --models qwen2.5:7b --minutes 10 --name pilot
#   bash hpc/valar/remote.sh --detach setup.sh     # long work: it goes on after you disconnect; prints its log
#
# The command travels as a script over `ssh valar bash -s`, so arguments arrive exactly as typed.
# Paths and arguments are quoted here with printf %q and sent ready to run: expanding them here is intended.
# shellcheck disable=SC2029,SC2087
set -euo pipefail
host=${VALAR_HOST:-valar}
detach=0
if [ "${1:-}" = --detach ]; then
    detach=1
    shift
fi
script=${1:?"which script of hpc/valar, for example setup.sh"}
shift
case "$script" in
    */* | .*) echo "Name a script of hpc/valar, for example setup.sh" >&2; exit 1 ;;
esac
# shellcheck disable=SC2016
remote_user=$(ssh "$host" 'echo $USER')
dest=${VALAR_REPO:-/scratch/$remote_user/CuriosityAI}
command="bash hpc/valar/$(printf %q "$script")"
if [ $# -gt 0 ]; then command+=$(printf ' %q' "$@"); fi

if [ "$detach" = 1 ]; then
    log=runs/valar/logs/${script%.sh}-$(date +%Y%m%d-%H%M%S).log
    ssh "$host" bash -s <<EOF
set -e
cd $(printf %q "$dest")
mkdir -p runs/valar/logs
nohup $command > $(printf %q "$log") 2>&1 < /dev/null &
echo "Started on $host. Its log: $(printf %q "$dest/$log")"
EOF
else
    ssh "$host" bash -s <<EOF
set -e
cd $(printf %q "$dest")
$command < /dev/null
EOF
fi
