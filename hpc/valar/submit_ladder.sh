#!/usr/bin/env bash
# Submit curiosity lives to VALAR: one job per model, all with the same judge, library, seed and rhythm.
#
#   bash hpc/valar/submit_ladder.sh                                    # the size ladder, an hour each
#   bash hpc/valar/submit_ladder.sh --heartbeats 30 --minutes 240      # the same 30 heartbeats for every model
#   bash hpc/valar/submit_ladder.sh --models qwen2.5:7b --minutes 10 --name pilot     # a short pilot
#   bash hpc/valar/submit_ladder.sh --models qwen2.5:32b --arch drive --name BATCH    # a v9 control, same batch
#   bash hpc/valar/submit_ladder.sh --dry-run                          # only show what would be submitted
#
# Options:
#   --models "A B"   the models (default: LADDER in common.sh)
#   --judge M        the judge of every run (default: JUDGE_MODEL in common.sh)
#   --minutes N      how long each lives (default 60)
#   --heartbeats N   also stop after N heartbeats, whichever comes first
#   --seed N         random seed (default 1)
#   --arch A         records (default) or drive
#   --ablate PARTS   switch parts of the records architecture off, e.g. decision,attention
#   --exam           the exam before and after (adds up to an hour to each job)
#   --traps          also the four trap tests for each model, in jobs of their own
#   --name NAME      the batch folder in runs/valar (default: date and time); an existing batch gets more runs
#   --dry-run        print the sbatch commands and submit nothing
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

models=$LADDER
judge=$JUDGE_MODEL
minutes=60
heartbeats=""
seed=1
arch=records
ablate=""
exam=0
traps=0
name=$(date +%Y%m%d-%H%M)
dry=0
while [ $# -gt 0 ]; do
    case "$1" in
        --models) models=$2; shift 2 ;;
        --judge) judge=$2; shift 2 ;;
        --minutes) minutes=$2; shift 2 ;;
        --heartbeats) heartbeats=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        --arch) arch=$2; shift 2 ;;
        --ablate) ablate=$2; shift 2 ;;
        --exam) exam=1; shift ;;
        --traps) traps=1; shift ;;
        --name) name=$2; shift 2 ;;
        --dry-run) dry=1; shift ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done
case "$arch" in records|drive) ;; *) echo "--arch is records or drive" >&2; exit 1 ;; esac
if [ -n "$ablate" ] && [ "$arch" != records ]; then
    echo "--ablate switches off parts of the records architecture" >&2
    exit 1
fi
if [ "$dry" = 0 ] && ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch not found: run this on the VALAR login node (or use --dry-run)." >&2
    exit 1
fi
if [ "$dry" = 0 ] && { [ ! -x "$PY" ] || [ ! -x "$OLLAMA" ]; }; then
    echo "No Python environment or Ollama in $STORE: run  bash hpc/valar/setup.sh  first." >&2
    exit 1
fi

BATCH=$RESULTS/$name
hhmm() {  # minutes -> HH:MM:00
    printf '%02d:%02d:00' $(($1 / 60)) $(($1 % 60))
}

submit() {  # label mode time-limit-minutes model
    local label=$1 mode=$2 limit=$3 model=$4
    local dir=$BATCH/$label gpus mem gres
    gpus=$(gpus_needed "$model" "$judge")
    mem=$(ram_needed "$model" "$judge")G
    gres=gpu:${GPU_TYPE:+$GPU_TYPE:}$gpus
    if [ -e "$dir" ]; then
        echo "  $label: skipped, it was submitted before ($dir exists; delete it to submit again)"
        return
    fi
    local cmd=(sbatch --parsable --export=ALL --job-name "cur-$label" --partition "$PARTITION" --account "$ACCOUNT"
               --qos "$QOS" --gres "$gres" --nodes 1 --ntasks 1 --cpus-per-task "$CPUS" --mem "$mem"
               --time "$(hhmm "$limit")" --output "$dir/slurm-%j.out")
    if [ -n "$MAIL_USER" ]; then cmd+=(--mail-user "$MAIL_USER" --mail-type "END,FAIL"); fi
    cmd+=("$VALAR_DIR/curiosity_job.sh")
    if [ "$dry" = 1 ]; then
        echo "  $label: MODEL=$model JUDGE=$judge MODE=$mode ${cmd[*]}"
        return
    fi
    mkdir -p "$dir"
    local id
    id=$(REPO=$REPO MODEL=$model JUDGE=$judge MINUTES=$minutes HEARTBEATS=$heartbeats SEED=$seed ARCH=$arch \
         ABLATE=$ablate EXAM=$exam MODE=$mode LABEL=$label RUN_DIR=$dir "${cmd[@]}")
    echo "  $label: job $id ($gpus GPU, at most $(hhmm "$limit"))"
    echo "| $id | $label | $model | $judge | $mode | $gpus | $(hhmm "$limit") |" >>"$BATCH/jobs.md"
}

if [ "$dry" = 0 ]; then
    mkdir -p "$BATCH"
    ln -sfn "$BATCH" "$RESULTS/latest"
    if [ ! -f "$BATCH/jobs.md" ]; then
        {
            echo "# Batch $name"
            echo
            echo "Code: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown). Library frozen (no --web)."
            echo
            echo "| job | run | model | judge | mode | GPUs | time limit |"
            echo "|---|---|---|---|---|---:|---|"
        } >"$BATCH/jobs.md"
    fi
fi
echo "Batch $BATCH: judge $judge, $minutes minutes${heartbeats:+ or $heartbeats heartbeats}, seed $seed, architecture $arch${ablate:+ without $ablate}"
suffix=""
if [ "$arch" = drive ]; then suffix+=-drive; fi
if [ -n "$ablate" ]; then suffix+=-no-${ablate//,/-}; fi
if [ "$seed" != 1 ]; then suffix+=-seed$seed; fi
for model in $models; do
    label=$(slug "$model")$suffix
    limit=$((minutes + 30 + exam * 60))   # half an hour for loading the models and writing the reports
    submit "$label" life "$limit" "$model"
    if [ "$traps" = 1 ]; then
        big=$(awk -v b="$(model_billions "$model")" 'BEGIN { print (b >= 60) ? 1 : 0 }')
        submit "traps-$label" traps $((big ? 8 * 60 : 4 * 60)) "$model"
    fi
done
if [ "$dry" = 0 ]; then
    echo
    echo "Watch them:   squeue -u \$USER"
    echo "A job's log:  $BATCH/<run>/slurm-<job>.out"
    echo "When all have finished:  bash hpc/valar/collect.sh $name"
fi
