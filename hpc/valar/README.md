# Curiosity lives on VALAR

These scripts run the curiosity organism on Koç University's VALAR cluster with language models of
different sizes. Each job starts its own Ollama server on its GPU node, loads the organism's model and
the judge's, lives one curiosity life, and writes the diary, the research map and the session report.
A last step gathers everything into one archive to read and share.

## The experiment

* **Topic:** the philosophy of curiosity (the default life; no `--topic`).
* **Architecture:** curiosity records (v10, [docs/CURIOSITY_RECORDS.md](../../docs/CURIOSITY_RECORDS.md)).
* **Models:** Qwen 2.5 with 3B, 7B, 14B, 32B and 72B parameters, one family so that only the size changes,
  plus Mistral 7B, the model of the laptop runs.
* **Judge:** `qwen2.5:32b` for every run. The judge decides what counts as evidence and as a satisfied
  curiosity; it is the measuring instrument, so it stays the same while the organism's model changes.
* **Library:** frozen. Every model reads the same texts (`data/philosophy_corpus`); no `--web`.
* **Seed and rhythm:** seed 1; a rest of 1 minute after every 15 minutes of thinking. After a rest it
  first goes back to a text that made it curious ("Spare time" in the diary).
* **Time:** 60 minutes per model by default.

Larger models think more slowly, so in the same hour they live fewer heartbeats (as a rough guess on a
48 GB card: a few dozen for 3B-7B, about ten for 72B; the pilot below measures it). Their diaries are then
shorter, and what needs many heartbeats (curiosities fading, waiting, returns to texts) shows less. For
the same amount of inquiry per model, give every model the same number of heartbeats with a generous time
limit: `--heartbeats 30 --minutes 240` stops at whichever comes first.

## Steps

**0. The code on VALAR.** On the login node:

```bash
git clone https://github.com/etaskiran18/CuriosityAI.git
cd CuriosityAI
git checkout claude/upbeat-lovelace-bsqcwj
```

(If GitHub asks for a password, copy the folder from your computer instead: `scp -r CuriosityAI <user>@<VALAR login>:~/`.)

**1. Your settings.** The defaults are partition, account and qos `ai`, as in VALAR's example job script.
If yours differ, or to choose the GPU type, write them into `hpc/valar/my_settings.sh` (git does not
track it), for example:

```bash
ACCOUNT=ai
GPU_TYPE=l40s                                # as SLURM names it; see below
STORE=/scratch/users/$USER/curiosity-hpc     # if your home has little space; default ~/curiosity-hpc
```

`sinfo -o "%20N %60G"` (or `valar-nodes`) lists the GPU types. Choose a 48 GB card (A40, L40S or
RTX A6000): it holds every model of the ladder with the judge, the 72B model on two of them. A 16 GB
Tesla T4 is too small. Without `GPU_TYPE` a job takes any GPU.

**2. Setup (once).** On the login node:

```bash
bash hpc/valar/setup.sh
```

It makes a Python environment, downloads Ollama (about 1.5 GB) and the models of the ladder (about 87 GB)
into `STORE`. Only downloads happen on the login node; no model runs there. Compute nodes may have no
internet, so every model a job uses must be downloaded here. More models later:
`bash hpc/valar/setup.sh models qwen2.5:72b`.

**3. A pilot of 10 minutes.**

```bash
bash hpc/valar/submit_ladder.sh --models qwen2.5:7b --minutes 10 --name pilot
```

When it has finished, `runs/valar/pilot/qwen2.5-7b/run_info.txt` says on which GPU it ran, whether all
layers of both models were on the GPU, and how many heartbeats it lived; the diary is in
`runs/valar/pilot/qwen2.5-7b/life/diary.md`.

**4. The ladder.**

```bash
bash hpc/valar/submit_ladder.sh
```

Six jobs, one per model; they run side by side when GPUs are free. `--dry-run` shows the `sbatch`
commands without submitting.

**5. Watching.** `squeue -u $USER` lists the jobs; a job's log is
`runs/valar/<batch>/<run>/slurm-<job>.out` (`tail -f` follows it).

**6. Collecting.** When all have finished:

```bash
bash hpc/valar/collect.sh
```

It writes `runs/valar/<batch>/share/` and `runs/valar/<batch>/<batch>-share.tar.gz`: per model the
diary, the research map, the report, the curiosity records and the judge sheet; a table comparing the
sessions (`comparison.md`); and a `README.md` saying what is where and what to look for. Copy the archive
to your computer with `scp` and send it on.

## More runs in the same batch

```bash
bash hpc/valar/submit_ladder.sh --name BATCH --models qwen2.5:32b --arch drive         # v9 as a control
bash hpc/valar/submit_ladder.sh --name BATCH --models qwen2.5:32b --ablate decision    # without deciding
bash hpc/valar/submit_ladder.sh --name BATCH --models "qwen2.5:7b qwen2.5:32b" --seed 2
bash hpc/valar/submit_ladder.sh --name BATCH --traps                                   # with the trap tests
```

`--exam` adds the exam on the philosophy of curiosity before and after (not counted in the minutes).
`bash hpc/valar/submit_ladder.sh --help` lists every option.

## What a job writes

In `runs/valar/<batch>/<run>/`:

* `run_info.txt`: model, judge, GPUs, layers on the GPU, settings, start and end.
* `slurm-<job>.out`: everything the job printed; `ollama.log`: the server's log.
* `config.yaml`: the configuration it lived with.
* `life/`: the life itself: `diary.md`, `research_map.md`, `mind.json` and
  `sessions/<id>/` with `report.md`, `metrics.json`, `curiosity_records.jsonl` and `judge_check.csv`.

## Beyond 72B

VALAR's A40 nodes have eight 48 GB cards each, so one job can hold far larger models. Qwen 2.5
stops at 72B; to go further, and to see whether what changes with size also changes in a second family,
the next ladder is Llama 3.1, whose 8B, 70B and 405B models were trained together:

```bash
bash hpc/valar/setup.sh models llama3.1:8b llama3.1:70b llama3.1:405b      # about 290 GB more
bash hpc/valar/submit_ladder.sh --name llama --models "llama3.1:8b llama3.1:70b llama3.1:405b" --heartbeats 15 --minutes 900
```

The 405B model is slow (a heartbeat can take tens of minutes), so give it heartbeats, not an hour; the
judge stays `qwen2.5:32b`. Other large models work the same way (`mistral-large:123b` takes three cards).
Models that think before they answer (reasoning models) need a pilot first: the organism asks for JSON
answers, and their thinking may get in the way.

## GPU memory

What a job needs: each model as downloaded, plus about 3 GB (more for the largest) for the context of 8192
tokens and work space; the judge `qwen2.5:32b` counts once more when it is another model. `submit_ladder.sh`
reads the sizes of the downloaded models and asks for that many GPUs (and computer memory):

| model | download | with the judge | 48 GB GPUs |
|---|---:|---:|---:|
| qwen2.5:3b | 1.9 GB | about 29 GB | 1 |
| qwen2.5:7b | 4.7 GB | about 32 GB | 1 |
| qwen2.5:14b | 9.0 GB | about 37 GB | 1 |
| qwen2.5:32b | 20 GB | about 24 GB (it is the judge) | 1 |
| qwen2.5:72b | 47 GB | about 76 GB | 2 |
| mistral:7b-instruct | 4.4 GB | about 32 GB | 1 |
| llama3.1:8b | 4.9 GB | about 32 GB | 1 |
| llama3.1:70b | 43 GB | about 72 GB | 2 |
| mistral-large:123b | 73 GB | about 103 GB | 3 |
| llama3.1:405b | 243 GB | about 280 GB | 7 |

With other cards, set `GPU_MEM_GB` in `my_settings.sh` (for example 80 for an 80 GB A100).

## When something goes wrong

* **`Invalid account or account/partition combination`**, **`Invalid qos`**: set `ACCOUNT`, `PARTITION`
  and `QOS` in `my_settings.sh` (`sacctmgr show assoc user=$USER` lists yours).
* **`Invalid generic resource (gres) specification`**: write `GPU_TYPE` exactly as `sinfo` shows it.
* **The job stops with "does not fit on the GPU"**: before living, every job checks that both models sit
  wholly on the GPU, because a model partly on the CPU is many times slower. Ask for larger or more GPUs
  (`GPU_TYPE`, `GPU_MEM_GB`), or set `ALLOW_CPU=1` before `submit_ladder.sh` to run anyway.
* **"No layer went to a GPU at all"**: the job had no GPU, or Ollama could not use it; `ollama.log`
  says what it found after "discovering available GPUs".
* **A model is missing**: `bash hpc/valar/setup.sh models NAME` on the login node.
* **No space left**: put `STORE` on scratch or project storage. VALAR keeps no backups: copy the results
  you want to keep to your computer.
* **The job reached its time limit**: the mind is saved and the report written all the same (the job's
  end is treated like Ctrl+C).

VALAR's own pages: [the cluster](https://help.ku.edu.tr/ithelp/valar-hpc-dc2-cluster) and
[job scripts](https://help.ku.edu.tr/ithelp/how-do-i-write-and-submit-a-jobscript).

## The files

| file | what it does |
|---|---|
| `common.sh` | default settings and shared helpers; your `my_settings.sh` overrides the settings |
| `setup.sh` | the Python environment, Ollama and the models (login node, once) |
| `submit_ladder.sh` | submits one job per model |
| `curiosity_job.sh` | the job: its own Ollama server, the models loaded and checked, one life |
| `prepare_server.py` | waits for the server, loads the models, checks that they are on the GPU |
| `collect.sh`, `collect.py` | gather a batch into `share/` and an archive |
