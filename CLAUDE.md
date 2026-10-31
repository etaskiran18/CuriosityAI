# CuriosityAI: notes for Claude

## The project

A curiosity organism (`live.py`): it chooses its questions by curiosity, reads its library, predicts what the
texts will say, debates, settles and keeps a diary. v10 adds curiosity records after İlhan İnan and Ady et al.
(`--architecture records`, `docs/CURIOSITY_RECORDS.md`). What changed and why: `CHANGELOG_v9.md`,
`CHANGELOG_v10.md`. Tests: `python -m pytest -q`.

The owner writes in Turkish: answer in Turkish, in plain words.

## Now: the first complete test on VALAR

Everything is in `hpc/valar/` (its `README.md` explains each step). From this computer:

1. Read-only checks: `ssh valar hostname`; `ssh valar 'sinfo -o "%20N %60G"'` (the GPU type names);
   `ssh valar 'squeue -u $USER'` (the owner's other jobs run too and share the limits);
   `ssh valar 'df -h /scratch'`.
2. Settings that differ from `hpc/valar/common.sh` go into `hpc/valar/my_settings.sh` on this computer (git
   ignores it; the sync copies it): above all `GPU_TYPE`, a 48 GB card (A40, L40S or RTX A6000).
3. `bash hpc/valar/sync_to_valar.sh`: the code to `/scratch/<user>/CuriosityAI`, checked on both sides.
4. `bash hpc/valar/remote.sh --detach setup.sh`: Python environment, Ollama and the models (about 87 GB) in
   `/scratch/<user>/curiosity-hpc`; it prints the log to follow.
5. Pilot: `bash hpc/valar/remote.sh submit_ladder.sh --models qwen2.5:7b --minutes 10 --name pilot`. Its
   `run_info.txt` on VALAR says whether every layer sat on the GPU and how many heartbeats it lived.
6. The ladder (qwen2.5 3B to 72B and mistral 7B, judge qwen2.5:32b for all):
   `bash hpc/valar/remote.sh submit_ladder.sh --heartbeats 30 --minutes 240 --at-once 3`
   (or `--minutes 60` for an hour each).
7. `bash hpc/valar/remote.sh collect.sh <batch>`, then `bash hpc/valar/fetch_from_valar.sh <batch>`: the
   archive of diaries, research maps and reports, to share with İlhan İnan.

Later: the Llama 3.1 ladder up to 405B ("Beyond 72B" in `hpc/valar/README.md`).

## Cluster rules (from the lab's VALAR guide)

- Cluster: `ssh valar` (Slurm, partition ai, always `--mem`, at most 8 running jobs and 50 submitted per user;
  array tasks count).
- Never submit, cancel or change jobs without the owner's explicit OK. Never touch jobs this project did not submit.
- Raw data never leave this machine; only code and derived inputs go to VALAR.
- Arrays run at most 7 tasks at once (`%7`).
- Never read, write or ask for passwords.
- Commands go over `ssh valar bash -s` (as `hpc/valar/remote.sh` does), not through nested quoting.
- Keep a record of every job id and decision: each batch's `runs/valar/<batch>/jobs.md` on VALAR.
