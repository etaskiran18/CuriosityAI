"""Gather what to read and share from a batch of runs (hpc/valar/collect.sh calls this).

    python hpc/valar/collect.py runs/valar/20261026-1500

For every run it copies the diary, the research map, the newest session's report, metrics, curiosity
records and judge sheet, and the trap results, into BATCH/share/<run>/; it writes a table comparing the
sessions (comparison.md and .csv), an index (README.md) and BATCH/<batch>-share.tar.gz to send on.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SESSION_FILES = ("report.md", "metrics.json", "curiosity_records.jsonl", "judge_check.csv")
HOW_TO_READ = """\
## Files of each run

* `diary.md`: the life in its own words, heartbeat by heartbeat.
* `research_map.md`: what it is curious about, the curiosities that ended and how, the texts that made it
  curious and what it has come to be interested in.
* `report.md` and `metrics.json`: the session in numbers.
* `curiosity_records.jsonl`: every curiosity record (what exactly is unknown, what would answer it, the
  decision to pursue or wait, how it ended).
* `judge_check.csv`: claims and the quotes offered for them, with the judge's verdict, for a human check.
* `trap_results.md` (trap runs): situations whose right outcome is known in advance.
* `run_info.txt`: model, judge, GPU, time and settings of the run.

## What to look for in a diary

"What I want to know", "Curiosity satisfied", "I am curious about ..., but I do not pursue it now,
because ...", "faded" and "Spare time: I went back to ...".

Larger models think more slowly: in the same time they live fewer heartbeats, so compare them per
heartbeat (the table gives heartbeats per hour).
"""


def by_size(run: Path) -> tuple:
    """Runs in order of family and model size: qwen2.5-3b before qwen2.5-14b."""
    match = re.search(r"-(\d+(?:\.\d+)?)b(?=$|-)", run.name)
    return (run.name.replace(match.group(0), "", 1), float(match.group(1)), run.name) if match else (run.name, 0.0, run.name)


def newest_session(life: Path) -> Path | None:
    sessions = sorted(p for p in (life / "sessions").glob("*") if (p / "metrics.json").is_file())
    return sessions[-1] if sessions else None


def info(run: Path) -> dict[str, str]:
    path = run / "run_info.txt"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.is_file() else []
    return {k.strip(): v.strip() for k, sep, v in (line.partition(":") for line in lines) if sep}


def collect(batch: Path) -> Path:
    share = batch / "share"
    if share.exists():
        shutil.rmtree(share)
    share.mkdir(parents=True)
    rows, sessions = [], []
    for run in sorted((p for p in batch.iterdir() if p.is_dir() and p.name != "share"), key=by_size):
        dest = share / run.name
        dest.mkdir()
        copied = []
        for source in (run / "run_info.txt", run / "life" / "diary.md", run / "life" / "research_map.md",
                       run / "traps" / "trap_results.md"):
            if source.is_file():
                shutil.copy2(source, dest / source.name)
                copied.append(source.name)
        session = newest_session(run / "life")
        metrics: dict = {}
        if session is not None:
            sessions.append(session)
            for name in SESSION_FILES:
                if (session / name).is_file():
                    shutil.copy2(session / name, dest / name)
                    copied.append(name)
            metrics = json.loads((session / "metrics.json").read_text(encoding="utf-8"))
        about = info(run)
        if not copied:
            dest.rmdir()
            continue
        rows.append((run.name, about, metrics))

    lines = [f"# Curiosity lives: {batch.name}", "", "| run | model | judge | heartbeats | minutes | GPUs | ended |",
             "|---|---|---|---:|---:|---|---|"]
    for name, about, m in rows:
        lines.append(f"| {name} | {about.get('model', m.get('model', ''))} | {about.get('judge', m.get('judge_model', ''))} | "
                     f"{m.get('heartbeats', '')} | {m.get('minutes', '')} | {about.get('gpus', '')} | "
                     f"{about.get('finished', 'not yet' if about else '')} |")
    if sessions:
        table = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "compare_sessions.py"), *map(str, sessions), "--csv", str(share / "comparison.csv")],
            capture_output=True, text=True, check=True,
        ).stdout
        table = table.split("\n\nWritten to ")[0].rstrip() + "\n"
        (share / "comparison.md").write_text(table, encoding="utf-8")
        lines += ["", "The sessions side by side: `comparison.md` (and `comparison.csv`)."]
    lines += ["", HOW_TO_READ]
    (share / "README.md").write_text("\n".join(lines), encoding="utf-8")

    archive = batch / f"{batch.name}-share.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(share, arcname=f"{batch.name}-share")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Gather the diaries and reports of a batch of runs.")
    parser.add_argument("batch", help="the batch folder, e.g. runs/valar/20261026-1500")
    args = parser.parse_args()
    batch = Path(args.batch).resolve()
    if not batch.is_dir():
        print(f"There is no batch at {batch}.")
        return 1
    archive = collect(batch)
    print(f"Collected in {batch / 'share'} (start with README.md there).")
    print(f"To send on: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
