"""Compare session reports side by side.

    python scripts/compare_sessions.py memory/exp_curiosity/sessions/* memory/exp_random/sessions/*
    python scripts/compare_sessions.py A B --csv results.csv

Each argument is a session folder (or its metrics.json). The table is Markdown,
so it can be pasted into notes or a paper.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROWS = [
    ("policy", "policy"),
    ("model", "model"),
    ("web", "web library"),
    ("minutes", "minutes"),
    ("heartbeats", "heartbeats"),
    ("heartbeats_per_hour", "heartbeats per hour"),
    ("rest_minutes", "minutes resting"),
    ("distinct_questions", "questions visited"),
    ("questions_born", "questions born"),
    ("questions_settled", "questions settled"),
    ("beliefs_new", "new beliefs"),
    ("beliefs_grounded", "grounded beliefs"),
    ("beliefs_doubted", "beliefs doubted"),
    ("quotes_verified", "quotes verified"),
    ("quotes_rejected", "invented quotes caught"),
    ("fabrication_rate", "fabrication rate"),
    ("mean_surprise", "mean surprise"),
    ("surprise_first_half", "surprise, first half"),
    ("surprise_second_half", "surprise, second half"),
    ("mean_informativeness", "texts addressed"),
    ("mean_confidence_change", "mean confidence change"),
    ("revisited_questions", "questions revisited"),
    ("surprise_change_on_revisits", "surprise change on revisits"),
    ("acquisitions", "texts acquired"),
    ("failed_heartbeats", "failed heartbeats"),
]


def load(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8"))


def per_hour(m: dict, key: str) -> float:
    return round(m[key] / (m["minutes"] / 60), 1) if m.get("minutes") else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare curiosity organism sessions.")
    parser.add_argument("sessions", nargs="+", help="session folders or metrics.json files")
    parser.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = parser.parse_args()

    sessions = [load(s) for s in args.sessions]
    names = [m.get("session_id", Path(s).name) for m, s in zip(sessions, args.sessions)]
    rows = [(label, [m.get(key, "") for m in sessions]) for key, label in ROWS]
    # Rates make runs of different length comparable.
    rows.append(("grounded beliefs per hour", [per_hour(m, "beliefs_grounded") for m in sessions]))
    rows.append(("questions born per hour", [per_hour(m, "questions_born") for m in sessions]))

    def cell(value) -> str:
        return f"{value:.3g}" if isinstance(value, float) else str(value)

    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    for label, values in rows:
        print(f"| {label} | " + " | ".join(cell(v) for v in values) + " |")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", *names])
            for label, values in rows:
                writer.writerow([label, *values])
        print(f"\nWritten to {args.csv}")


if __name__ == "__main__":
    main()
