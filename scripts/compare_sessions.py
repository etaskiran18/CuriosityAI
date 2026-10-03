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
    ("judge_model", "judge model"),
    ("mode", "mode"),
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
    ("predictions", "predictions made"),
    ("hedged_rate", "predictions hedged"),
    ("predictions_confirmed", "predictions confirmed"),
    ("predictions_contradicted", "predictions contradicted"),
    ("mean_brier", "mean Brier score (lower is better)"),
    ("brier_first_half", "Brier, first half"),
    ("brier_second_half", "Brier, second half"),
    ("judge_pairs", "claim/quote pairs judged"),
    ("judge_agreement", "judge agreed with the organism"),
    ("judge_rejected", "quotes the judge rejected"),
    ("concession_rate", "debates conceded"),
    ("skeptic_quoted_rate", "debates where the skeptic quoted a text"),
    ("vague_answers", "answers too vague to be wrong"),
    ("hedged_answers", "answers hedged (may, could)"),
    ("strawman_falsifiers", "answers wrong only if nothing were at play"),
    ("paper_doubts", "doubts refused (no contradiction)"),
    ("mean_on_topic", "closeness to the topic"),
    ("questions_set_aside", "questions set aside (off topic)"),
    ("questions_held_back", "questions kept for a second look"),
    ("dialogue_unverified_citations", "debate turns citing unseen papers"),
    ("acquisitions", "texts acquired"),
    ("library_rejected", "texts found but off topic"),
    ("library_busy", "searches blocked by a busy source"),
    ("exam_before", "exam before"),
    ("exam_after", "exam after"),
    ("exam_gain", "exam gain"),
    ("exam_grounded_after", "grounded exam score after"),
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
        if value is None:
            return ""
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
