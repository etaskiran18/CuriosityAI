"""How often does the blind judge agree with you? Check it by hand before trusting it.

Each session writes ``judge_check.csv``: up to 20 random claim/quote pairs with
what the organism claimed and what the judge decided. Open it in a spreadsheet,
read each claim and quote, and write your own verdict in the last column:
supports, contradicts or neither. Then:

    python scripts/judge_agreement.py memory/organism/sessions/<id>/judge_check.csv [more.csv ...]

It prints how often the judge, and the organism itself, agreed with you. If the
judge agrees with you much less than about 80% of the time, use a larger judge
model (live.py --judge-model ...) before trusting the measurements.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

VERDICTS = {"supports", "contradicts", "neither"}
CLAIMED = {"confirmed": "supports", "contradicted": "contradicts", "belief": "supports", "finding": "supports"}


def normal(verdict: str) -> str:
    v = (verdict or "").strip().lower()
    for word in VERDICTS:
        if v.startswith(word[:4]):
            return word
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Agreement between you, the judge and the organism.")
    parser.add_argument("files", nargs="+", help="judge_check.csv files with your verdicts filled in")
    args = parser.parse_args()

    rows = []
    for name in args.files:
        with Path(name).open(encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f) if normal(r.get("your_verdict", ""))]
    if not rows:
        print("No rows with your verdict yet: fill the your_verdict column (supports, contradicts or neither).")
        return
    judge = sum(1 for r in rows if normal(r["judge_said"]) == normal(r["your_verdict"]))
    organism = sum(1 for r in rows if CLAIMED.get(r["organism_said"], "") == normal(r["your_verdict"]))
    print(f"Rows you checked: {len(rows)}")
    print(f"The judge agreed with you:    {judge}/{len(rows)} = {judge / len(rows):.0%}")
    print(f"The organism agreed with you: {organism}/{len(rows)} = {organism / len(rows):.0%}  (what it claimed before the judge)")
    disagreements = [r for r in rows if normal(r["judge_said"]) != normal(r["your_verdict"])]
    for r in disagreements[:10]:
        print(f"- heartbeat {r['heartbeat']}: judge said {r['judge_said']}, you said {r['your_verdict']}: {r['claim'][:90]}")


if __name__ == "__main__":
    main()
