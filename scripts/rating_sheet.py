"""Blind rating sheets for curiosity records (docs/CURIOSITY_RECORDS.md, "Is it curiosity?").

Make a sheet from several runs (label=folder), give it to raters, then score the filled copies:

    python scripts/rating_sheet.py make --run drive=runs/drive-1 --run records=runs/records-1 \
        --run plain=runs/plain --n 15 --level 1 --sheet sheet.csv --key key.csv
    python scripts/rating_sheet.py score --key key.csv rater1.csv rater2.csv

Level 1 shows only the questions (all conditions have them) and asks: intellectual curiosity? worth asking?
Level 2 shows whole records (unknown, what is known, what would answer it, interest, reason) with five 0-2
scores. Keep key.csv away from the raters. A run is a life's folder, or a plain_records.json from
scripts/plain_curiosity.py (the model without the loop).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curiosity_ai.organism.rating import make_sheet, score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Blind rating sheets for curiosity records.")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make", help="make a sheet and its key")
    make.add_argument("--run", action="append", required=True, metavar="LABEL=FOLDER")
    make.add_argument("--n", type=int, default=15, help="items per run")
    make.add_argument("--level", type=int, choices=(1, 2), default=1)
    make.add_argument("--sheet", default="sheet.csv")
    make.add_argument("--key", default="key.csv")
    make.add_argument("--seed", type=int, default=1)
    scores = sub.add_parser("score", help="score filled sheets")
    scores.add_argument("--key", required=True)
    scores.add_argument("sheets", nargs="+")
    args = parser.parse_args()

    if args.command == "make":
        runs = {}
        for spec in args.run:
            label, _, folder = spec.partition("=")
            if not folder:
                print(f"--run needs LABEL=FOLDER, not {spec!r}")
                return 1
            runs[label] = Path(folder)
        n = make_sheet(runs, Path(args.sheet), Path(args.key), n=args.n, level=args.level, seed=args.seed)
        print(f"{n} items in {args.sheet}; which run each came from is in {args.key} (keep it from the raters).")
        return 0
    print(json.dumps(score([Path(s) for s in args.sheets], Path(args.key)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
