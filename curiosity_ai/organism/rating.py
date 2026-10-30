"""Blind rating sheets: is what it is curious about intellectual curiosity, in İnan's sense? (docs/CURIOSITY_RECORDS.md)

Questions from several runs (conditions) are mixed, their source hidden, and given to raters. Level 1 shows only
the question (every condition has one); level 2 shows the whole curiosity record and asks for five 0-2 scores.
A separate key file says which run each item came from; ``score`` joins the filled sheets with the key.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

YES_NO = ("intellectual_curiosity", "worth_asking")
SCORES = ("unknown_specific", "known_correct", "answer_form_clear", "interest_real", "reason_sensible")
EXPLAIN = {
    "intellectual_curiosity": "yes / no / unsure: is this an intellectual curiosity, the wish to know a specific unknown?",
    "worth_asking": "yes / no / unsure: is it worth asking, for someone who works on this topic?",
    "unknown_specific": "0-2: does it name one specific thing that is unknown?",
    "known_correct": "0-2: is what it says it knows correct and relevant?",
    "answer_form_clear": "0-2: is it clear what would answer it?",
    "interest_real": "0-2: is the interest real: does it connect to the inquiry?",
    "reason_sensible": "0-2: is the reason for pursuing it now or later sensible?",
}


def collect(path: Path) -> list[dict[str, Any]]:
    """The curiosities a run gave birth to: from a life's mind.json (or its folder), or from plain_records.json."""
    path = Path(path)
    if path.is_dir():
        path = path / "mind.json" if (path / "mind.json").exists() else path / "plain_records.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):  # plain_records.json: the model without the loop
        return [dict(item, id=item.get("id", f"P{i}")) for i, item in enumerate(data, start=1)]
    items = []
    for q in data.get("questions", {}).values():
        if q.get("trigger") in ("seed", "human"):
            continue  # what it was given, not what it became curious about
        items.append({
            "id": q["id"], "question": q["text"], "unknown": q.get("unknown", ""), "known": "; ".join(q.get("known", [])),
            "answer_form": q.get("answer_form", ""), "interest": q.get("interest", ""),
            "reason": q.get("decision_reason") or q.get("wish_reason", ""),
        })
    return items


def make_sheet(runs: dict[str, Path], sheet: Path, key: Path, *, n: int = 15, level: int = 1, seed: int = 1) -> int:
    """Up to ``n`` items per run, shuffled, written to ``sheet``; which run each came from goes to ``key``."""
    rng = random.Random(seed)
    rows: list[tuple[str, dict[str, Any]]] = []
    for label, path in runs.items():
        items = collect(path)
        if level == 2:
            items = [i for i in items if i.get("unknown")]
        rows += [(label, item) for item in rng.sample(items, min(n, len(items)))]
    rng.shuffle(rows)
    shown = ["question"] if level == 1 else ["question", "unknown", "known", "answer_form", "interest", "reason"]
    asked = list(YES_NO) if level == 1 else [*SCORES, "intellectual_curiosity"]
    sheet.parent.mkdir(parents=True, exist_ok=True)
    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item", *shown, *asked, "notes"])
        writer.writerow(["(how to rate)", *([""] * len(shown)), *(EXPLAIN[a] for a in asked), ""])
        for i, (_, item) in enumerate(rows, start=1):
            writer.writerow([i, *(item.get(col, "") for col in shown), *([""] * len(asked)), ""])
    with key.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "run", "id"])
        for i, (label, item) in enumerate(rows, start=1):
            writer.writerow([i, label, item.get("id", "")])
    return len(rows)


def _read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["item"]: row for row in csv.DictReader(f) if row.get("item", "").isdigit()}


def _yes(value: str) -> float | None:
    v = (value or "").strip().lower()
    return 1.0 if v.startswith("y") else 0.0 if v.startswith("n") else None


def _number(value: str) -> float | None:
    try:
        return float((value or "").strip())
    except ValueError:
        return None


def score(sheets: list[Path], key: Path) -> dict[str, Any]:
    """Per run: the mean of each 0-2 score and the share of "yes"; with two or more raters, how far they agree."""
    runs = {row["item"]: row["run"] for row in _read(key).values()}
    filled = [_read(s) for s in sheets]
    by_run: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rater in filled:
        for item, row in rater.items():
            run = runs.get(item)
            if run is None:
                continue
            for col in YES_NO:
                if col in row and _yes(row[col]) is not None:
                    by_run[run][col].append(_yes(row[col]))
            for col in SCORES:
                if col in row and _number(row[col]) is not None:
                    by_run[run][col].append(_number(row[col]))
    result: dict[str, Any] = {
        "runs": {run: {col: round(sum(v) / len(v), 3) for col, v in cols.items() if v} | {"n": max(len(v) for v in cols.values())}
                 for run, cols in by_run.items() if cols},
    }
    if len(filled) >= 2:
        result["agreement"] = agreement(filled[0], filled[1])
    return result


def agreement(a: dict[str, dict[str, str]], b: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Two raters: Cohen's kappa on intellectual_curiosity (yes/no), and the mean difference of the 0-2 scores."""
    pairs = [(_yes(a[i].get("intellectual_curiosity", "")), _yes(b[i].get("intellectual_curiosity", ""))) for i in a if i in b]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    out: dict[str, Any] = {"items_both_rated": len(pairs)}
    if pairs:
        observed = sum(1 for x, y in pairs if x == y) / len(pairs)
        pa, pb = sum(x for x, _ in pairs) / len(pairs), sum(y for _, y in pairs) / len(pairs)
        expected = pa * pb + (1 - pa) * (1 - pb)
        out["percent_agreement"] = round(observed, 3)
        out["cohens_kappa"] = round((observed - expected) / (1 - expected), 3) if expected < 1 else None
    diffs = [abs(_number(a[i][c]) - _number(b[i][c])) for i in a if i in b for c in SCORES
             if c in a[i] and c in b[i] and _number(a[i][c]) is not None and _number(b[i][c]) is not None]
    if diffs:
        out["mean_score_difference"] = round(sum(diffs) / len(diffs), 3)
    return out
