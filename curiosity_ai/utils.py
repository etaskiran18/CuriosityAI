from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_id(*parts: str, length: int = 16) -> str:
    raw = "::".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:length]


def write_jsonl(path: str | Path, item: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:] if limit else rows


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract first JSON object from an LLM response, tolerant of markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    repaired = close_truncated_json(text)
    if repaired is not None:
        return repaired
    raise ValueError(f"Could not extract JSON object from response: {text[:500]}")


def close_truncated_json(text: str, max_attempts: int = 400) -> dict[str, Any] | None:
    """Best effort for a reply cut off by the length limit: keep every value that was complete.

    Walks the text once, remembering each point where a value had just ended
    (a closed string, object or list, or the place before a comma), and which
    brackets were open there. Then, from the last such point backwards, it cuts
    the text there and closes the open brackets until the result parses.
    """
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    stack: list[str] = []
    cuts: list[tuple[int, str]] = []
    in_string = escaped = False
    for i, ch in enumerate(s):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                cuts.append((i + 1, "".join(reversed(stack))))
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            cuts.append((i + 1, "".join(reversed(stack))))
            if not stack:
                break
        elif ch == ",":
            cuts.append((i, "".join(reversed(stack))))
    for end, closers in reversed(cuts[-max_attempts:]):
        candidate = s[:end].rstrip().rstrip(",") + closers
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_score(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def jaccard(a: str, b: str) -> float:
    tokenize = lambda s: set(re.findall(r"[a-zA-Z0-9_]+", s.lower()))
    sa, sb = tokenize(a), tokenize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def strip_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Read simple YAML-like front matter. Keeps dependency surface small."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    raw_meta = parts[1]
    body = parts[2].lstrip()
    meta: dict[str, str] = {}
    for line in raw_meta.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta, body


def line_range_for_char_span(text: str, start: int, end: int) -> tuple[int, int]:
    before = text[:start]
    span = text[start:end]
    line_start = before.count("\n") + 1
    line_end = line_start + span.count("\n")
    return line_start, line_end
