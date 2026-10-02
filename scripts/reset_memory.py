from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for rel in [
    "memory/chroma",
    "memory/journal.jsonl",
    "memory/journal_v3.jsonl",
    "memory/journal_v4.jsonl",
    "memory/journal_v5.jsonl",
    "memory/journal_v6.jsonl",
    "memory/curiosity_theory_state_v6.json",
]:
    p = ROOT / rel
    if p.is_dir():
        shutil.rmtree(p)
        print(f"[removed] {p}")
    elif p.exists():
        p.unlink()
        print(f"[removed] {p}")
print("Memory reset complete.")
