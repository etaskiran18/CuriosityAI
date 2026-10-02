from __future__ import annotations

import json
from pathlib import Path
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="datasets/curiosity_sft.jsonl")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        print(f"No dataset found at {path}")
        return
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()[: args.limit], start=1):
        print(f"\n--- Example {i} ---")
        print(json.dumps(json.loads(line), indent=2, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
