"""The model without the loop, for the blind ratings: same model, same library, same questions, no architecture.

    python scripts/plain_curiosity.py --home runs/records-1 --model mistral:7b-instruct --n 20 --out runs/plain/plain_records.json

It reads --n random passages from the library of the life in --home (the life is not changed), asks what puzzles
it and what exactly it does not know, and writes the curiosities to --out. Give that file (or its folder) to
scripts/rating_sheet.py as a run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curiosity_ai.config import load_config  # noqa: E402
from curiosity_ai.organism import CuriosityOrganism  # noqa: E402
from curiosity_ai.organism.plain import plain_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Curiosities of the model without the loop.")
    parser.add_argument("--home", required=True, help="a life whose library to read (not changed)")
    parser.add_argument("--out", required=True, help="where to write plain_records.json")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-url", default=None, metavar="URL", help="where Ollama listens (default: llm.base_url in the config)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not (Path(args.home) / "mind.json").exists():
        print(f"There is no life in {args.home}.")
        return 1
    config = load_config(args.config)
    if args.model:
        config.llm.model = args.model
    if args.ollama_url:
        config.llm.base_url = args.ollama_url
    config.organism.home = args.home
    config.organism.body.enabled = False
    config.organism.curiosity.architecture = "records"
    org = CuriosityOrganism(config)  # loaded only to read its library and use its prompts; it is not saved
    items = plain_records(org, n=args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} curiosities written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
