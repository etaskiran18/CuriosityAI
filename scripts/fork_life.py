"""What if it had pursued it? Fork a life at a deferred curiosity and live it both ways.

    python scripts/fork_life.py --home memory/organism --out runs/fork-Q7 --question Q7 --heartbeats 5
    python scripts/fork_life.py --home memory/organism --out runs/fork-latest      # the latest deferred curiosity

The life in --home is not changed: two copies are made under --out. In "as-decided" it lives on as it decided;
in "pursued" it pursues only that curiosity. runs/.../fork_result.md says what deferring cost or saved.
Repeat for many deferred curiosities (and seeds) to measure how good its decisions to wait are.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curiosity_ai.config import load_config  # noqa: E402
from curiosity_ai.organism.fork import fork  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fork a life at a deferred curiosity and live both ways.")
    parser.add_argument("--home", required=True, help="the life to fork (left unchanged)")
    parser.add_argument("--out", required=True, help="folder for the two copies and the result")
    parser.add_argument("--question", default=None, help="the deferred curiosity, e.g. Q7 (default: the latest deferred)")
    parser.add_argument("--heartbeats", type=int, default=5)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-url", default=None, metavar="URL", help="where Ollama listens (default: llm.base_url in the config)")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config.llm.model = args.model
    if args.ollama_url:
        config.llm.base_url = args.ollama_url
    if args.judge_model:
        config.organism.judge.model = args.judge_model
    config.organism.random_seed = args.seed
    config.organism.body.enabled = False
    config.organism.librarian.enabled = False  # both copies read the same texts
    qid, branches, verdict = fork(config, Path(args.home), Path(args.out), question_id=args.question, heartbeats=args.heartbeats)
    print(f"{qid}: {verdict}.")
    for b in branches:
        print(f"  {b.name}: {b.status}, answered {b.satisfied}, {b.looks} looks in {b.heartbeats} heartbeats")
    print(f"Result: {Path(args.out) / 'fork_result.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
