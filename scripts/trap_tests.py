"""Trap tests for the records architecture: situations whose right outcome is known in advance.

    python scripts/trap_tests.py --home runs/traps --model qwen2.5:3b --heartbeats 8
    python scripts/trap_tests.py --home runs/traps --from-home memory/research/my-topic --trap noise,unreachable

Each trap gets a life of its own under --home (a fresh philosophy life, or a copy of the life in --from-home),
is lived for --heartbeats heartbeats, and is judged passed, failed, or not walked into. The results go to
<home>/trap_results.md and .json; every trap life keeps its diary and session report. See
docs/CURIOSITY_RECORDS.md.

  noise        scrambled texts are new, so they draw attention, but no curiosity should come of them
  unreachable  a person asks what nobody could ever find out: it should let it go within a few looks
  hidden       the answer to a waiting curiosity is planted where it will read for other questions:
               it should recognize it and end the curiosity without pursuing it
  old          the words that ended a curiosity are shown again: it should not take it up again
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curiosity_ai.config import CURIOSITY_PARTS, load_config  # noqa: E402
from curiosity_ai.organism.traps import HIDDEN, TRAPS, run_trap, write_results  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Trap tests for the curiosity records architecture.")
    parser.add_argument("--home", required=True, help="folder for the trap lives and the results")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-url", default=None, metavar="URL", help="where Ollama listens (default: llm.base_url in the config)")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--heartbeats", type=int, default=8)
    parser.add_argument("--trap", default=",".join(TRAPS), help=f"comma-separated: {', '.join(TRAPS)}")
    parser.add_argument("--from-home", default=None, help="set the traps in copies of this life instead of fresh ones")
    parser.add_argument("--question", default=None, help="the unreachable question (default: one for the life's mode)")
    parser.add_argument("--hidden-question", default=HIDDEN["question"])
    parser.add_argument("--hidden-unknown", default=HIDDEN["unknown"])
    parser.add_argument("--hidden-answer-form", default=HIDDEN["answer_form"])
    parser.add_argument("--hidden-text", default=None, metavar="FILE", help="a text that answers the hidden curiosity")
    parser.add_argument("--ablate", default=None, help="switch parts off, as in live.py: " + ", ".join(CURIOSITY_PARTS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rest", action="store_true", help="keep the body's rests and temperature guard on (slower)")
    args = parser.parse_args()

    traps = [t.strip() for t in args.trap.split(",") if t.strip()]
    unknown = [t for t in traps if t not in TRAPS]
    if unknown:
        print(f"Unknown trap(s): {', '.join(unknown)}. Choose from: {', '.join(TRAPS)}")
        return 1
    config = load_config(args.config)
    if args.model:
        config.llm.model = args.model
    if args.ollama_url:
        config.llm.base_url = args.ollama_url
    if args.judge_model:
        config.organism.judge.model = args.judge_model
    config.organism.random_seed = args.seed
    config.organism.body.enabled = args.rest
    config.organism.librarian.enabled = False  # a frozen library: the same texts for every trap
    for part in [p.strip() for p in (args.ablate or "").split(",") if p.strip()]:
        if part not in CURIOSITY_PARTS:
            print(f"--ablate knows only {', '.join(CURIOSITY_PARTS)}")
            return 1
        setattr(config.organism.curiosity, part, False)
    hidden = {
        "question": args.hidden_question, "unknown": args.hidden_unknown, "answer_form": args.hidden_answer_form,
        "text": Path(args.hidden_text).read_text(encoding="utf-8") if args.hidden_text else HIDDEN["text"],
    }
    home = Path(args.home)
    results = []
    for trap in traps:
        print(f"Trap '{trap}': living {args.heartbeats} heartbeats in {home / trap} ...", flush=True)
        result = run_trap(
            trap, config, home / trap, heartbeats=args.heartbeats,
            from_home=Path(args.from_home) if args.from_home else None, hidden=hidden, question=args.question,
        )
        print(f"  {'passed' if result.passed else 'FAILED' if result.passed is False else 'not walked into'}: {result.evidence}")
        results.append(result)
    report = write_results(results, home / "trap_results.md")
    print(f"\nResults: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
