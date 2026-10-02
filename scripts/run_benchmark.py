from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS = ROOT / "data" / "evaluation" / "curiosity_benchmark_topics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed curiosity benchmark topics one by one.")
    parser.add_argument("--topics", default=str(DEFAULT_TOPICS))
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    data = json.loads(Path(args.topics).read_text(encoding="utf-8"))
    topics = data.get("topics", [])[: args.limit]
    for i, topic in enumerate(topics, start=1):
        print(f"\n=== Benchmark {i}/{len(topics)} ===\n{topic}\n")
        subprocess.run(
            [sys.executable, "run.py", "--topic", topic, "--iterations", str(args.iterations)],
            cwd=str(ROOT),
            check=False,
        )


if __name__ == "__main__":
    main()
