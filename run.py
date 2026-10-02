from __future__ import annotations

import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from curiosity_ai.config import load_config
from curiosity_ai.controller import LoopController


console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the closed-loop Curiosity AI system.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--topic", required=True, help="Initial philosophical topic")
    parser.add_argument("--iterations", type=int, default=None, help="Number of loop iterations")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    controller = LoopController(config)
    records = controller.run(args.topic, args.iterations)

    table = Table(title=f"Curiosity Run: {controller.run_id}")
    table.add_column("Iter", justify="right")
    table.add_column("QScore", justify="right")
    table.add_column("Depth", justify="right")
    table.add_column("Move", justify="right")
    table.add_column("Eval", justify="right")
    table.add_column("Topic")
    table.add_column("Next")
    for r in records:
        table.add_row(
            str(r.iteration),
            f"{r.curiosity_score.total:.3f}" if r.curiosity_score else "N/A",
            f"{r.philosophical_depth.final_score:.3f}" if r.philosophical_depth else "N/A",
            f"{r.philosophical_move.final_quality_score:.3f}" if r.philosophical_move else "N/A",
            f"{r.evaluation.score:.3f}" if r.evaluation else "N/A",
            r.input_topic[:70],
            (r.next_topic or "STOP")[:70],
        )
    console.print(table)
    report_dir = f"{config.reports.output_dir}/{controller.run_id}"
    console.print(f"Reports written to: {report_dir}")
    console.print(f"Visible debate transcript: {report_dir}/discussion_transcript.md")
    console.print(f"Curiosity theory state: {config.curiosity_state.path}")


if __name__ == "__main__":
    main()
