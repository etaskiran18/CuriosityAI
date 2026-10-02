from __future__ import annotations

import argparse
from dotenv import load_dotenv
from curiosity_ai.config import load_config
from curiosity_ai.publishing import PublishingManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Approve and publish a queued curiosity report.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--approve", required=True, help="publication_id to approve")
    args = parser.parse_args()
    load_dotenv()
    config = load_config(args.config)
    manager = PublishingManager(config)
    dst = manager.approve_and_publish(args.approve)
    print(f"Published after approval: {dst}")


if __name__ == "__main__":
    main()
