from __future__ import annotations

import os
from pathlib import Path
import shutil
import requests

from .config import AppConfig
from .schema import IterationRecord, PublicationItem
from .agents import PublisherAgent
from .utils import ensure_dir, stable_id


class PublishingManager:
    def __init__(self, config: AppConfig, publisher_agent: PublisherAgent | None = None):
        self.config = config
        self.publisher_agent = publisher_agent
        self.queue_dir = ensure_dir(config.publishing.queue_dir)
        self.published_dir = ensure_dir(config.publishing.published_dir)

    def maybe_queue(self, record: IterationRecord) -> Path | None:
        if not self.config.publishing.enabled or not record.synthesis or not record.evaluation:
            return None
        score = record.evaluation.score or (record.curiosity_score.total if record.curiosity_score else 0.0)
        if score < self.config.publishing.min_score_to_queue:
            return None
        if not record.evaluation.should_publish:
            return None
        if not self.publisher_agent:
            body = record.synthesis.report_markdown or record.synthesis.thesis
            title = record.synthesis.title
        else:
            prepared = self.publisher_agent.prepare_public_post(record.synthesis, record.evaluation)
            title = prepared.get("title") or record.synthesis.title
            body = prepared.get("body_markdown") or record.synthesis.report_markdown
        pub_id = stable_id(record.run_id, str(record.iteration), title)
        item = PublicationItem(
            publication_id=pub_id,
            run_id=record.run_id,
            iteration=record.iteration,
            title=title,
            body_markdown=body,
            score=score,
            approved=False,
            platform=self.config.publishing.platform,
        )
        path = self.queue_dir / f"{pub_id}.md"
        path.write_text(self._format_pending(item), encoding="utf-8")
        return path

    def _format_pending(self, item: PublicationItem) -> str:
        return "\n".join([
            "---",
            f"publication_id: {item.publication_id}",
            f"run_id: {item.run_id}",
            f"iteration: {item.iteration}",
            f"score: {item.score}",
            f"approved: {str(item.approved).lower()}",
            f"platform: {item.platform}",
            "---",
            "",
            f"# {item.title}",
            "",
            item.body_markdown,
            "",
            "<!-- Human approval required. To publish locally, run:",
            f"python scripts/publish_pending.py --approve {item.publication_id}",
            "-->",
            "",
        ])

    def approve_and_publish(self, publication_id: str) -> Path:
        matches = list(self.queue_dir.glob(f"{publication_id}.md"))
        if not matches:
            raise FileNotFoundError(f"No pending publication found for id {publication_id}")
        src = matches[0]
        platform = self.config.publishing.platform
        if platform == "local_markdown":
            ensure_dir(self.config.publishing.public_site_dir)
            dst = Path(self.config.publishing.public_site_dir) / src.name
            text = src.read_text(encoding="utf-8").replace("approved: false", "approved: true")
            dst.write_text(text, encoding="utf-8")
            archived = self.published_dir / src.name
            shutil.move(str(src), archived)
            return dst
        if platform == "webhook":
            webhook = os.environ.get(self.config.publishing.webhook_url_env)
            if not webhook:
                raise RuntimeError(f"Webhook publishing requested but {self.config.publishing.webhook_url_env} is unset.")
            text = src.read_text(encoding="utf-8")
            resp = requests.post(webhook, json={"publication_id": publication_id, "markdown": text}, timeout=30)
            resp.raise_for_status()
            archived = self.published_dir / src.name
            shutil.move(str(src), archived)
            return archived
        raise ValueError(f"Unsupported publishing platform: {platform}")
