from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import requests

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curiosity_ai.organism.librarian import already_in  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "real_corpus_manifest.json"
DEFAULT_OUT = ROOT / "data" / "philosophy_corpus"


def safe_text(value: Any) -> str:
    return str(value or "").replace("---", "-").strip()


def candidate_urls(ebook_id: str) -> list[str]:
    # Project Gutenberg asks automated tools to use a mirror; the main site is the fallback.
    return [
        f"https://aleph.pglaf.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
        f"https://www.gutenberg.org/ebooks/{ebook_id}.txt.utf-8",
    ]


def fetch_text(ebook_id: str, timeout: int = 60) -> tuple[str, str]:
    errors: list[str] = []
    headers = {
        "User-Agent": "CuriosityAI/0.8 (https://github.com/etaskiran18/CuriosityAI) corpus downloader"
    }
    for url in candidate_urls(ebook_id):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            # Gutenberg's texts are UTF-8, but mirrors do not always say so; without a
            # charset, requests would guess Latin-1 and garble every accented letter.
            text = resp.content.decode("utf-8", errors="replace")
            if resp.status_code == 200 and len(text) > 1000:
                return url, text
            errors.append(f"{url} -> HTTP {resp.status_code}, {len(text)} chars")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {exc}")
    raise RuntimeError("Could not download ebook_id=" + ebook_id + "\n" + "\n".join(errors))


def normalize_pg_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Keep the Gutenberg header/license material because redistribution rules may require it.
    # Just collapse extreme whitespace for cleaner chunking.
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def yaml_front_matter(work: dict[str, Any], downloaded_url: str) -> str:
    themes = work.get("themes") or []
    themes_str = ", ".join(safe_text(x) for x in themes)
    lines = [
        "---",
        f"title: {safe_text(work.get('title'))}",
        f"author: {safe_text(work.get('author'))}",
        f"year: {safe_text(work.get('year'))}",
        f"translator: {safe_text(work.get('translator'))}",
        f"tradition: {safe_text(work.get('tradition'))}",
        f"themes: {themes_str}",
        f"source_url: {safe_text(work.get('source_url'))}",
        f"downloaded_url: {downloaded_url}",
        f"ebook_id: {safe_text(work.get('ebook_id'))}",
        "license: Project Gutenberg text; generally public domain in the United States, but verify copyright status in your jurisdiction before redistribution.",
        "corpus_note: Downloaded primary text for local RAG grounding; do not treat as model pretraining data unless licensing and jurisdiction are checked.",
        "---",
        "",
    ]
    return "\n".join(lines)


def download_manifest(manifest_path: Path, out_dir: Path, clear_existing: bool, sleep_s: float) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    works = manifest.get("works") or []
    out_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for item in out_dir.iterdir():
            if item.name.lower().startswith("readme"):
                continue
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    successes = 0
    failures: list[str] = []
    for work in works:
        slug = work["slug"]
        ebook_id = str(work["ebook_id"])
        out_path = out_dir / f"{slug}.md"
        same_name = [p for p in out_dir.glob(f"{slug}.*") if p.suffix.lower() in {".md", ".txt"}]
        if same_name or already_in(out_dir, ebook_id, work.get("title", "")):
            print(f"[skip] {slug}: already in {out_dir}")
            continue
        print(f"[download] {work.get('author')} — {work.get('title')} (Project Gutenberg #{ebook_id})")
        try:
            url, text = fetch_text(ebook_id)
            body = normalize_pg_text(text)
            out_path.write_text(yaml_front_matter(work, url) + body, encoding="utf-8")
            print(f"[ok] {out_path} ({len(body):,} chars)")
            successes += 1
            if sleep_s:
                time.sleep(sleep_s)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {slug}: {exc}")
            failures.append(f"{slug}: {exc}")
    print(f"\nDownloaded {successes} works into {out_dir}")
    if failures:
        print("\nFailures:")
        for fail in failures:
            print("-", fail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real public-domain philosophical primary texts for Curiosity AI.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to real_corpus_manifest.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory, default data/philosophy_corpus")
    parser.add_argument("--clear-existing", action="store_true", help="Remove existing corpus files before downloading, except README files.")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to wait between downloads.")
    args = parser.parse_args()
    download_manifest(Path(args.manifest), Path(args.out), args.clear_existing, args.sleep)


if __name__ == "__main__":
    main()
