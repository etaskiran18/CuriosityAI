"""Researcher mode: bring a person's own papers into the organism's library.

PDFs become text with pypdf (pure Python, no other programs needed); .txt and
.md files are copied as they are. Each paper becomes a Markdown file in
``<home>/papers/`` with front matter (title, author, the original file, its
pages). Page markers such as ``[page 3]`` stay in the text, so a quote the
organism cites can be found again in the PDF.

Scanned PDFs (pictures of pages) contain no text; they are reported, not
guessed at. Run them through OCR first if you need them.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..utils import strip_front_matter

EXTENSIONS = (".pdf", ".txt", ".md", ".markdown")


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    already: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (file, why)


def ingest_papers(source: Path, dest: Path, *, max_pages: int = 300) -> IngestReport:
    """Convert every paper in ``source`` (and its subfolders) that is not in ``dest`` yet."""
    source, dest = Path(source).expanduser(), Path(dest)
    report = IngestReport()
    if not source.is_dir():
        report.failed.append((str(source), "no such folder"))
        return report
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS or path.name.startswith("."):
            continue
        data = path.read_bytes()
        digest = hashlib.sha1(data).hexdigest()[:10]
        slug = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")[:60] or "paper"
        target = dest / f"{slug}_{digest}.md"
        if target.exists():
            report.already.append(path.name)
            continue
        try:
            if path.suffix.lower() == ".pdf":
                meta, text = pdf_text(path, max_pages=max_pages)
            else:
                meta, text = strip_front_matter(data.decode("utf-8", errors="replace"))
        except Exception as exc:  # a broken file must not stop the others
            report.failed.append((path.name, f"{type(exc).__name__}: {exc}"))
            continue
        if len(text.strip()) < 200:
            report.failed.append((path.name, "no text found (a scanned PDF needs OCR first)"))
            continue
        title = meta.get("title") or path.stem.replace("_", " ")
        front = {
            "title": title,
            "author": meta.get("author", ""),
            "source_file": str(path),
            "pages": meta.get("pages", ""),
        }
        lines = ["---", *(f"{k}: {_one_line(v)}" for k, v in front.items() if v), "---", "", text.strip(), ""]
        target.write_text("\n".join(lines), encoding="utf-8")
        report.added.append(path.name)
    return report


def pdf_text(path: Path, *, max_pages: int = 300) -> tuple[dict[str, str], str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("reading PDFs needs pypdf: pip install pypdf") from exc
    reader = PdfReader(str(path))
    pages: list[str] = []
    for number, page in enumerate(reader.pages[:max_pages], start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = clean_pdf_text(text)
        if text.strip():
            pages.append(f"[page {number}]\n{text}")
    info = reader.metadata
    title = _clean_meta(getattr(info, "title", None) if info else None)
    author = _clean_meta(getattr(info, "author", None) if info else None)
    body = "\n\n".join(pages)
    if not title or re.search(r"microsoft word|untitled|\.docx?$|\.tex$|^paper$", title, re.IGNORECASE):
        title = _first_title_line(body) or path.stem.replace("_", " ")
    return {"title": title, "author": author, "pages": str(len(reader.pages))}, body


def clean_pdf_text(text: str) -> str:
    """Rejoin words broken at line ends ("infor-\\nmation") and tidy spaces; keep the lines."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace(" ", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _first_title_line(body: str) -> str:
    for line in body.splitlines()[:15]:
        line = line.strip()
        if line.startswith("[page") or not line:
            continue
        if 3 <= len(line.split()) <= 25 and not line.lower().startswith(("abstract", "arxiv", "doi", "http")):
            return line
    return ""


def _clean_meta(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return "" if text.lower() in ("none", "unknown", "anonymous") else text


def _one_line(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).replace("---", "-")).strip()
