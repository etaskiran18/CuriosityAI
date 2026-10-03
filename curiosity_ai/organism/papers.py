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
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..utils import strip_front_matter
from .textutil import one_line

EXTENSIONS = (".pdf", ".txt", ".md", ".markdown")


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    already: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (file, why)
    skipped: list[str] = field(default_factory=list)  # left out on purpose (e.g. your own article, shared separately)


def ingest_papers(source: Path, dest: Path, *, max_pages: int = 300, skip: set[Path] | None = None) -> IngestReport:
    """Convert every paper in ``source`` (and its subfolders) that is not in ``dest`` yet."""
    source, dest = Path(source).expanduser(), Path(dest)
    skip = {Path(p).resolve() for p in skip or set()}
    report = IngestReport()
    if not source.is_dir():
        report.failed.append((str(source), "no such folder"))
        return report
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS or path.name.startswith("."):
            continue
        if path.resolve() in skip:
            report.skipped.append(path.name)
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


def read_paper(path: Path, *, max_pages: int = 300) -> tuple[dict[str, str], str]:
    """(front matter or PDF metadata, text) of one paper: a PDF, or a text or Markdown file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return pdf_text(path, max_pages=max_pages)
    return strip_front_matter(path.read_text(encoding="utf-8", errors="replace"))


def opening(text: str, chars: int = 400) -> str:
    """The start of a paper's abstract, or of its first real paragraph: how its field speaks."""
    head = text[:8000]
    match = re.search(r"\babstract\b[\s:.\u2013\u2014-]*", head, re.IGNORECASE)
    if match:
        body = text[match.end():match.end() + chars * 3]
    else:
        lines = head.splitlines()
        first = next(
            (i for i, line in enumerate(lines)
             if len(line.split()) >= 10 and not _looks_like_header(line.strip()) and not _looks_like_affiliation(line)),
            0,
        )
        body = "\n".join(lines[first:])[: chars * 3]
    return one_line(re.sub(r"\[page \d+\]", " ", body), chars)


def pdf_text(path: Path, *, max_pages: int = 300) -> tuple[dict[str, str], str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("reading PDFs needs pypdf: pip install pypdf") from exc
    # Real PDFs are full of small format oddities; pypdf reads past them, so its warnings are only noise.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
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
    if _junk_author(author):
        author = ""  # "iitm", "user", "Microsoft Office User": who made the file, not who wrote the paper
    body = "\n\n".join(pages)
    if not title or _looks_like_header(title) or not _mostly_latin(title) or re.search(r"microsoft word|untitled|\.docx?$|\.tex$|^paper$", title, re.IGNORECASE):
        title = _first_title_line(body) or path.stem.replace("_", " ")
    return {"title": title, "author": author, "pages": str(len(reader.pages))}, body


def clean_pdf_text(text: str) -> str:
    """Rejoin words broken at line ends ("infor-\\nmation") and tidy spaces; keep the lines."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace(" ", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    text = _rejoin_letter_spaced(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _rejoin_letter_spaced(text: str) -> str:
    """'interactio n' and 'i n' (letter-spaced PDF text) become 'interaction' and 'in'.

    Only when the joined word occurs elsewhere in the same text, so "the frequency of f" stays as it is
    unless the text also has "off".
    """
    vocabulary = Counter(re.findall(r"[a-z]+", text.lower()))

    def known(word: str) -> bool:
        w = word.lower()
        return bool(vocabulary.get(w) or vocabulary.get(w + "s") or (w.endswith("s") and vocabulary.get(w[:-1])))

    lines = []
    for line in text.split("\n"):
        words: list[str] = []
        for word in line.split(" "):
            head = words[-1] if words else ""
            split = head.isalpha() and word.isalpha() and word.islower() and (
                len(word) == 1
                # "gener al": a two-letter tail only when the head is no word of its own ("a re" stays)
                or (len(word) == 2 and vocabulary.get(head.lower(), 0) <= 1 and vocabulary.get(word, 0) <= 1)
            )
            if split and known(head + word):
                words[-1] += word
            else:
                words.append(word)
        lines.append(" ".join(words))
    return "\n".join(lines)


_HEADER_START = (
    "abstract", "arxiv", "doi", "http", "www", "received", "accepted", "published", "citation", "keywords",
    "key points", "volume", "vol.", "journal", "research article", "original article", "original paper",
    "article", "letter", "copyright", "open access", "cite this", "correspondence", "page ",
    "manuscript", "submitted to", "preprint", "draft", "confidential", "running title", "running head",
)


def _looks_like_header(line: str) -> bool:
    """A journal's running header, not a title: 'SCIENCE ADVANCES | RESEARCH ARTICLE',
    'Annales Geophysicae (2001) 19: 147-157 (c) European Geophysical Society 2001'."""
    low = line.lower()
    letters = [c for c in line if c.isalpha()]
    return (
        low.startswith(_HEADER_START)
        or any(mark in line for mark in ("|", "\u00a9", "c\u00a9", "(c)"))
        or bool(re.search(r"\(\d{4}\)\s*\d+\s*[:,]", line))  # "(2001) 19: 147"
        or bool(re.search(r"\bdoi\b|\bissn\b|\bvol\.|\bpp\.", low))
        or (len(letters) > 8 and len(line.split()) <= 5 and sum(c.isupper() for c in letters) / len(letters) > 0.8)  # a SHOUTED banner
    )


_AFFILIATION = re.compile(
    r"@|\b(?:universit\w*|institut\w*|laborator\w*|department|dept\.|faculty|college|school of|cent(?:er|re) for"
    r"|observatory|academy|e-?mail|corresponding author)\b|\b\d{5}\b",
    re.IGNORECASE,
)


def _looks_like_affiliation(line: str) -> bool:
    """An author's address, not the start of the paper: 'Space Science Laboratory, University of California,
    Berkeley, CA 94720, USA'."""
    return bool(_AFFILIATION.search(line))


def _mostly_latin(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and sum(unicodedata.name(c, "").startswith("LATIN") for c in letters) >= 0.7 * len(letters)


def _first_title_line(body: str) -> str:
    lines = [line.strip() for line in body.splitlines()[:26]]
    for i, line in enumerate(lines[:25]):
        if line.startswith("[page") or not line or _looks_like_header(line) or not _mostly_latin(line):
            continue  # a running header, or a line in another script (a journal's banner)
        if 3 <= len(line.split()) <= 25:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if _continues_title(line, nxt):
                return f"{line} {nxt}"  # a title wrapped onto a second line
            return line
    return ""


def _continues_title(line: str, nxt: str) -> bool:
    """Is ``nxt`` the rest of a title that wrapped? ("... distribution in the" / "equatorial region of magnetosphere")

    Not when the first line ends a sentence, nor when the next line looks like authors (initials, commas,
    affiliation numbers) or a header.
    """
    if not nxt or line.endswith((".", ":", "?", "!")) or len(nxt.split()) > 12:
        return False
    if line.isupper() and nxt.isupper():  # a title set in capitals continues in capitals
        return not re.search(r"[\d,;@|]", nxt)
    if _looks_like_header(nxt):
        return False
    if nxt[0].islower():
        return True
    return not re.search(r"[\d,;@]|\b[A-Z]\.", nxt)


_JUNK_AUTHORS = {"user", "admin", "administrator", "owner", "author", "authors", "microsoft office user", "default",
                 "guest", "pc", "home", "editor", "test", "windows user"}


def _junk_author(author: str) -> bool:
    a = author.strip().lower()
    return a in _JUNK_AUTHORS or (" " not in a and "," not in a and author == author.lower())


def _clean_meta(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return "" if text.lower() in ("none", "unknown", "anonymous") else text


def _one_line(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).replace("---", "-")).strip()
