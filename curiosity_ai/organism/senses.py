"""How the organism observes: its library, the web, and what humans give it.

The default library search is BM25 over the local corpus. It needs no
embedding model, gives the same results every time, and its rankings can be
explained. Chroma retrieval from v7 can be switched on in config.
"""
from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..memory import CorpusChunker
from ..utils import line_range_for_char_span, stable_id, strip_front_matter
from .textutil import clip, coverage, tokens

if TYPE_CHECKING:
    from ..config import AppConfig

_GUTENBERG_START = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE)
_GUTENBERG_END = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE)


@dataclass
class Observation:
    label: str
    citation: str
    title: str
    author: str
    text: str
    kind: Literal["corpus", "inbox", "web"]
    score: float = 0.0

    @property
    def heading(self) -> str:
        return f"{self.author}, {self.title}" if self.author else self.title


@dataclass
class _Chunk:
    citation: str
    title: str
    author: str
    text: str
    kind: Literal["corpus", "inbox"]
    source_key: str
    length: int = 0
    tf: Counter = field(default_factory=Counter)


def library_body(text: str) -> tuple[dict[str, str], str, int]:
    """Split front matter and Project Gutenberg boilerplate from the readable text.

    Returns (metadata, body, line_offset). Adding line_offset to a line number
    within ``body`` gives the line number in the file itself, so a citation
    can be checked by opening the file.
    """
    meta, body = strip_front_matter(text)
    offset = text[: len(text) - len(body)].count("\n")
    start = _GUTENBERG_START.search(body)
    if start and body[: start.start()].count("\n") < 150:
        offset += body[: start.end()].count("\n")
        body_rest = body[start.end():]
    else:
        body_rest = body
    end = _GUTENBERG_END.search(body_rest)
    if end and end.start() > len(body_rest) * 0.5:
        body_rest = body_rest[: end.start()]
    return meta, body_rest, offset


class LexicalLibrary:
    """BM25 retrieval over chunked corpus files (plus the human inbox)."""

    def __init__(
        self,
        corpus_dirs: list[Path],
        *,
        chunk_chars: int = 1200,
        overlap_chars: int = 200,
        extensions: tuple[str, ...] = (".txt", ".md", ".markdown"),
        exclude: tuple[str, ...] = (),
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.corpus_dirs = [Path(d) for d in corpus_dirs]
        self.chunker = CorpusChunker(chunk_chars, overlap_chars)
        self.extensions = tuple(e.lower() for e in extensions)
        self.exclude = {name.lower() for name in exclude}
        self.k1 = k1
        self.b = b
        self._chunks: list[_Chunk] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._indexed_files: set[str] = set()
        self._built = False

    def __len__(self) -> int:
        self._ensure_built()
        return len(self._chunks)

    def _ensure_built(self) -> None:
        if not self._built:
            self.refresh()
            self._built = True

    def refresh(self) -> int:
        """Index any files that appeared since the last call. Returns new chunk count."""
        added = 0
        for directory in self.corpus_dirs:
            if not directory.exists():
                continue
            kind: Literal["corpus", "inbox"] = "inbox" if directory.name == "inbox" else "corpus"
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in self.extensions:
                    continue
                if path.name.lower().startswith("readme") or path.name.lower() in self.exclude:
                    continue
                if str(path) in self._indexed_files:
                    continue
                added += self._index_file(path, kind)
                self._indexed_files.add(str(path))
        self._built = True
        return added

    def _index_file(self, path: Path, kind: Literal["corpus", "inbox"]) -> int:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        meta, body, line_offset = library_body(raw)
        title = meta.get("title") or path.stem.replace("_", " ").title()
        author = meta.get("author") or ("Human observer" if kind == "inbox" else "")
        added = 0
        for idx, (start, end, text) in enumerate(self.chunker.chunk(body)):
            if start > 0 and body[start - 1].isalnum():
                # Overlapping chunks can begin mid-word; start at the next whole word.
                text = text.split(None, 1)[1] if len(text.split(None, 1)) > 1 else text
            terms = tokens(text)
            if len(terms) < 8:
                continue
            line_start, line_end = line_range_for_char_span(body, start, end)
            chunk = _Chunk(
                citation=_citation(kind, author, title, idx, line_start + line_offset, line_end + line_offset),
                title=title,
                author=author,
                text=text,
                kind=kind,
                source_key=str(path),
                length=len(terms),
                tf=Counter(terms),
            )
            chunk_id = len(self._chunks)
            self._chunks.append(chunk)
            for term, count in chunk.tf.items():
                self._postings.setdefault(term, []).append((chunk_id, count))
            added += 1
        return added

    def search(self, query: str, k: int = 5, max_per_source: int = 2) -> list[Observation]:
        self._ensure_built()
        if not self._chunks:
            return []
        n = len(self._chunks)
        avg_len = sum(c.length for c in self._chunks) / n
        scores: dict[int, float] = {}
        for term in set(tokens(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for chunk_id, tf in postings:
                length = self._chunks[chunk_id].length
                denom = tf + self.k1 * (1 - self.b + self.b * length / avg_len)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * tf * (self.k1 + 1) / denom
        ranked = heapq.nlargest(k * 8, scores.items(), key=lambda item: item[1])
        chosen: list[Observation] = []
        per_source: Counter = Counter()
        for chunk_id, score in ranked:
            chunk = self._chunks[chunk_id]
            if per_source[chunk.source_key] >= max_per_source:
                continue
            per_source[chunk.source_key] += 1
            chosen.append(
                Observation(
                    label="",
                    citation=chunk.citation,
                    title=chunk.title,
                    author=chunk.author,
                    text=chunk.text,
                    kind=chunk.kind,
                    score=score,
                )
            )
            if len(chosen) >= k:
                break
        return chosen


def _citation(kind: str, author: str, title: str, idx: int, line_start: int, line_end: int) -> str:
    prefix = "INBOX" if kind == "inbox" else "LOCAL"
    author_part = (author or "Unknown").replace(" ", "_")
    title_part = title.replace(" ", "_")[:32]
    return f"[{prefix}:{author_part}:{title_part}:chunk{idx}:L{line_start}-L{line_end}]"


class ChromaLibrary:
    """Adapter that lets the organism read through v7's Chroma vector memory.

    The corpus is searched by embedding similarity; what humans shared (the
    inbox) is searched with BM25 alongside it, since v7's memory does not ingest it.
    """

    def __init__(self, config: "AppConfig", inbox_dir: Path):
        from ..memory import ChromaMemory

        self.memory = ChromaMemory(config)
        self.memory.ingest_corpus()
        self.exclude = {name.lower() for name in config.organism.library_exclude}
        self.inbox = LexicalLibrary([inbox_dir], chunk_chars=config.organism.chunk_chars, overlap_chars=config.organism.chunk_overlap_chars)

    def refresh(self) -> int:
        return self.inbox.refresh()

    def search(self, query: str, k: int = 5, max_per_source: int = 2) -> list[Observation]:
        out = self.inbox.search(query, k=min(2, k), max_per_source=max_per_source)
        per_source: Counter = Counter()
        for p in self.memory.query_passages(query, top_k=k * 3):
            if len(out) >= k:
                break
            if Path(p.source.file_path or "").name.lower() in self.exclude:
                continue
            if per_source[p.source.source_id] >= max_per_source:
                continue
            per_source[p.source.source_id] += 1
            out.append(
                Observation(
                    label="",
                    citation=p.citation,
                    title=p.source.title,
                    author=p.source.author or "",
                    text=p.text,
                    kind="corpus",
                    score=p.score or 0.0,
                )
            )
        return out


class Senses:
    """Everything the organism can observe in one act of looking."""

    def __init__(
        self,
        library,
        *,
        web=None,
        passages: int = 5,
        max_per_source: int = 2,
        passage_chars: int = 1100,
        web_results: int = 2,
        web_min_relevance: float = 0.25,
    ):
        self.library = library
        self.web = web
        self.passages = passages
        self.max_per_source = max_per_source
        self.passage_chars = passage_chars
        self.web_results = web_results
        self.web_min_relevance = web_min_relevance
        self.rejected_web: list[str] = []

    @classmethod
    def from_config(cls, config: "AppConfig", inbox_dir: Path) -> "Senses":
        oc = config.organism
        if oc.retrieval == "chroma":
            library = ChromaLibrary(config, inbox_dir)
        else:
            library = LexicalLibrary(
                [Path(config.corpus.path), inbox_dir],
                chunk_chars=oc.chunk_chars,
                overlap_chars=oc.chunk_overlap_chars,
                extensions=tuple(config.corpus.accepted_extensions),
                exclude=tuple(oc.library_exclude),
            )
        web = None
        if config.web_grounding.enabled:
            from ..grounding import WebGroundingAgent

            web = WebGroundingAgent(config)
        return cls(
            library,
            web=web,
            passages=oc.evidence_passages,
            max_per_source=oc.max_passages_per_source,
            passage_chars=oc.passage_chars,
            web_results=oc.max_web_results,
            web_min_relevance=oc.web_min_relevance,
        )

    def notice_new_material(self) -> int:
        return self.library.refresh()

    def observe(self, question: str, expectations: list[str] | None = None) -> list[Observation]:
        query = " ".join([question, *(expectations or [])])
        found = self.library.search(query, k=self.passages, max_per_source=self.max_per_source)
        if self.web is not None:
            found.extend(self._observe_web(question))
        for i, obs in enumerate(found, start=1):
            obs.label = f"S{i}"
            obs.text = clip(obs.text, self.passage_chars)
        return found

    def _observe_web(self, question: str) -> list[Observation]:
        """Web results must be about the question; off-topic hits are dropped."""
        self.rejected_web = []
        try:
            results = self.web.search(question)
        except Exception as exc:  # network trouble must not kill a heartbeat
            self.rejected_web.append(f"web search failed: {exc}")
            return []
        kept: list[Observation] = []
        for r in results:
            if (r.raw or {}).get("provider") == "error":
                continue
            relevance = coverage(question, f"{r.title} {r.content}")
            if relevance < self.web_min_relevance:
                self.rejected_web.append(f"{r.title[:80]} (relevance {relevance:.2f})")
                continue
            kept.append(
                Observation(
                    label="",
                    citation=r.citation,
                    title=r.title,
                    author=(r.raw or {}).get("provider", "web"),
                    text=r.content,
                    kind="web",
                    score=relevance,
                )
            )
            if len(kept) >= self.web_results:
                break
        return kept


def write_inbox_item(inbox_dir: Path, text: str, title: str, when: str) -> Path:
    """Store something a human shared, in the same front-matter format as the corpus."""
    inbox_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40] or "observation"
    path = inbox_dir / f"{when[:19].replace(':', '').replace('-', '')}_{slug}_{stable_id(text, length=6)}.md"
    safe_title = title.replace("\n", " ").replace("---", "-")
    path.write_text(
        f"---\ntitle: {safe_title}\nauthor: Human observer\nyear: {when[:4]}\n---\n\n{text.strip()}\n",
        encoding="utf-8",
    )
    return path
