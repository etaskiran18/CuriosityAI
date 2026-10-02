from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig
from .schema import Source, Passage, IterationRecord
from .utils import ensure_dir, stable_id, strip_front_matter, line_range_for_char_span, write_jsonl, read_jsonl


class CorpusChunker:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[tuple[int, int, str]]:
        if len(text) <= self.chunk_size:
            return [(0, len(text), text)]
        chunks: list[tuple[int, int, str]] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # Avoid cutting in the middle of a sentence if possible.
            sentence_end = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if sentence_end > start + int(self.chunk_size * 0.55):
                end = sentence_end + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append((start, end, chunk))
            if end >= len(text):
                break
            start = max(0, end - self.overlap)
        return chunks


class ChromaMemory:
    def __init__(self, config: AppConfig):
        self.config = config
        self.persist_dir = ensure_dir(config.memory.persist_dir)
        self.journal_path = Path(config.memory.journal_path)
        ensure_dir(self.journal_path.parent)
        self._chroma_available = False
        self._client = None
        self._corpus = None
        self._memory = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        if self.config.memory.backend != "chroma":
            return
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            prefix = self.config.memory.collection_prefix
            self._corpus = self._client.get_or_create_collection(f"{prefix}_corpus")
            self._memory = self._client.get_or_create_collection(f"{prefix}_memory")
            self._chroma_available = True
        except Exception as exc:
            print(f"[memory] Chroma unavailable, falling back to journal only: {exc}")
            self._chroma_available = False

    def ingest_corpus(self) -> int:
        corpus_dir = Path(self.config.corpus.path)
        if not corpus_dir.exists():
            print(f"[memory] Corpus directory not found: {corpus_dir}")
            return 0
        accepted = set(self.config.corpus.accepted_extensions)
        chunker = CorpusChunker(
            self.config.corpus.chunk_size_chars,
            self.config.corpus.chunk_overlap_chars,
        )
        count = 0
        for file_path in sorted(corpus_dir.rglob("*")):
            if file_path.name.lower().startswith("readme"):
                continue
            if file_path.suffix.lower() not in accepted:
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            meta, body = strip_front_matter(text)
            title = meta.get("title") or file_path.stem.replace("_", " ").title()
            author = meta.get("author")
            year = meta.get("year")
            license_ = meta.get("license")
            tradition = meta.get("tradition")
            source_url = meta.get("source_url") or meta.get("url")
            source = Source(
                source_id=stable_id(str(file_path), title, author or ""),
                source_type="local",
                title=title,
                author=author,
                year=year,
                file_path=str(file_path),
                url=source_url,
                license=license_,
                tradition=tradition,
            )
            for idx, (start, end, chunk) in enumerate(chunker.chunk(body)):
                line_start, line_end = line_range_for_char_span(body, start, end)
                passage_id = stable_id(str(file_path), str(idx), chunk[:80])
                citation = self.local_citation(source, idx, line_start, line_end)
                passage = Passage(
                    passage_id=passage_id,
                    text=chunk,
                    source=source,
                    citation=citation,
                    line_start=line_start,
                    line_end=line_end,
                    chunk_index=idx,
                )
                self.add_passage(passage)
                count += 1
        return count

    def local_citation(self, source: Source, idx: int, line_start: int | None, line_end: int | None) -> str:
        author = (source.author or "Unknown").replace(" ", "_")
        title = source.title.replace(" ", "_")[:32]
        lines = f"L{line_start}-L{line_end}" if line_start and line_end else f"chunk{idx}"
        return f"[LOCAL:{author}:{title}:chunk{idx}:{lines}]"

    def add_passage(self, passage: Passage) -> None:
        if not self._chroma_available or self._corpus is None:
            return
        meta = {
            "source_id": passage.source.source_id,
            "source_type": passage.source.source_type,
            "title": passage.source.title,
            "author": passage.source.author or "",
            "year": passage.source.year or "",
            "file_path": passage.source.file_path or "",
            "url": passage.source.url or "",
            "license": passage.source.license or "",
            "tradition": passage.source.tradition or "",
            "citation": passage.citation,
            "line_start": passage.line_start or -1,
            "line_end": passage.line_end or -1,
            "chunk_index": passage.chunk_index or 0,
        }
        try:
            self._corpus.add(
                ids=[passage.passage_id],
                documents=[passage.text],
                metadatas=[meta],
            )
        except Exception:
            # Likely duplicate ID; ignore.
            pass

    def query_passages(self, query: str, top_k: int | None = None) -> list[Passage]:
        if not self._chroma_available or self._corpus is None:
            return []
        top_k = top_k or self.config.memory.top_k_passages
        result = self._corpus.query(query_texts=[query], n_results=top_k)
        return self._passages_from_query_result(result)

    def query_memories(self, query: str, top_k: int | None = None) -> list[Passage]:
        if not self._chroma_available or self._memory is None:
            return []
        top_k = top_k or self.config.memory.top_k_memories
        result = self._memory.query(query_texts=[query], n_results=top_k)
        return self._passages_from_query_result(result, default_type="memory")

    def _passages_from_query_result(self, result: dict[str, Any], default_type: str = "local") -> list[Passage]:
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0] if result.get("distances") else [None] * len(docs)
        passages: list[Passage] = []
        for doc, meta, pid, dist in zip(docs, metas, ids, distances):
            source = Source(
                source_id=meta.get("source_id", stable_id(pid)),
                source_type=meta.get("source_type", default_type),
                title=meta.get("title", "Memory"),
                author=meta.get("author") or None,
                year=meta.get("year") or None,
                file_path=meta.get("file_path") or None,
                url=meta.get("url") or None,
                license=meta.get("license") or None,
                tradition=meta.get("tradition") or None,
            )
            score = None if dist is None else float(1.0 / (1.0 + dist))
            citation = meta.get("citation") or f"[MEMORY:{pid}]"
            passages.append(
                Passage(
                    passage_id=pid,
                    text=doc,
                    source=source,
                    citation=citation,
                    line_start=meta.get("line_start") if meta.get("line_start", -1) != -1 else None,
                    line_end=meta.get("line_end") if meta.get("line_end", -1) != -1 else None,
                    chunk_index=meta.get("chunk_index"),
                    score=score,
                )
            )
        return passages

    def store_iteration(self, record: IterationRecord) -> None:
        data = record.model_dump(mode="json")
        write_jsonl(self.journal_path, data)
        if self._chroma_available and self._memory is not None:
            text = self._iteration_to_memory_text(record)
            mid = stable_id(record.run_id, str(record.iteration), text[:100])
            score = record.curiosity_score.total if record.curiosity_score else 0.0
            citation = f"[MEMORY:{record.run_id}:iter{record.iteration}]"
            meta = {
                "source_id": record.run_id,
                "source_type": "memory",
                "title": f"Run {record.run_id} Iteration {record.iteration}",
                "author": "Curiosity AI",
                "year": "",
                "file_path": str(self.journal_path),
                "url": "",
                "license": "private/internal",
                "tradition": "system memory",
                "citation": citation,
                "line_start": -1,
                "line_end": -1,
                "chunk_index": record.iteration,
                "curiosity_score": score,
            }
            try:
                self._memory.add(ids=[mid], documents=[text], metadatas=[meta])
            except Exception:
                pass

    def _iteration_to_memory_text(self, record: IterationRecord) -> str:
        parts = [
            f"Input topic: {record.input_topic}",
            f"Next topic: {record.next_topic or ''}",
        ]
        if record.critique:
            parts.append(f"Strongest question: {record.critique.strongest_question}")
            parts.append(f"Critique: {'; '.join(record.critique.main_objections)}")
        if record.curiosity_score:
            parts.append(f"Curiosity score: {record.curiosity_score.total:.3f}. {record.curiosity_score.explanation}")
        if record.synthesis:
            parts.append(f"Synthesis title: {record.synthesis.title}")
            parts.append(f"Thesis: {record.synthesis.thesis}")
            parts.append("Next directions: " + "; ".join(record.synthesis.next_directions))
        return "\n".join(parts)

    def recent_records(self, limit: int = 10) -> list[dict[str, Any]]:
        return read_jsonl(self.journal_path, limit=limit)
