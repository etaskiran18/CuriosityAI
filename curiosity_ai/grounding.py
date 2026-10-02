from __future__ import annotations

import os
from typing import Any
import requests

from .config import AppConfig
from .memory import ChromaMemory
from .schema import Passage, WebResult, SourceGateReport
from .utils import stable_id


class LocalGroundingAgent:
    def __init__(self, config: AppConfig, memory: ChromaMemory):
        self.config = config
        self.memory = memory

    def retrieve(self, topic: str) -> list[Passage]:
        passages = self.memory.query_passages(topic, top_k=self.config.memory.top_k_passages)
        memories = self.memory.query_memories(topic, top_k=self.config.memory.top_k_memories)
        return passages + memories


class WebGroundingAgent:
    """Optional web / academic grounding with strict source labels.

    v7 supports:
    - tavily: live web search for broad source grounding (requires TAVILY_API_KEY)
    - crossref: scholarly metadata search, no key required
    - semantic_scholar: scholarly metadata search, optional SEMANTIC_SCHOLAR_API_KEY
    - hybrid: Tavily if a key exists + Crossref/Semantic Scholar metadata

    Web/academic results are labeled [WEB:*] and downstream agents must cite those
    labels exactly. Web results are not allowed to silently replace primary texts;
    they are used mainly to find secondary literature, context, and follow-up sources.
    """

    def __init__(self, config: AppConfig):
        self.config = config

    def search(self, query: str) -> list[WebResult]:
        if not self.config.web_grounding.enabled:
            return []
        provider = self.config.web_grounding.provider.lower().strip()
        results: list[WebResult] = []
        if provider == "tavily":
            results.extend(self._search_tavily(query))
        elif provider == "crossref":
            results.extend(self._search_crossref(query))
        elif provider in {"semantic_scholar", "semanticscholar"}:
            results.extend(self._search_semantic_scholar(query))
        elif provider == "hybrid":
            # Tavily is optional in hybrid mode; academic metadata still works without the key.
            try:
                if os.environ.get(self.config.web_grounding.tavily_api_key_env):
                    results.extend(self._search_tavily(query))
            except Exception as exc:
                results.append(self._error_result("WEB:TAVILY_ERROR", "Tavily search failed", str(exc)))
            if self.config.web_grounding.academic_search_enabled:
                results.extend(self._search_crossref(query))
                results.extend(self._search_semantic_scholar(query))
        else:
            raise ValueError(f"Unsupported web grounding provider: {provider}")
        return results[: max(1, self.config.web_grounding.max_results + self.config.web_grounding.academic_max_results * 2)]

    def _search_tavily(self, query: str) -> list[WebResult]:
        api_key = os.environ.get(self.config.web_grounding.tavily_api_key_env)
        if not api_key:
            if self.config.web_grounding.provider.lower() == "hybrid":
                return []
            raise RuntimeError(
                f"web_grounding.enabled=true but {self.config.web_grounding.tavily_api_key_env} is not set."
            )
        body: dict[str, Any] = {
            "query": query,
            "max_results": self.config.web_grounding.max_results,
            "search_depth": self.config.web_grounding.search_depth,
            "include_answer": self.config.web_grounding.include_answer,
            "include_raw_content": self.config.web_grounding.include_raw_content,
        }
        if self.config.web_grounding.allowed_domains:
            body["include_domains"] = self.config.web_grounding.allowed_domains
        if self.config.web_grounding.blocked_domains:
            body["exclude_domains"] = self.config.web_grounding.blocked_domains

        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=self.config.web_grounding.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        results: list[WebResult] = []
        for idx, item in enumerate(data.get("results", []), start=1):
            title = item.get("title") or "Untitled web result"
            url = item.get("url") or ""
            content = item.get("content") or item.get("raw_content") or ""
            wid = stable_id("web", url, title, str(idx))
            results.append(
                WebResult(
                    web_id=wid,
                    title=title,
                    url=url,
                    content=content[:4000],
                    citation=f"[WEB:T{idx}]",
                    score=item.get("score"),
                    raw={"provider": "tavily", **item},
                )
            )
        return results

    def _search_crossref(self, query: str) -> list[WebResult]:
        try:
            resp = requests.get(
                "https://api.crossref.org/works",
                params={
                    "query.bibliographic": query,
                    "rows": self.config.web_grounding.academic_max_results,
                    "select": "DOI,title,author,issued,URL,abstract,container-title",
                },
                timeout=self.config.web_grounding.timeout_seconds,
            )
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
        except Exception as exc:
            return [self._error_result("WEB:CR_ERROR", "Crossref search failed", str(exc))]
        results: list[WebResult] = []
        for idx, item in enumerate(items, start=1):
            title = " ".join(item.get("title") or ["Untitled Crossref work"])
            authors = ", ".join(
                " ".join(filter(None, [a.get("given", ""), a.get("family", "")])).strip()
                for a in item.get("author", [])[:4]
            ) or "Unknown author"
            year = ""
            try:
                year = str(item.get("issued", {}).get("date-parts", [[""]])[0][0])
            except Exception:
                year = ""
            doi = item.get("DOI", "")
            url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
            abstract = item.get("abstract", "")
            venue = " ".join(item.get("container-title") or [])
            content = f"Provider: Crossref scholarly metadata\nTitle: {title}\nAuthors: {authors}\nYear: {year}\nVenue: {venue}\nDOI: {doi}\nAbstract/metadata: {abstract}"
            results.append(WebResult(
                web_id=stable_id("crossref", doi, title, str(idx)),
                title=title,
                url=url,
                content=content[:3500],
                citation=f"[WEB:CR{idx}]",
                score=None,
                raw={"provider": "crossref", **item},
            ))
        return results

    def _search_semantic_scholar(self, query: str) -> list[WebResult]:
        headers = {}
        api_key = os.environ.get(self.config.web_grounding.semantic_scholar_api_key_env)
        if api_key:
            headers["x-api-key"] = api_key
        try:
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": self.config.web_grounding.academic_max_results,
                    "fields": "title,authors,year,abstract,url,venue,citationCount,externalIds",
                },
                headers=headers,
                timeout=self.config.web_grounding.timeout_seconds,
            )
            resp.raise_for_status()
            items = resp.json().get("data", [])
        except Exception as exc:
            return [self._error_result("WEB:S2_ERROR", "Semantic Scholar search failed", str(exc))]
        results: list[WebResult] = []
        for idx, item in enumerate(items, start=1):
            title = item.get("title") or "Untitled Semantic Scholar result"
            authors = ", ".join(a.get("name", "") for a in item.get("authors", [])[:4]) or "Unknown author"
            year = item.get("year") or ""
            url = item.get("url") or ""
            venue = item.get("venue") or ""
            abstract = item.get("abstract") or ""
            cites = item.get("citationCount")
            content = f"Provider: Semantic Scholar metadata\nTitle: {title}\nAuthors: {authors}\nYear: {year}\nVenue: {venue}\nCitation count: {cites}\nAbstract: {abstract}"
            results.append(WebResult(
                web_id=stable_id("s2", url, title, str(idx)),
                title=title,
                url=url,
                content=content[:3500],
                citation=f"[WEB:S2-{idx}]",
                score=None,
                raw={"provider": "semantic_scholar", **item},
            ))
        return results

    def _error_result(self, citation: str, title: str, message: str) -> WebResult:
        return WebResult(
            web_id=stable_id("web-error", citation, message),
            title=title,
            url="",
            content=f"Search provider error: {message}",
            citation=f"[{citation}]" if not citation.startswith("[") else citation,
            score=None,
            raw={"provider": "error", "message": message},
        )

def format_passages(passages: list[Passage], max_chars_per_passage: int = 1200) -> str:
    if not passages:
        return "No local passages or memory retrieved."
    blocks = []
    for p in passages:
        title = p.source.title
        author = p.source.author or "Unknown"
        head = f"{p.citation} {author}, {title}"
        if p.source.year:
            head += f" ({p.source.year})"
        body = p.text.strip().replace("\n\n", "\n")[:max_chars_per_passage]
        blocks.append(f"{head}\n{body}")
    return "\n\n---\n\n".join(blocks)


def format_web_results(results: list[WebResult], max_chars_per_result: int = 1000) -> str:
    if not results:
        return "No web results retrieved."
    blocks = []
    for r in results:
        body = r.content.strip().replace("\n\n", "\n")[:max_chars_per_result]
        blocks.append(f"{r.citation} {r.title}\nURL: {r.url}\n{body}")
    return "\n\n---\n\n".join(blocks)


def build_source_gate_report(config: AppConfig, passages: list[Passage], web_results: list[WebResult]) -> SourceGateReport:
    """Deterministic source gate.

    Philosophy mode should not let the model synthesize from its own prior memory only.
    This report is passed to the agents and printed in transcripts.
    """
    primary = [p for p in passages if p.source.source_type == "local"]
    memories = [p for p in passages if p.source.source_type == "memory"]
    usable = [p.citation for p in primary] + [w.citation for w in web_results]
    required = config.depth.min_primary_sources
    if len(primary) + len(web_results) >= required:
        decision = "pass"
        reason = f"Sufficient primary grounding: {len(primary)} local primary passages and {len(web_results)} web sources."
        missing: list[str] = []
    elif len(primary) + len(web_results) > 0:
        decision = "limited"
        reason = (
            f"Limited primary grounding: {len(primary)} local primary passages and {len(web_results)} web sources; "
            f"required {required}. The synthesis must explicitly mark unsupported claims as tentative."
        )
        missing = [f"Need at least {required - (len(primary) + len(web_results))} more primary source passage(s)."]
    else:
        decision = "block" if config.depth.block_synthesis_without_primary_sources else "limited"
        reason = (
            "No primary local/web sources were retrieved. Memory may help continuity, but it cannot support "
            "a serious philosophical answer by itself."
        )
        missing = ["Retrieve real primary philosophical passages before making substantive interpretive claims."]
    return SourceGateReport(
        primary_source_count=len(primary),
        memory_source_count=len(memories),
        web_source_count=len(web_results),
        required_primary_sources=required,
        decision=decision,
        reason=reason,
        missing_sources=missing,
        usable_citations=usable,
    )
