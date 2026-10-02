"""The librarian: how the organism grows its own library.

When the texts it owns stay silent on a question, a curious reader goes and
looks elsewhere. The librarian looks in three places:

* Wikipedia, for concepts and thinkers (CC BY-SA; stored with attribution);
* Project Gutenberg, for whole public-domain books. It searches the
  catalogue file that Gutenberg publishes for programs and downloads the text
  from a mirror, as Gutenberg asks automated tools to do;
* Semantic Scholar, for abstracts of scientific papers (optional, and
  rate-limited without an API key).

Everything it acquires goes into the organism's own library folder, with
front matter saying what it is, where it came from, its license, and why it
was acquired (which question, at which heartbeat). Quotas, a minimum interval
between requests and respect for Retry-After keep it a polite visitor.
"""
from __future__ import annotations

import csv
import gzip
import html
import io
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import quote, urlparse

import requests

from ..utils import read_jsonl, stable_id, write_jsonl
from .textutil import coverage, token_set

if TYPE_CHECKING:
    from ..config import LibrarianConfig

USER_AGENT_PRODUCT = "CuriosityAI/0.8"


@dataclass
class Acquisition:
    kind: str  # "encyclopedia", "book" or "paper"
    title: str
    author: str
    source_url: str
    path: str
    chars: int
    reason: str = ""
    question_id: str | None = None
    heartbeat: int | None = None

    def label(self) -> str:
        if self.kind == "encyclopedia":
            return f"Wikipedia: {self.title}"
        if self.kind == "book":
            return f"book: {self.title} ({self.author})"
        return f"paper: {self.title}"


@dataclass
class ReadingWish:
    topics: list[str] = field(default_factory=list)  # encyclopedia topics
    books: list[str] = field(default_factory=list)  # "author title" queries
    papers: list[str] = field(default_factory=list)  # search phrases


class Http:
    """requests with a polite User-Agent, a minimum interval per host, and Retry-After."""

    def __init__(
        self,
        user_agent: str,
        *,
        timeout: float = 45,
        min_interval: float = 1.0,
        max_wait: float = 60,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        session: requests.Session | None = None,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_wait = max_wait
        self._sleep = sleep
        self._clock = clock
        self.session = session or requests.Session()
        self._last: dict[str, float] = {}
        self._resting_until: dict[str, float] = {}

    def resting(self, url: str) -> bool:
        host = urlparse(url).netloc
        return self._clock() < self._resting_until.get(host, 0.0)

    def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> requests.Response | None:
        host = urlparse(url).netloc
        if self.resting(url):
            return None
        for attempt in range(2):
            wait = self._last.get(host, -1e9) + self.min_interval - self._clock()
            if wait > 0:
                self._sleep(wait)
            try:
                resp = self.session.get(
                    url, params=params, timeout=self.timeout, headers={"User-Agent": self.user_agent, **(headers or {})}
                )
            except requests.RequestException:
                return None
            finally:
                self._last[host] = self._clock()
            if resp.status_code in (429, 503):
                retry = _retry_after(resp)
                if attempt == 0 and retry <= self.max_wait:
                    self._sleep(retry)
                    continue
                self._resting_until[host] = self._clock() + max(retry, 300)
                return None
            return resp if resp.status_code < 400 else None
        return None


def _retry_after(resp: requests.Response) -> float:
    try:
        return max(5.0, float(resp.headers.get("Retry-After", "")))
    except ValueError:
        return 5.0  # Wikimedia: without the header, wait at least five seconds


def user_agent(contact: str) -> str:
    """Wikimedia and Gutenberg ask programs to identify themselves with contact details."""
    return f"{USER_AGENT_PRODUCT} ({contact}) python-requests/{requests.__version__}"


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

_END_SECTIONS = {
    "see also", "references", "notes", "external links", "further reading", "bibliography",
    "sources", "citations", "footnotes", "works cited", "notes and references",
}


def clean_wikipedia_extract(text: str) -> str:
    """Keep the article, drop the reference sections at its end."""
    kept: list[str] = []
    for line in text.splitlines():
        if line.strip().lower() in _END_SECTIONS:
            break
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


class Wikipedia:
    API = "https://en.wikipedia.org/w/api.php"

    def __init__(self, http: Http):
        self.http = http

    def find(self, topic: str) -> tuple[str, str, str] | None:
        """(title, text, url) of the article that best covers ``topic``, or None."""
        candidates: list[str] = []
        if len(topic.split()) <= 5:
            candidates.append(topic.strip())  # a concept or a name: try the exact article first
        resp = self.http.get(self.API, params={
            "action": "query", "list": "search", "srsearch": topic, "srlimit": 5,
            "srnamespace": 0, "format": "json", "utf8": 1,
        })
        if resp is not None:
            hits = (resp.json().get("query") or {}).get("search") or []
            ranked = sorted(
                hits,
                key=lambda h: (coverage(topic, h.get("title", "")), coverage(topic, _strip_tags(h.get("snippet", "")))),
                reverse=True,
            )
            candidates += [h["title"] for h in ranked if h.get("title") and h["title"] not in candidates]
        for title in candidates[:3]:
            page = self._page(title)
            if page is None:
                continue
            real_title, text = page
            if coverage(topic, f"{real_title} {text[:3000]}") >= 0.5:
                return real_title, text, "https://en.wikipedia.org/wiki/" + quote(real_title.replace(" ", "_"))
        return None

    def _page(self, title: str) -> tuple[str, str] | None:
        resp = self.http.get(self.API, params={
            "action": "query", "prop": "extracts|pageprops", "explaintext": 1, "exsectionformat": "plain",
            "redirects": 1, "titles": title, "format": "json", "utf8": 1,
        })
        if resp is None:
            return None
        pages = (resp.json().get("query") or {}).get("pages") or {}
        for page in pages.values():
            if "missing" in page or "disambiguation" in (page.get("pageprops") or {}):
                return None
            text = clean_wikipedia_extract(page.get("extract") or "")
            if len(text) < 500 or "may refer to:" in text[:300]:
                return None
            return page.get("title", title), text
        return None


def _strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or ""))


# ---------------------------------------------------------------------------
# Project Gutenberg
# ---------------------------------------------------------------------------


@dataclass
class CatalogEntry:
    ebook_id: str
    title: str
    authors: str
    subjects: str
    title_tokens: frozenset = field(repr=False, default=frozenset())
    author_tokens: frozenset = field(repr=False, default=frozenset())
    subject_tokens: frozenset = field(repr=False, default=frozenset())

    @property
    def author(self) -> str:
        """'Hobbes, Thomas, 1588-1679; X [Editor]' -> 'Thomas Hobbes'; 'Augustine, of Hippo, Saint' -> 'Augustine of Hippo'."""
        first = re.sub(r"\s*\([^)]*\)", "", self.authors.split(";")[0])
        parts = [p.strip() for p in first.split(",") if p.strip() and not re.search(r"\d", p)]
        if not parts:
            return "Unknown"
        if len(parts) >= 2 and parts[1].startswith("of "):
            return f"{parts[0]} {parts[1]}"
        if len(parts) >= 2 and len(parts[0].split()) == 1:
            return f"{parts[1]} {parts[0]}"
        return parts[0]


class GutenbergCatalog:
    CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
    MAIN_SITE = "https://www.gutenberg.org"

    def __init__(self, cache_dir: Path, http: Http, *, max_age_days: int = 30, mirror: str = "https://aleph.pglaf.org"):
        self.cache_path = Path(cache_dir) / "pg_catalog.csv.gz"
        self.http = http
        self.max_age_days = max_age_days
        self.mirror = mirror.rstrip("/")
        self._entries: list[CatalogEntry] | None = None

    def _load(self) -> list[CatalogEntry]:
        if self._entries is not None:
            return self._entries
        fresh = self.cache_path.exists() and time.time() - self.cache_path.stat().st_mtime < self.max_age_days * 86400
        if not fresh:
            resp = self.http.get(self.CATALOG_URL)
            if resp is not None and resp.content:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_bytes(resp.content)
        if not self.cache_path.exists():
            return []
        self._entries = parse_catalog(gzip.decompress(self.cache_path.read_bytes()).decode("utf-8", errors="replace"))
        return self._entries

    def search(self, query: str, k: int = 5, *, strict: bool = False) -> list[CatalogEntry]:
        """Books whose title and author match the query.

        ``strict`` (used when the organism chooses by itself) requires both an
        author and a title word to match, so a vague wish such as "curiosity"
        cannot pull in "The City Curious". Among equal matches the lower
        Gutenberg number wins: it is usually the standard edition.
        """
        q = token_set(query)
        if not q:
            return []
        scored = []
        for e in self._load():
            in_title, in_author = len(q & e.title_tokens), len(q & e.author_tokens)
            if not (in_title and in_author) if strict else not (in_title or in_author):
                continue
            covered = len(q & (e.title_tokens | e.author_tokens | e.subject_tokens)) / len(q)
            if covered < 0.5:
                continue
            exact_title = e.title_tokens == q or e.title_tokens | e.author_tokens == q
            scored.append((covered, exact_title, in_title + in_author, -int(e.ebook_id), e))
        scored.sort(key=lambda row: row[:4], reverse=True)
        return [row[-1] for row in scored[:k]]

    def fetch_text(self, entry: CatalogEntry) -> tuple[str, str] | None:
        """(text, url) from a mirror first, then from Gutenberg itself."""
        for base in (self.mirror, self.MAIN_SITE):
            url = f"{base}/cache/epub/{entry.ebook_id}/pg{entry.ebook_id}.txt"
            resp = self.http.get(url)
            if resp is not None and len(resp.content) > 2000:
                return resp.content.decode("utf-8", errors="replace"), url
        return None


def parse_catalog(text: str) -> list[CatalogEntry]:
    """English texts from Project Gutenberg's pg_catalog.csv."""
    entries = []
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("Type") != "Text" or "en" not in (row.get("Language") or "").split("; "):
            continue
        title = re.sub(r"\s+", " ", row.get("Title") or "").strip()
        authors = row.get("Authors") or ""
        subjects = f"{row.get('Subjects') or ''} {row.get('Bookshelves') or ''}"
        entries.append(CatalogEntry(
            ebook_id=(row.get("Text#") or "").strip(),
            title=title,
            authors=authors,
            subjects=subjects,
            title_tokens=token_set(title),
            author_tokens=token_set(re.sub(r"\d|\[[^\]]*\]", " ", authors)),
            subject_tokens=token_set(subjects),
        ))
    return [e for e in entries if e.ebook_id.isdigit()]


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


class SemanticScholar:
    API = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, http: Http, api_key: str | None = None):
        self.http = http
        self.api_key = api_key

    def find(self, query: str, k: int = 1) -> list[dict[str, Any]]:
        resp = self.http.get(
            self.API,
            params={"query": query, "limit": 8, "fields": "title,authors,year,abstract,url,venue,citationCount"},
            headers={"x-api-key": self.api_key} if self.api_key else None,
        )
        if resp is None:
            return []
        papers = [p for p in resp.json().get("data") or [] if len(p.get("abstract") or "") >= 200]
        papers = [p for p in papers if coverage(query, f"{p.get('title', '')} {p.get('abstract', '')}") >= 0.5]
        papers.sort(key=lambda p: (coverage(query, p.get("title", "")), p.get("citationCount") or 0), reverse=True)
        return papers[:k]


# ---------------------------------------------------------------------------
# The librarian
# ---------------------------------------------------------------------------


_OWNED = "already owned"  # returned by the _from_* helpers when the text is already in the library


class Librarian:
    def __init__(self, config: "LibrarianConfig", library_dir: Path, *, http: Http | None = None, owned_dirs: list[Path] | None = None):
        self.config = config
        self.dir = Path(library_dir)
        self.owned_dirs = [Path(d) for d in owned_dirs or []]  # e.g. the shared corpus: never fetch what it has
        self.http = http or Http(
            user_agent(config.contact),
            timeout=config.timeout_seconds,
            min_interval=config.min_request_interval_seconds,
        )
        self.wikipedia = Wikipedia(self.http)
        self.gutenberg = GutenbergCatalog(
            self.dir / ".cache", self.http, max_age_days=config.catalog_max_age_days, mirror=config.gutenberg_mirror
        )
        self.scholar = SemanticScholar(self.http, os.environ.get(config.semantic_scholar_api_key_env) or None)
        self.log_path = self.dir / "acquisitions.jsonl"
        self.used: dict[str, int] = {"encyclopedia": 0, "book": 0, "paper": 0}
        self.missed: list[str] = []  # looked for, not found (or the source was busy)
        self.owned: list[str] = []  # wished for, but already in the library

    def history(self) -> list[dict[str, Any]]:
        return read_jsonl(self.log_path)

    def _known(self, url: str) -> bool:
        return any(row.get("source_url") == url for row in self.history())

    def _quota_left(self, kind: str) -> bool:
        limit = {
            "encyclopedia": self.config.max_articles_per_run,
            "book": self.config.max_books_per_run,
            "paper": self.config.max_papers_per_run,
        }[kind]
        return self.used[kind] < limit

    def acquire(self, wish: ReadingWish, *, reason: str, question_id: str | None = None, heartbeat: int | None = None) -> list[Acquisition]:
        """Look up what the organism wished to read. Returns what was actually added."""
        self.missed, self.owned = [], []
        found: list[Acquisition] = []
        context = {"reason": reason, "question_id": question_id, "heartbeat": heartbeat}
        if "wikipedia" in self.config.sources:
            for topic in wish.topics[:2]:
                if self._quota_left("encyclopedia"):
                    self._collect(found, self._from_wikipedia(topic, **context), f"Wikipedia: {topic}")
        if "gutenberg" in self.config.sources:
            for query in wish.books[:1]:
                if self._quota_left("book"):
                    self._collect(found, self._from_gutenberg(query, self.dir / "books", **context), f"book: {query}")
        if "semantic_scholar" in self.config.sources:
            for query in wish.papers[:1]:
                if self._quota_left("paper"):
                    self._collect(found, self._from_scholar(query, **context), f"paper: {query}")
        return found

    def add_book(self, query: str, target_dir: Path, *, reason: str = "added by a human") -> tuple[Acquisition | None, list[CatalogEntry]]:
        """Find a book in Gutenberg's catalogue and download it into ``target_dir`` (no quota)."""
        candidates = self.gutenberg.search(query, k=5)
        if not candidates:
            return None, []
        got = self._from_gutenberg(query, Path(target_dir), reason=reason, candidates=candidates)
        return (got if isinstance(got, Acquisition) else None), candidates

    def _collect(self, found: list[Acquisition], acquisition: "Acquisition | str | None", wish: str) -> None:
        if acquisition is _OWNED:
            self.owned.append(wish)
            return
        if acquisition is None:
            self.missed.append(wish)
            return
        found.append(acquisition)
        self.used[acquisition.kind] += 1
        write_jsonl(self.log_path, asdict(acquisition))

    def _from_wikipedia(self, topic: str, **context) -> Acquisition | None:
        result = self.wikipedia.find(topic)
        if result is None:
            return None
        title, text, url = result
        if self._known(url):
            return _OWNED
        meta = {
            "title": title, "author": "Wikipedia", "cite_as": "WIKI", "source_url": url,
            "license": "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); text by Wikipedia contributors",
        }
        return self._save("encyclopedia", self.dir / "encyclopedia", title, meta, text, url, **context)

    def _from_gutenberg(self, query: str, target_dir: Path, *, candidates: list[CatalogEntry] | None = None, **context) -> Acquisition | None:
        # Only the best match: if it is owned already, a weaker match is not a substitute.
        for entry in (candidates or self.gutenberg.search(query, k=1, strict=True))[:1]:
            page = f"https://www.gutenberg.org/ebooks/{entry.ebook_id}"
            owned = [target_dir, *self.owned_dirs]
            if self._known(page) or any(already_in(d, entry.ebook_id, entry.title, entry.author) for d in owned):
                return _OWNED
            fetched = self.gutenberg.fetch_text(entry)
            if fetched is None:
                return None
            text, _ = fetched
            if len(text) > self.config.max_book_chars:
                return None
            meta = {
                "title": entry.title, "author": entry.author, "cite_as": "BOOK", "source_url": page,
                "ebook_id": entry.ebook_id,
                "license": "Project Gutenberg text; public domain in the United States. Check your country's law before redistributing.",
            }
            return self._save("book", target_dir, f"{entry.author} {entry.title}", meta, text, page, **context)
        return None

    def _from_scholar(self, query: str, **context) -> Acquisition | None:
        for paper in self.scholar.find(query, k=2):
            url = paper.get("url") or ""
            if not url:
                continue
            if self._known(url):
                return _OWNED
            authors = ", ".join(a.get("name", "") for a in (paper.get("authors") or [])[:3]) or "Unknown"
            year = str(paper.get("year") or "")
            body = f"{paper.get('title', '')}\n\n{authors} ({year}). {paper.get('venue') or ''}\n\nAbstract:\n{paper['abstract']}"
            meta = {
                "title": paper.get("title", "Untitled"), "author": authors, "year": year, "cite_as": "PAPER",
                "source_url": url, "license": "Abstract and metadata from Semantic Scholar, kept for research use",
            }
            return self._save("paper", self.dir / "papers", paper.get("title", "paper"), meta, body, url, **context)
        return None

    def _save(self, kind: str, folder: Path, name: str, meta: dict[str, str], text: str, url: str, *, reason: str = "", question_id: str | None = None, heartbeat: int | None = None) -> Acquisition:
        folder.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60] or kind
        path = folder / f"{slug}_{stable_id(url, length=6)}.md"
        meta = {**meta, "acquired_for": question_id or "", "acquired_at_heartbeat": str(heartbeat or ""), "reason": reason}
        lines = ["---", *(f"{k}: {_front_matter_value(v)}" for k, v in meta.items() if v != ""), "---", "", text.strip(), ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return Acquisition(kind, meta["title"], meta["author"], url, str(path), len(text), reason, question_id, heartbeat)


def _front_matter_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).replace("---", "-")).strip()


def already_in(folder: Path, ebook_id: str, title: str = "", author: str = "") -> bool:
    """True if the folder already holds this book, perhaps in another edition.

    Same Gutenberg number, the same title, or the same author with one title
    inside the other ("The Republic" and "The Republic of Plato"). Older corpus
    files do not all record their Gutenberg number, so titles matter.
    """
    if not folder.exists():
        return False
    by_id = re.compile(rf"^ebook_id:\s*{re.escape(ebook_id)}\s*$", re.MULTILINE)
    wanted = _normal_title(title)
    surname = author.split()[-1].lower() if author.split() else ""
    for path in folder.glob("*.*"):
        try:
            with path.open(encoding="utf-8", errors="ignore") as f:
                head = f.read(2000)
        except OSError:
            continue
        if by_id.search(head):
            return True
        found = re.search(r"^title:\s*(.+)$", head, re.MULTILINE)
        if not (wanted and found):
            continue
        have = _normal_title(found.group(1))
        if have == wanted:
            return True
        their_author = re.search(r"^author:\s*(.+)$", head, re.MULTILINE)
        same_author = bool(surname and their_author and surname in their_author.group(1).lower())
        if same_author and min(len(have), len(wanted)) >= 5 and (have in wanted or wanted in have):
            return True
    return False


def _normal_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", re.sub(r"^(the|a|an)\s+", "", title.strip().lower()))
