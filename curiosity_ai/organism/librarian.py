"""The librarian: how the organism grows its own library.

When the texts it owns stay silent on a question, a curious reader goes and
looks elsewhere. The librarian looks in four places:

* Wikipedia, for concepts and thinkers (CC BY-SA; stored with attribution);
* Project Gutenberg, for whole public-domain books. It searches the
  catalogue file that Gutenberg publishes for programs and downloads the text
  from a mirror, as Gutenberg asks automated tools to do;
* Semantic Scholar, for abstracts of scientific papers (rate-limited without
  a free API key);
* arXiv, for abstracts of preprints (free, at most one request every three
  seconds).

Everything it acquires goes into the organism's own library folder, with
front matter saying what it is, where it came from, its license, and why it
was acquired (which question, at which heartbeat). A text is kept only if it
shares enough words with the question and the topic (the Mars rover
"Curiosity" shares only its name). A source that is busy is reported as busy,
not as having nothing. Hourly quotas, a minimum interval between requests and
respect for Retry-After keep it a polite visitor.
"""
from __future__ import annotations

import csv
import gzip
import html
import io
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import quote, urlparse

import requests

from ..utils import read_jsonl, stable_id, write_jsonl
from .textutil import coverage, keywords_of, token_set

if TYPE_CHECKING:
    from ..config import LibrarianConfig

USER_AGENT_PRODUCT = "CuriosityAI/0.8"

# Other worlds, stars and laboratory devices: a text about one of them is about another place, unless the
# question or the topic names it (a topic on Venus wants Venus).
_ELSEWHERE_RE = re.compile(
    r"\b(?:Venus|Venusian|Mars|Martian|Jupiter|Jovian|Saturn|Saturnian|Uranus|Neptune|Pluto|Ganymede|Enceladus"
    r"|exoplanets?|magnetars?|pulsars?|neutron stars?|black holes?|white dwarfs?|quasars?|galax(?:y|ies)|accretion dis[ck]s?"
    r"|tokamaks?|stellarators?|in the solar wind|solar[- ]wind plasmas?|solar flares?)\b",
    re.IGNORECASE,
)


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
    papers: list[str] = field(default_factory=list)  # search keywords


class SourceBusy(Exception):
    """The source could not be asked (rate limit, server trouble, no network): not the same as 'nothing found'."""


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
        host_intervals: dict[str, float] | None = None,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.min_interval = min_interval
        self.host_intervals = host_intervals or {}
        self.max_wait = max_wait
        self.busy = False  # True when the last request could not be answered (rate limit, server, network)
        self._sleep = sleep
        self._clock = clock
        self.session = session or requests.Session()
        self._last: dict[str, float] = {}
        self._resting_until: dict[str, float] = {}

    def resting(self, url: str) -> bool:
        host = urlparse(url).netloc
        return self._clock() < self._resting_until.get(host, 0.0)

    def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> requests.Response | None:
        """The response, or None. After None, ``busy`` says whether the source could not be asked at all."""
        host = urlparse(url).netloc
        self.busy = False
        if self.resting(url):
            self.busy = True
            return None
        interval = self.host_intervals.get(host, self.min_interval)
        for attempt in range(2):
            wait = self._last.get(host, -1e9) + interval - self._clock()
            if wait > 0:
                self._sleep(wait)
            try:
                resp = self.session.get(
                    url, params=params, timeout=self.timeout, headers={"User-Agent": self.user_agent, **(headers or {})}
                )
            except requests.RequestException:
                self.busy = True
                return None
            finally:
                self._last[host] = self._clock()
            if resp.status_code in (429, 500, 502, 503, 504):
                retry = _retry_after(resp)
                if attempt == 0 and retry <= self.max_wait:
                    self._sleep(retry)
                    continue
                self._resting_until[host] = self._clock() + max(retry, 120)
                self.busy = True
                return None
            return resp if resp.status_code < 400 else None
        self.busy = True
        return None

    def get_or_busy(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> requests.Response | None:
        """Like get, but a source that could not be asked raises SourceBusy."""
        resp = self.get(url, params=params, headers=headers)
        if resp is None and self.busy:
            raise SourceBusy(urlparse(url).netloc)
        return resp


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
        resp = self.http.get_or_busy(self.API, params={
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
        resp = self.http.get_or_busy(self.API, params={
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
    def surname_tokens(self) -> frozenset:
        """The first author's family name ('Hobbes, Thomas' -> hobbes): a first name like John proves nothing."""
        first = re.sub(r"\s*\([^)]*\)", "", self.authors.split(";")[0]).split(",")[0]
        return token_set(re.sub(r"\d|\[[^\]]*\]", " ", first))

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
            elif self.http.busy and not self.cache_path.exists():
                raise SourceBusy("www.gutenberg.org")
        if not self.cache_path.exists():
            return []
        self._entries = parse_catalog(gzip.decompress(self.cache_path.read_bytes()).decode("utf-8", errors="replace"))
        return self._entries

    def search(self, query: str, k: int = 5, *, strict: bool = False) -> list[CatalogEntry]:
        """Books whose title and author match the query.

        ``strict`` (used when the organism chooses by itself) requires the
        author's family name and a title word to match, so a vague wish such as
        "curiosity" cannot pull in "The City Curious", and a first name such as
        "John" cannot pull in a lawyer's diary from 1602. Among equal matches the
        lower Gutenberg number wins: it is usually the standard edition.
        """
        q = token_set(query)
        if not q:
            return []
        scored = []
        for e in self._load():
            in_title, in_author = len(q & e.title_tokens), len(q & e.author_tokens)
            if strict and not (in_title and q & e.surname_tokens):
                continue
            if not strict and not (in_title or in_author):
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
        busy = False
        for base in (self.mirror, self.MAIN_SITE):
            url = f"{base}/cache/epub/{entry.ebook_id}/pg{entry.ebook_id}.txt"
            resp = self.http.get(url)
            if resp is not None and len(resp.content) > 2000:
                return resp.content.decode("utf-8", errors="replace"), url
            busy = busy or self.http.busy
        if busy:
            raise SourceBusy("Project Gutenberg")
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

    def find(self, query: str, k: int = 1, *, until_year: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"query": query, "limit": 8, "fields": "title,authors,year,abstract,url,venue,citationCount"}
        if until_year:
            params["year"] = f"-{until_year}"
        resp = self.http.get_or_busy(self.API, params=params, headers={"x-api-key": self.api_key} if self.api_key else None)
        if resp is None:
            return []
        papers = [p for p in resp.json().get("data") or [] if len(p.get("abstract") or "") >= 200]
        if until_year:
            papers = [p for p in papers if p.get("year") and int(p["year"]) <= until_year]
        papers = [p for p in papers if coverage(query, f"{p.get('title', '')} {p.get('abstract', '')}") >= 0.5]
        papers.sort(key=lambda p: (coverage(query, p.get("title", "")), p.get("citationCount") or 0), reverse=True)
        return papers[:k]


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

_ATOM = "{http://www.w3.org/2005/Atom}"


class Arxiv:
    """arXiv's public API: abstracts of preprints, free, one request every three seconds."""

    API = "https://export.arxiv.org/api/query"

    def __init__(self, http: Http):
        self.http = http

    def find(self, query: str, k: int = 1, *, until_year: int | None = None) -> list[dict[str, Any]]:
        words = keywords_of(query, 6).split()
        if not words:
            return []
        search = " AND ".join(f"all:{w}" for w in words)
        if until_year:
            search += f" AND submittedDate:[190001010000 TO {until_year}12312359]"
        resp = self.http.get_or_busy(self.API, params={"search_query": search, "start": 0, "max_results": 8})
        if resp is None:
            return []
        papers = parse_arxiv(resp.content)
        papers = [p for p in papers if len(p["abstract"]) >= 200 and coverage(query, f"{p['title']} {p['abstract']}") >= 0.5]
        papers.sort(key=lambda p: coverage(query, p["title"]), reverse=True)
        return papers[:k]


def parse_arxiv(content: bytes) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    papers = []
    for entry in root.findall(f"{_ATOM}entry"):
        def text(tag: str) -> str:
            node = entry.find(f"{_ATOM}{tag}")
            return re.sub(r"\s+", " ", node.text or "").strip() if node is not None else ""

        url = text("id")
        if not url or "arxiv.org" not in url:
            continue
        papers.append({
            "title": text("title"),
            "abstract": text("summary"),
            "url": url,
            "year": text("published")[:4],
            "authors": [{"name": re.sub(r"\s+", " ", a.findtext(f"{_ATOM}name") or "").strip()} for a in entry.findall(f"{_ATOM}author")],
            "venue": "arXiv",
        })
    return papers


# ---------------------------------------------------------------------------
# The librarian
# ---------------------------------------------------------------------------


_OWNED = "already owned"  # returned by the _from_* helpers when the text is already in the library
_OFF_TOPIC = "off topic"  # found, but it shares too few words with the question and the topic
_HOUR = 3600.0


class Librarian:
    def __init__(
        self,
        config: "LibrarianConfig",
        library_dir: Path,
        *,
        http: Http | None = None,
        owned_dirs: list[Path] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.dir = Path(library_dir)
        self.owned_dirs = [Path(d) for d in owned_dirs or []]  # e.g. the shared corpus: never fetch what it has
        self.http = http or Http(
            user_agent(config.contact),
            timeout=config.timeout_seconds,
            min_interval=config.min_request_interval_seconds,
            host_intervals={"export.arxiv.org": config.arxiv_interval_seconds},
        )
        self.clock = clock
        self.wikipedia = Wikipedia(self.http)
        self.gutenberg = GutenbergCatalog(
            self.dir / ".cache", self.http, max_age_days=config.catalog_max_age_days, mirror=config.gutenberg_mirror
        )
        self.scholar = SemanticScholar(self.http, os.environ.get(config.semantic_scholar_api_key_env) or None)
        self.arxiv = Arxiv(self.http)
        self.log_path = self.dir / "acquisitions.jsonl"
        self._acquired_at: dict[str, list[float]] = {"encyclopedia": [], "book": [], "paper": []}
        self.missed: list[str] = []  # looked for, found nothing
        self.owned: list[str] = []  # wished for, but already in the library
        self.busy: list[str] = []  # could not look: the source was busy or unreachable
        self.rejected: list[str] = []  # found, but off the topic, so not kept
        self.failed_searches: list[str] = []  # searches that found nothing in this run (not repeated)
        self.approve: Callable[[str, str], bool] | None = None

    def history(self) -> list[dict[str, Any]]:
        return read_jsonl(self.log_path)

    def _known(self, url: str) -> bool:
        return any(row.get("source_url") == url for row in self.history())

    def _quota_left(self, kind: str) -> bool:
        limit = {
            "encyclopedia": self.config.max_articles_per_hour,
            "book": self.config.max_books_per_hour,
            "paper": self.config.max_papers_per_hour,
        }[kind]
        now = self.clock()
        recent = [t for t in self._acquired_at[kind] if now - t < _HOUR]
        self._acquired_at[kind] = recent
        return len(recent) < limit

    def acquire(
        self,
        wish: ReadingWish,
        *,
        reason: str,
        question_id: str | None = None,
        heartbeat: int | None = None,
        context: str = "",
        until_year: int | None = None,
        approve: Callable[[str, str], bool] | None = None,
    ) -> list[Acquisition]:
        """Look up what the organism wished to read. Returns what was actually added.

        ``context`` (the question and the topic's words) is what a found text
        must be about to be kept. ``approve(title, beginning)``, when given (the
        organism's judge), must also accept it: shared words are not enough
        ("Magnetosphere of Saturn" for a question about Earth's). ``until_year``
        keeps out papers published later, and Wikipedia altogether, since
        today's articles report later work (for tests that pretend it is that year).
        """
        self.missed, self.owned, self.busy, self.rejected = [], [], [], []
        self.approve = approve
        found: list[Acquisition] = []
        meta = {"reason": reason, "question_id": question_id, "heartbeat": heartbeat}
        sources = self.config.sources
        if "wikipedia" in sources and not until_year:
            for topic in wish.topics[:2]:
                if self._quota_left("encyclopedia") and self._fresh(f"wiki:{topic}"):
                    self._attempt(found, f"wiki:{topic}", f"Wikipedia: {topic}", lambda t=topic: self._from_wikipedia(t, context, **meta))
        if "gutenberg" in sources:
            for query in wish.books[:1]:
                if self._quota_left("book") and self._fresh(f"book:{query}"):
                    self._attempt(found, f"book:{query}", f"book: {query}", lambda q=query: self._from_gutenberg(q, self.dir / "books", context=context, **meta))
        for query in wish.papers[:2]:
            if self._quota_left("paper") and self._fresh(f"paper:{query}"):
                self._attempt(found, f"paper:{query}", f"paper: {query}", lambda q=query: self._from_papers(q, context, until_year, **meta))
        return found

    def add_book(self, query: str, target_dir: Path, *, reason: str = "added by a human") -> tuple[Acquisition | None, list[CatalogEntry]]:
        """Find a book in Gutenberg's catalogue and download it into ``target_dir`` (no quota)."""
        candidates = self.gutenberg.search(query, k=5)
        if not candidates:
            return None, []
        got = self._from_gutenberg(query, Path(target_dir), reason=reason, candidates=candidates)
        return (got if isinstance(got, Acquisition) else None), candidates

    def _fresh(self, key: str) -> bool:
        """A search that found nothing earlier in this run is not repeated."""
        return _normal_query(key) not in {_normal_query(k) for k in self.failed_searches}

    def _attempt(self, found: list[Acquisition], key: str, wish: str, fetch: Callable[[], "Acquisition | str | None"]) -> None:
        try:
            result = fetch()
        except SourceBusy as exc:
            self.busy.append(f"{wish} ({exc})")
            return
        if result is _OWNED:
            self.owned.append(wish)
        elif isinstance(result, str) and result.startswith(_OFF_TOPIC):
            self.rejected.append(f"{wish} ({result[len(_OFF_TOPIC) + 1:].strip() or 'off topic'})")
        elif result is None:
            self.missed.append(wish)
            self.failed_searches.append(key)
        else:
            found.append(result)
            self._acquired_at[result.kind].append(self.clock())
            write_jsonl(self.log_path, asdict(result))

    def _relevant(self, query: str, context: str, text: str) -> bool:
        """Does the found text share enough words with the question and topic, beyond the words searched for?"""
        if not context.strip():
            return True
        shared = (token_set(context) - token_set(query)) & token_set(text)
        return len(shared) >= self.config.min_relevance_words

    def _approved(self, title: str, beginning: str) -> bool:
        return self.approve is None or self.approve(title, beginning)

    @staticmethod
    def _elsewhere(title: str, context: str) -> str:
        """The other world, star or device a title is about, if the question and topic never mention it.

        A small judge let in "Whistler wave propagation through the ionosphere of Venus", "Resonant inverse
        Compton scattering in magnetar magnetospheres" and "Runaway electron interactions with whistler waves
        in tokamak plasmas" for questions about Earth's inner magnetosphere: they share every word but the place.
        """
        found = _ELSEWHERE_RE.search(title or "")
        if not found:
            return ""
        return "" if token_set(found.group(0)) <= token_set(context) else found.group(0)

    def _from_wikipedia(self, topic: str, context: str = "", **meta) -> "Acquisition | str | None":
        result = self.wikipedia.find(topic)
        if result is None:
            return None
        title, text, url = result
        if self._known(url):
            return _OWNED
        if not self._relevant(topic, context, f"{title} {text[:3000]}"):
            return f"{_OFF_TOPIC}: found '{title}'"
        if self._elsewhere(title, context):
            return f"{_OFF_TOPIC}: found '{title}', which is about {self._elsewhere(title, context)}"
        if not self._approved(f"Wikipedia: {title}", text[:1500]):
            return f"{_OFF_TOPIC}: found '{title}', which the judge found not useful"
        front = {
            "title": title, "author": "Wikipedia", "cite_as": "WIKI", "source_url": url,
            "license": "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); text by Wikipedia contributors",
        }
        return self._save("encyclopedia", self.dir / "encyclopedia", title, front, text, url, **meta)

    def _from_gutenberg(
        self, query: str, target_dir: Path, *, candidates: list[CatalogEntry] | None = None, context: str = "", **meta
    ) -> "Acquisition | str | None":
        # Only the best match: if it is owned already, a weaker match is not a substitute.
        for entry in (candidates or self.gutenberg.search(query, k=1, strict=True))[:1]:
            page = f"https://www.gutenberg.org/ebooks/{entry.ebook_id}"
            owned = [target_dir, *self.owned_dirs]
            if self._known(page) or any(already_in(d, entry.ebook_id, entry.title, entry.author) for d in owned):
                return _OWNED
            if candidates is None and not self._relevant(query, context, f"{entry.title} {entry.subjects}"):
                return f"{_OFF_TOPIC}: found '{entry.title}' by {entry.author}"
            if candidates is None and not self._approved(f"{entry.title}, by {entry.author}", entry.subjects):
                return f"{_OFF_TOPIC}: found '{entry.title}' by {entry.author}, which the judge found not useful"
            fetched = self.gutenberg.fetch_text(entry)
            if fetched is None:
                return None
            text, _ = fetched
            if len(text) > self.config.max_book_chars:
                return None
            front = {
                "title": entry.title, "author": entry.author, "cite_as": "BOOK", "source_url": page,
                "ebook_id": entry.ebook_id,
                "license": "Project Gutenberg text; public domain in the United States. Check your country's law before redistributing.",
            }
            return self._save("book", target_dir, f"{entry.author} {entry.title}", front, text, page, **meta)
        return None

    def _from_papers(self, query: str, context: str, until_year: int | None, **meta) -> "Acquisition | str | None":
        """Semantic Scholar first, then arXiv. Busy only if no source could be asked."""
        busy: list[str] = []
        outcome: "Acquisition | str | None" = None
        searches = [("semantic_scholar", self.scholar), ("arxiv", self.arxiv)]
        for name, source in searches:
            if name not in self.config.sources:
                continue
            try:
                papers = source.find(query, k=2, until_year=until_year)
            except SourceBusy as exc:
                busy.append(str(exc))
                continue
            for paper in papers:
                result = self._save_paper(paper, query, context, name, **meta)
                if isinstance(result, Acquisition) or result is _OWNED:
                    return result
                outcome = outcome or result  # off topic: remember, but try the next source
        if outcome is None and busy:
            raise SourceBusy(", ".join(busy))
        return outcome

    def _save_paper(self, paper: dict[str, Any], query: str, context: str, source: str, **meta) -> "Acquisition | str | None":
        url = paper.get("url") or ""
        if not url:
            return None
        if self._known(url):
            return _OWNED
        title = paper.get("title", "Untitled")
        if not self._relevant(query, context, f"{title} {paper.get('abstract', '')}"):
            return f"{_OFF_TOPIC}: found '{one_line_title(title)}'"
        if self._elsewhere(title, context):
            return f"{_OFF_TOPIC}: found '{one_line_title(title)}', which is about {self._elsewhere(title, context)}"
        if not self._approved(title, paper.get("abstract", "")):
            return f"{_OFF_TOPIC}: found '{one_line_title(title)}', which the judge found not useful"
        authors = ", ".join(a.get("name", "") for a in (paper.get("authors") or [])[:3]) or "Unknown"
        year = str(paper.get("year") or "")
        body = f"{title}\n\n{authors} ({year}). {paper.get('venue') or ''}\n\nAbstract:\n{paper['abstract']}"
        license_note = {
            "semantic_scholar": "Abstract and metadata from Semantic Scholar, kept for research use",
            "arxiv": "Abstract from arXiv, kept for research use; see the paper's own license at its arXiv page",
        }[source]
        front = {"title": title, "author": authors, "year": year, "cite_as": "PAPER", "source_url": url, "license": license_note}
        return self._save("paper", self.dir / "papers", title, front, body, url, **meta)

    def _save(self, kind: str, folder: Path, name: str, meta: dict[str, str], text: str, url: str, *, reason: str = "", question_id: str | None = None, heartbeat: int | None = None) -> Acquisition:
        folder.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60] or kind
        path = folder / f"{slug}_{stable_id(url, length=6)}.md"
        meta = {**meta, "acquired_for": question_id or "", "acquired_at_heartbeat": str(heartbeat or ""), "reason": reason}
        lines = ["---", *(f"{k}: {_front_matter_value(v)}" for k, v in meta.items() if v != ""), "---", "", text.strip(), ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return Acquisition(kind, meta["title"], meta["author"], url, str(path), len(text), reason, question_id, heartbeat)


def one_line_title(title: str, limit: int = 80) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    return title if len(title) <= limit else title[: limit - 3] + "..."


def _normal_query(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", key.lower()).strip()


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
