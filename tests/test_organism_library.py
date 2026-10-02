from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from curiosity_ai.config import LibrarianConfig
from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.librarian import (
    Acquisition,
    CatalogEntry,
    Http,
    Librarian,
    ReadingWish,
    clean_wikipedia_extract,
    parse_catalog,
)
from curiosity_ai.organism.senses import LexicalLibrary
from curiosity_ai.utils import strip_front_matter

from .organism_fakes import ScriptedLLM

CURIOSITY_ARTICLE = (
    "Curiosity is a quality related to inquisitive thinking, such as exploration, investigation, and learning. "
    "The information gap theory holds that curiosity arises when attention is drawn to a gap in knowledge. " * 6
    + "\n\nSee also\nWonder\n\nReferences\nLoewenstein 1994."
)

CATALOG_CSV = """Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves
3207,Text,2002-05-01,Leviathan,en,"Hobbes, Thomas, 1588-1679",Political science,JC,Philosophy
59699,Text,2019-06-01,"History of the U.S.S. Leviathan, cruiser",en,U.S.S. Leviathan History Committee,World War,D,
32406,Text,2010-05-01,The City Curious,en,"Boschère, Jean de, 1878-1953",Fairy tales,PZ,
3296,Text,2002-07-01,The Confessions of St. Augustine,en,"Augustine, of Hippo, Saint, 354-430; Pusey, E. B. [Translator]",Christian saints,BR,
77585,Text,2026-01-01,Confessions of St. Augustine,en,"Augustine, of Hippo, Saint, 354-430",Christian saints,BR,
9999,Text,2003-01-01,Leviathan,fr,"Hobbes, Thomas, 1588-1679",Political science,JC,
"""

BOOK = "*** START OF THE PROJECT GUTENBERG EBOOK LEVIATHAN ***\n" + "Curiosity, desire to know why and how, is a lust of the mind. " * 80


class Reply:
    def __init__(self, status=200, data=None, content=b"", headers=None):
        self.status_code = status
        self._data = data
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._data


class FakeInternet:
    """Serves canned replies for Wikipedia, Gutenberg and Semantic Scholar; counts requests."""

    def __init__(self, *, scholar_status=200, disambiguation=False):
        self.requests: list[str] = []
        self.scholar_status = scholar_status
        self.disambiguation = disambiguation

    def get(self, url, params=None, headers=None):
        params = params or {}
        self.requests.append(url)
        if "wikipedia.org" in url and params.get("list") == "search":
            return Reply(data={"query": {"search": [
                {"title": "Clickbait", "snippet": "exploit the curiosity gap"},
                {"title": "Curiosity", "snippet": "a quality related to inquisitive thinking"},
            ]}})
        if "wikipedia.org" in url:
            title = params["titles"]
            if title == "Curiosity" and self.disambiguation:
                return Reply(data={"query": {"pages": {"1": {"title": "Curiosity", "pageprops": {"disambiguation": ""}, "extract": "Curiosity may refer to:"}}}})
            if title == "Curiosity":
                return Reply(data={"query": {"pages": {"7": {"title": "Curiosity", "extract": CURIOSITY_ARTICLE}}}})
            return Reply(data={"query": {"pages": {"-1": {"title": title, "missing": ""}}}})
        if url.endswith("pg_catalog.csv.gz"):
            return Reply(content=gzip.compress(CATALOG_CSV.encode()))
        if "/cache/epub/" in url:
            return Reply(content=BOOK.encode())
        if "semanticscholar" in url:
            if self.scholar_status != 200:
                return Reply(status=self.scholar_status, headers={"Retry-After": "1000"})
            return Reply(data={"data": [{
                "title": "The psychology of curiosity: A review and reinterpretation",
                "authors": [{"name": "George Loewenstein"}], "year": 1994, "venue": "Psychological Bulletin",
                "url": "https://www.semanticscholar.org/paper/abc",
                "abstract": "Information gap theory of curiosity: curiosity arises when attention becomes focused on a gap in one's knowledge. " * 3,
                "citationCount": 3000,
            }]})
        return Reply(status=404)


class FakeSession:
    def __init__(self, internet: FakeInternet):
        self.internet = internet

    def get(self, url, params=None, timeout=None, headers=None):
        return self.internet.get(url, params, headers)


def make_librarian(tmp_path: Path, internet: FakeInternet, **config) -> Librarian:
    http = Http("test-agent", min_interval=0, sleep=lambda s: None, session=FakeSession(internet))
    return Librarian(LibrarianConfig(enabled=True, **config), tmp_path / "library", http=http)


def test_wikipedia_reference_sections_are_cut():
    assert clean_wikipedia_extract("Body text.\n\nSee also\nWonder\n\nReferences\nX") == "Body text."


def test_the_catalogue_keeps_english_texts_and_formats_authors():
    entries = parse_catalog(CATALOG_CSV)
    assert "9999" not in {e.ebook_id for e in entries}  # French
    authors = {e.ebook_id: e.author for e in entries}
    assert authors["3207"] == "Thomas Hobbes"
    assert authors["3296"] == "Augustine of Hippo"


def test_a_vague_wish_cannot_pull_in_a_junk_book(tmp_path: Path):
    lib = make_librarian(tmp_path, FakeInternet())
    assert lib.gutenberg.search("curiosity", strict=True) == []
    assert [e.ebook_id for e in lib.gutenberg.search("Augustine Confessions", strict=True)] == ["3296", "77585"]
    assert lib.gutenberg.search("Leviathan")[0].ebook_id == "3207"


def test_acquired_texts_carry_their_provenance(tmp_path: Path):
    internet = FakeInternet()
    lib = make_librarian(tmp_path, internet)
    got = lib.acquire(
        ReadingWish(topics=["Curiosity"], books=["Thomas Hobbes Leviathan"], papers=["information gap theory of curiosity"]),
        reason="my books said little about Q3", question_id="Q3", heartbeat=12,
    )
    assert [a.kind for a in got] == ["encyclopedia", "book", "paper"]
    meta, body = strip_front_matter(Path(got[0].path).read_text(encoding="utf-8"))
    assert meta["cite_as"] == "WIKI" and meta["acquired_for"] == "Q3" and meta["acquired_at_heartbeat"] == "12"
    assert "CC BY-SA" in meta["license"] and "References" not in body
    book_meta, _ = strip_front_matter(Path(got[1].path).read_text(encoding="utf-8"))
    assert book_meta["ebook_id"] == "3207" and book_meta["author"] == "Thomas Hobbes"
    assert any("aleph.pglaf.org" in url for url in internet.requests)  # Gutenberg asks programs to use mirrors
    assert len(lib.history()) == 3


def test_disambiguation_pages_are_skipped_for_a_real_article(tmp_path: Path):
    lib = make_librarian(tmp_path, FakeInternet(disambiguation=True))
    assert lib.acquire(ReadingWish(topics=["Curiosity"]), reason="r") == []
    assert lib.missed == ["Wikipedia: Curiosity"]


def test_nothing_is_acquired_twice_and_quotas_hold(tmp_path: Path):
    lib = make_librarian(tmp_path, FakeInternet(), max_books_per_run=1)
    assert len(lib.acquire(ReadingWish(books=["Hobbes Leviathan"]), reason="r")) == 1
    assert lib.acquire(ReadingWish(books=["Augustine Confessions"]), reason="r") == []  # quota used
    lib.used["book"] = 0
    assert lib.acquire(ReadingWish(books=["Hobbes Leviathan"]), reason="r") == []  # already owned
    assert lib.owned == ["book: Hobbes Leviathan"]


def test_a_busy_source_is_skipped_without_stopping_the_others(tmp_path: Path):
    lib = make_librarian(tmp_path, FakeInternet(scholar_status=429))
    got = lib.acquire(ReadingWish(topics=["Curiosity"], papers=["information gap theory"]), reason="r")
    assert [a.kind for a in got] == ["encyclopedia"]
    assert lib.missed == ["paper: information gap theory"]


def test_http_waits_as_asked_then_rests_the_host():
    waits = []

    class Session:
        calls = 0

        def get(self, url, params=None, timeout=None, headers=None):
            Session.calls += 1
            return Reply(status=429, headers={"Retry-After": "7"})

    now = [0.0]
    http = Http("ua", min_interval=0, sleep=lambda s: (waits.append(s), now.__setitem__(0, now[0] + s)), clock=lambda: now[0], session=Session())
    assert http.get("https://en.wikipedia.org/w/api.php") is None
    assert waits == [7.0] and Session.calls == 2
    assert http.get("https://en.wikipedia.org/w/api.php") is None
    assert Session.calls == 2  # resting: no request at all


def test_acquired_texts_are_read_and_labelled(tmp_path: Path):
    lib = make_librarian(tmp_path, FakeInternet())
    lib.acquire(ReadingWish(topics=["Curiosity"], books=["Hobbes Leviathan"]), reason="r")
    shelf = LexicalLibrary([tmp_path / "library"])
    hits = shelf.search("information gap curiosity lust of the mind", k=4)
    citations = " ".join(h.citation for h in hits)
    assert "[WIKI:Wikipedia:Curiosity" in citations and "[BOOK:Thomas_Hobbes:Leviathan" in citations
    assert {h.note for h in hits} == {"encyclopedia article", "book it acquired"}
    assert "Wikipedia, Curiosity" in shelf.titles()


class FakeLibrarian:
    """Stands in for the internet: 'acquires' one encyclopedia article on demand."""

    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.wishes: list[ReadingWish] = []
        self.missed: list[str] = []
        self.owned: list[str] = []

    def acquire(self, wish, *, reason, question_id=None, heartbeat=None):
        self.wishes.append(wish)
        folder = self.library_dir / "encyclopedia"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"wonder_{heartbeat}.md"
        path.write_text("---\ntitle: Wonder\nauthor: Wikipedia\ncite_as: WIKI\n---\n\n" + "Wonder is an emotion comparable to surprise that people feel when perceiving something rare or unexpected. " * 5, encoding="utf-8")
        self.missed = ["paper: information gap theory of curiosity"]
        return [Acquisition("encyclopedia", "Wonder", "Wikipedia", f"https://w/{heartbeat}", str(path), 500, reason, question_id, heartbeat)]


def test_a_silent_library_sends_the_organism_to_look_elsewhere(config):
    config.organism.librarian.hunger_informativeness = 0.9  # the scripted texts address 43%: hungry
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    org.librarian = FakeLibrarian(org.library_dir)
    ep = org.heartbeat()
    assert ep.acquisitions == ["Wikipedia: Wonder"]
    assert ep.library_misses == ["paper: information gap theory of curiosity"]
    wish = org.librarian.wishes[0]
    assert wish.topics == ["Curiosity"] and wish.books == ["Thomas Hobbes Leviathan"]
    assert org.senses.library.search("wonder emotion rare unexpected", k=1)[0].note == "encyclopedia article"
    assert "went to the library" in (org.home / "diary.md").read_text(encoding="utf-8")
    assert json.loads((org.home / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])["acquisitions"] == ["Wikipedia: Wonder"]


def test_no_library_visit_when_the_books_answered_or_too_soon(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    org.librarian = FakeLibrarian(org.library_dir)
    assert org.heartbeat().acquisitions == []  # 43% addressed is above the default hunger level
    config.organism.librarian.hunger_informativeness = 0.9
    for other in org.state.questions.values():
        if other.id != "Q1":
            other.status = "settled"  # keep the choice on Q1
    q = org.state.questions["Q1"]
    q.last_library_visit = org.state.heartbeat  # it has just been to the library for Q1
    q.last_visited = None
    ep = org.heartbeat()  # one heartbeat later: well inside the 4-heartbeat cooldown
    assert ep.question_id == "Q1" and ep.acquisitions == []


@pytest.mark.parametrize("item, expected", [
    ({"author": "Thomas Hobbes", "title": "Leviathan"}, "Thomas Hobbes Leviathan"),
    ('"Hobbes, Leviathan"', "Hobbes Leviathan"),
])
def test_book_wishes_become_author_title_queries(item, expected):
    from curiosity_ai.organism.organism import _book_queries

    assert _book_queries([item]) == [expected]


def test_catalog_entry_author_without_comma():
    assert CatalogEntry("1", "t", "Plato, 428? BCE-348? BCE", "").author == "Plato"


def test_a_book_already_in_the_corpus_is_not_fetched_again(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "hobbes.txt").write_text("---\ntitle: Leviathan, or the Matter of a Commonwealth\nauthor: Thomas Hobbes\n---\n\ntext", encoding="utf-8")
    http = Http("test-agent", min_interval=0, sleep=lambda s: None, session=FakeSession(FakeInternet()))
    lib = Librarian(LibrarianConfig(enabled=True), tmp_path / "library", http=http, owned_dirs=[corpus])
    assert lib.acquire(ReadingWish(books=["Hobbes Leviathan"]), reason="r") == []
    assert lib.owned == ["book: Hobbes Leviathan"] and lib.missed == []


def test_same_author_and_contained_title_means_same_book(tmp_path: Path):
    from curiosity_ai.organism.librarian import already_in

    (tmp_path / "republic.txt").write_text("---\ntitle: The Republic\nauthor: Plato\n---\n", encoding="utf-8")
    assert already_in(tmp_path, "150", "The Republic of Plato", "Plato")
    assert not already_in(tmp_path, "150", "The Republic of Plato", "Cicero")
    assert not already_in(tmp_path, "999", "Laws", "Plato")
