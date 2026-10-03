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
44336,Text,2013-12-01,"Diary of John Manningham, Barrister-at-Law, 1602-1603",en,"Manningham, John, -1622",England -- Social life and customs,DA,
3296,Text,2002-07-01,The Confessions of St. Augustine,en,"Augustine, of Hippo, Saint, 354-430; Pusey, E. B. [Translator]",Christian saints,BR,
77585,Text,2026-01-01,Confessions of St. Augustine,en,"Augustine, of Hippo, Saint, 354-430",Christian saints,BR,
9999,Text,2003-01-01,Leviathan,fr,"Hobbes, Thomas, 1588-1679",Political science,JC,
"""

ROVER = (
    "Curiosity is a car-sized Mars rover exploring Gale crater and Mount Sharp on Mars as part of NASA's Mars Science "
    "Laboratory mission. It landed in 2012 and studies the Martian climate and geology, and whether the site ever offered "
    "environmental conditions favorable for microbial life. " * 4
)

BOOK = "*** START OF THE PROJECT GUTENBERG EBOOK LEVIATHAN ***\n" + "Curiosity, desire to know why and how, is a lust of the mind. " * 80


class Reply:
    def __init__(self, status=200, data=None, content=b"", headers=None):
        self.status_code = status
        self._data = data
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._data


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1703.00001v1</id>
    <published>2017-03-01T00:00:00Z</published>
    <title>Curiosity-driven exploration by self-supervised prediction</title>
    <summary>We formulate curiosity as the error in an agent's ability to predict the consequence of its own actions.
    The information gap between what the agent can predict and what happens drives exploration and learning, and
    curiosity helps the agent to explore its environment and learn skills that might be useful later.</summary>
    <author><name>Deepak Pathak</name></author>
  </entry>
</feed>"""


class FakeInternet:
    """Serves canned replies for Wikipedia, Gutenberg, Semantic Scholar and arXiv; counts requests."""

    def __init__(self, *, scholar_status=200, arxiv_status=200, disambiguation=False, rover=False):
        self.requests: list[str] = []
        self.params: list[dict] = []
        self.scholar_status = scholar_status
        self.arxiv_status = arxiv_status
        self.disambiguation = disambiguation
        self.rover = rover

    def get(self, url, params=None, headers=None):
        params = params or {}
        self.requests.append(url)
        self.params.append(params)
        if "wikipedia.org" in url and params.get("list") == "search":
            return Reply(data={"query": {"search": [
                {"title": "Clickbait", "snippet": "exploit the curiosity gap"},
                {"title": "Curiosity", "snippet": "a quality related to inquisitive thinking"},
            ]}})
        if "wikipedia.org" in url:
            title = params["titles"]
            if title == "Curiosity" and self.rover:
                return Reply(data={"query": {"pages": {"9": {"title": "Curiosity (rover)", "extract": ROVER}}}})
            if title == "Curiosity" and self.disambiguation:
                return Reply(data={"query": {"pages": {"1": {"title": "Curiosity", "pageprops": {"disambiguation": ""}, "extract": "Curiosity may refer to:"}}}})
            if title == "Curiosity":
                return Reply(data={"query": {"pages": {"7": {"title": "Curiosity", "extract": CURIOSITY_ARTICLE}}}})
            return Reply(data={"query": {"pages": {"-1": {"title": title, "missing": ""}}}})
        if url.endswith("pg_catalog.csv.gz"):
            return Reply(content=gzip.compress(CATALOG_CSV.encode()))
        if "/cache/epub/" in url:
            return Reply(content=BOOK.encode())
        if "arxiv.org" in url:
            if self.arxiv_status != 200:
                return Reply(status=self.arxiv_status, headers={"Retry-After": "1000"})
            return Reply(content=ARXIV_FEED.encode())
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


def test_nothing_is_acquired_twice_and_quotas_hold_for_an_hour(tmp_path: Path):
    now = [0.0]
    http = Http("test-agent", min_interval=0, sleep=lambda s: None, session=FakeSession(FakeInternet()))
    lib = Librarian(LibrarianConfig(enabled=True, max_books_per_hour=1), tmp_path / "library", http=http, clock=lambda: now[0])
    assert len(lib.acquire(ReadingWish(books=["Hobbes Leviathan"]), reason="r")) == 1
    assert lib.acquire(ReadingWish(books=["Augustine Confessions"]), reason="r") == []  # this hour's quota is used
    now[0] = 3601.0  # an hour later the quota is free again
    assert lib.acquire(ReadingWish(books=["Hobbes Leviathan"]), reason="r") == []  # but this one is already owned
    assert lib.owned == ["book: Hobbes Leviathan"]
    assert len(lib.acquire(ReadingWish(books=["Augustine Confessions"]), reason="r")) == 1


def test_a_busy_source_is_reported_as_busy_not_as_nothing_found(tmp_path: Path):
    lib = make_librarian(tmp_path, FakeInternet(scholar_status=429, arxiv_status=503))
    got = lib.acquire(ReadingWish(topics=["Curiosity"], papers=["information gap curiosity"]), reason="r")
    assert [a.kind for a in got] == ["encyclopedia"]
    assert lib.busy == ["paper: information gap curiosity (api.semanticscholar.org, export.arxiv.org)"] and lib.missed == []


def test_when_semantic_scholar_is_busy_arxiv_is_asked(tmp_path: Path):
    lib = make_librarian(tmp_path, FakeInternet(scholar_status=429))
    got = lib.acquire(ReadingWish(papers=["curiosity prediction exploration"]), reason="r")
    assert [a.title for a in got] == ["Curiosity-driven exploration by self-supervised prediction"]
    meta, body = strip_front_matter(Path(got[0].path).read_text(encoding="utf-8"))
    assert meta["cite_as"] == "PAPER" and "arXiv" in meta["license"] and meta["year"] == "2017"
    assert "Deepak Pathak" in body


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

    busy: list[str] = []
    rejected: list[str] = []
    failed_searches: list[str] = []

    def acquire(self, wish, *, reason, question_id=None, heartbeat=None, context="", until_year=None, approve=None):
        self.wishes.append(wish)
        self.contexts = getattr(self, "contexts", []) + [context]
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
    assert wish.papers == ["information gap theory curiosity review reinterpretation"]  # keywords, not an invented title
    assert "wonder" in org.librarian.contexts[0].lower()  # what a found text must be about
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


CONTEXT = "When does curiosity turn into a vice, a restless hunger for novelty? curiosity wonder doubt inquiry knowledge attention"


def test_a_text_that_only_shares_a_name_with_the_topic_is_not_kept(tmp_path: Path):
    lib = make_librarian(tmp_path, FakeInternet(rover=True))
    assert lib.acquire(ReadingWish(topics=["Curiosity"]), reason="r", context=CONTEXT) == []
    assert lib.rejected == ["Wikipedia: Curiosity (found 'Curiosity (rover)')"]
    kept = make_librarian(tmp_path / "other", FakeInternet())
    assert len(kept.acquire(ReadingWish(topics=["Curiosity"]), reason="r", context=CONTEXT)) == 1  # the real article


def test_a_first_name_cannot_pull_in_a_strangers_book(tmp_path: Path):
    lib = make_librarian(tmp_path, FakeInternet())
    assert lib.gutenberg.search("John Locke Diary", strict=True) == []
    assert lib.gutenberg.search("Manningham Diary", strict=True)[0].ebook_id == "44336"


def test_a_search_that_found_nothing_is_not_repeated(tmp_path: Path):
    internet = FakeInternet()
    lib = make_librarian(tmp_path, internet)
    assert lib.acquire(ReadingWish(topics=["Zeigarnik effect"]), reason="r") == []
    assert lib.missed == ["Wikipedia: Zeigarnik effect"]
    asked = len(internet.requests)
    assert lib.acquire(ReadingWish(topics=["zeigarnik  Effect"]), reason="r") == []
    assert len(internet.requests) == asked and lib.missed == []


def test_a_time_split_reads_only_older_papers_and_no_encyclopedia(tmp_path: Path):
    internet = FakeInternet()
    lib = make_librarian(tmp_path, internet)
    got = lib.acquire(ReadingWish(topics=["Curiosity"], papers=["information gap curiosity"]), reason="r", until_year=2020)
    assert [a.kind for a in got] == ["paper"]
    assert not any("wikipedia" in url for url in internet.requests)
    scholar = next(p for url, p in zip(internet.requests, internet.params) if "semanticscholar" in url)
    assert scholar["year"] == "-2020"


def test_arxiv_entries_are_parsed():
    from curiosity_ai.organism.librarian import parse_arxiv

    papers = parse_arxiv(ARXIV_FEED.encode())
    assert papers[0]["url"] == "http://arxiv.org/abs/1703.00001v1" and papers[0]["year"] == "2017"
    assert papers[0]["authors"] == [{"name": "Deepak Pathak"}]
    assert parse_arxiv(b"not xml") == []



def test_after_fetching_texts_it_returns_to_the_question(config):
    config.organism.librarian.hunger_informativeness = 0.9
    config.organism.seed_questions.append("Is doubt the engine of inquiry or its enemy?")
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    org.librarian = FakeLibrarian(org.library_dir)
    first = org.heartbeat()
    assert first.acquisitions and org.state.questions[first.question_id].news
    second = org.heartbeat()
    assert second.question_id == first.question_id  # it goes back to read what it fetched
    assert not org.state.questions[first.question_id].news



def test_the_judge_can_refuse_a_text_that_only_shares_words(tmp_path: Path):
    lib = make_librarian(tmp_path, FakeInternet())
    seen = []

    def approve(title, beginning, kind):
        seen.append((title, kind))
        return False

    assert lib.acquire(ReadingWish(topics=["Curiosity"], papers=["information gap curiosity"]), reason="r", context=CONTEXT, approve=approve) == []
    # The word check runs first; the judge is asked only about texts that pass it (the arXiv paper did not).
    assert seen == [("Wikipedia: Curiosity", "encyclopedia"), ("The psychology of curiosity: A review and reinterpretation", "paper")]
    assert sum("the judge found not useful" in r for r in lib.rejected) == 2


def test_the_organism_lets_its_judge_approve_what_it_fetches(config):
    config.organism.librarian.hunger_informativeness = 0.9
    llm = ScriptedLLM(text_rating=0)
    org = CuriosityOrganism(config, llm=llm)
    http = Http("test-agent", min_interval=0, sleep=lambda s: None, session=FakeSession(FakeInternet()))
    org.librarian = Librarian(config.organism.librarian, org.library_dir, http=http)
    ep = org.heartbeat()
    assert ep.acquisitions == [] and ep.library_rejected
    assert any("A text found in a search" in prompt for step, prompt in llm.history if step == "JUDGE")


def test_background_texts_pass_and_unrelated_ones_do_not(config):
    """Encyclopedia background rated 1 (loosely related) is kept; a paper must be rated 2 (useful); 0 is refused."""
    config.organism.librarian.hunger_informativeness = 0.9
    for rating, kinds in ((3, {"encyclopedia", "paper"}), (1, {"encyclopedia"}), (0, set())):
        config.organism.home = config.organism.home + f"-{rating}"
        org = CuriosityOrganism(config, llm=ScriptedLLM(text_rating=rating))
        http = Http("test-agent", min_interval=0, sleep=lambda s: None, session=FakeSession(FakeInternet()))
        org.librarian = Librarian(config.organism.librarian, org.library_dir, http=http)
        ep = org.heartbeat()
        got = {"encyclopedia" if a.startswith("Wikipedia") else "paper" for a in ep.acquisitions}
        assert got <= kinds and (bool(got) == bool(kinds)), (rating, ep.acquisitions)


@pytest.mark.parametrize("title, elsewhere", [
    ("Whistler wave propagation through the ionosphere of Venus", True),
    ("Resonant Inverse Compton Scattering and Hard X-ray Emission in Magnetar Magnetospheres", True),
    ("Runaway electron interactions with whistler waves in tokamak plasmas", True),
    ("Modeling whistler-mode waves with electrons in the solar wind inside 0.3 AU", True),
    ("Co-existence of Whistler Waves with Kinetic Alfven Wave Turbulence for the High-beta Solar Wind Plasma", True),
    ("Solar wind driving of plasmaspheric erosion", False),
    ("Overview of Saturn lightning observations", True),
    ("VLF transmitters as tools for monitoring the plasmasphere", False),
    ("Specularly reflected whistler: A low-latitude channel to couple lightning energy to the magnetosphere", False),
])
def test_a_text_about_another_world_star_or_device_is_off_topic(title, elsewhere):
    context = "How does lightning illuminate the inner magnetosphere? whistlers plasmapause radiation belt Earth"
    assert bool(Librarian._elsewhere(title, context)) is elsewhere


def test_a_topic_about_that_place_keeps_it():
    assert Librarian._elsewhere("Lightning on Venus", "How common is lightning on Venus?") == ""


def test_the_persons_own_work_found_online_is_not_fetched_as_a_paper(tmp_path: Path):
    lib = make_librarian(tmp_path, FakeInternet())
    lib.own_beginnings = ["[page 1]\nA reconstruction method of electron density distribution in the equatorial region of "
                          "magnetosphere\nV. V. Shastun, O. V. Agapitov"]
    assert lib._is_own_work("A reconstruction method of electron density distribution in the equatorial region of magnetosphere")
    assert not lib._is_own_work("The source regions of whistlers")


def test_a_refusal_names_the_place():
    context = "How does lightning illuminate the inner magnetosphere? whistlers"
    place = Librarian._elsewhere("Electron properties and the whistler heat-flux instability in the solar wind", context)
    assert place == "in the solar wind"

