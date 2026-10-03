from __future__ import annotations

import re
from pathlib import Path

from curiosity_ai.organism.senses import LexicalLibrary, Senses, library_body, not_prose, without_number_lines, write_inbox_item
from curiosity_ai.organism.textutil import coverage, jaccard, stem, verify_quote
from curiosity_ai.schema import WebResult

SOURCE = """SOCRATES: I see, my dear Theaetetus, that Theodorus had a true insight into
your nature when he said that you were a philosopher, for wonder is the feeling
of a philosopher, and philosophy begins in wonder."""


def test_verify_quote_accepts_real_quotes_regardless_of_case_and_line_breaks():
    assert verify_quote("wonder is the feeling of a philosopher", SOURCE)
    assert verify_quote("WONDER is the feeling of a Philosopher, and philosophy begins in wonder", SOURCE)
    assert verify_quote("Theodorus had a true insight ... philosophy begins in wonder", SOURCE)


def test_long_quotes_are_kept_short_but_still_verifiable():
    long_quote = " ".join(SOURCE.split()[:45])
    kept = verify_quote(long_quote, SOURCE, keep_words=10)
    assert kept.endswith(" ...") and len(kept.split()) == 11
    assert verify_quote(kept[:-4], SOURCE)


def test_verify_quote_rejects_fabrications_paraphrases_and_fragments():
    assert verify_quote("curiosity is the mother of all science and of every art", SOURCE) is None
    assert verify_quote("Socrates says that wonder is how philosophers feel", SOURCE) is None
    assert verify_quote("in wonder", SOURCE) is None
    assert verify_quote("", SOURCE) is None


def test_stemming_unifies_old_and_new_spellings():
    assert stem("enquire") == stem("inquiry") == stem("inquiring")
    assert stem("curious") == stem("curiosity")
    assert jaccard("How will you enquire into that which you do not know?", "how can one inquire into what one does not know") > 0.5


def test_library_body_strips_front_matter_and_gutenberg_boilerplate(corpus_dir: Path):
    raw = (corpus_dir / "theaetetus.md").read_text(encoding="utf-8")
    meta, body, offset = library_body(raw)
    assert meta["author"] == "Plato"
    assert "license" not in body
    assert "END OF THE PROJECT GUTENBERG" not in body
    assert "philosophy begins in wonder" in body
    assert offset > 0


def test_citation_line_numbers_point_into_the_real_file(corpus_dir: Path):
    library = LexicalLibrary([corpus_dir])
    [obs] = library.search("wonder philosopher Theodorus", k=1)
    start = int(re.search(r":L(\d+)-L(\d+)\]", obs.citation).group(1))
    file_lines = (corpus_dir / "theaetetus.md").read_text(encoding="utf-8").split("\n")
    first_words = obs.text.split()[:4]
    window = " ".join(file_lines[start - 1 : start + 2])
    assert all(w in window for w in first_words)


def test_bm25_ranks_the_relevant_text_first_and_ignores_unrelated_text(corpus_dir: Path):
    library = LexicalLibrary([corpus_dir])
    assert library.search("questions reason cannot decline nor answer", k=1)[0].author == "Immanuel Kant"
    assert library.search("curiosity becomes intellectual through problems", k=1)[0].author == "John Dewey"
    assert all(o.author != "A Cook" for o in library.search("wonder and curiosity", k=5))


def test_inbox_material_is_indexed_when_it_appears(tmp_path: Path, corpus_dir: Path):
    inbox = tmp_path / "inbox"
    library = LexicalLibrary([corpus_dir, inbox])
    assert len(library) > 0
    write_inbox_item(inbox, "Boredom is not the absence of curiosity but its frustrated form, a restless wish for something worth attending to.", "On boredom", "2026-10-02T12:00:00")
    assert library.refresh() == 1
    hits = library.search("boredom restless frustrated", k=1)
    assert hits[0].kind == "inbox"
    assert hits[0].citation.startswith("[INBOX:")


class FakeWeb:
    def search(self, query: str) -> list[WebResult]:
        return [
            WebResult(web_id="1", title="Wonder and curiosity in Plato's Theaetetus", url="u1", content="Philosophy begins in wonder; curiosity and wonder in Plato.", citation="[WEB:T1]"),
            WebResult(web_id="2", title="Moral rights under the Berne Convention", url="u2", content="Copyright law and the right of integrity of authors.", citation="[WEB:CR1]"),
            WebResult(web_id="3", title="Crossref search failed", url="", content="timeout", citation="[WEB:CR_ERROR]", raw={"provider": "error"}),
        ]


def test_web_results_must_be_about_the_question(corpus_dir: Path):
    senses = Senses(LexicalLibrary([corpus_dir]), web=FakeWeb(), passages=2)
    found = senses.observe("Is wonder in Plato the same as curiosity?")
    web = [o for o in found if o.kind == "web"]
    assert [o.citation for o in web] == ["[WEB:T1]"]
    assert any("Berne" in r for r in senses.rejected_web)
    assert [o.label for o in found] == [f"S{i}" for i in range(1, len(found) + 1)]


def test_coverage_measures_how_much_of_the_question_a_text_touches():
    q = "Is curiosity epistemically dangerous according to Kant?"
    assert coverage(q, "Kant warns that curiosity can be dangerous to reason") > 0.5
    assert coverage(q, "Copyright law and the Berne Convention") == 0.0


# -- reference lists and figures are not read ----------------------------------------------------

PROSE = (
    "Whistlers propagate along ducts of enhanced density (Smith, 1961; Inan et al., 1995). Their dispersion "
    "carries the electron density along the path, and recent work with sferics has focused on the delayed "
    "precipitation of electrons (Rodger et al., 2007; Clilverd et al., 2009) and on how the plasmapause "
    "shapes the paths that the waves can take from one hemisphere to the other."
)
REFERENCES = (
    "Nunn, D. and Smith, A.J.: 1996, 'Numerical simulation of whistler-triggered VLF emissions observed in "
    "Antarctica', J. Geophys. Res., 101, 5261-5277.\n"
    "Abel, B. and Thorne, R.M.: 1998, 'Electron scattering loss in Earth's inner magnetosphere 1: Dominant "
    "physical processes', J. Geophys. Res., 103, 2385-2395.\n"
    "Albert, J. M., N. P. Meredith, and R. B. Horne (2009), Three-dimensional diffusion simulation of outer "
    "radiation belt electrons, J. Geophys. Res., 114, A09214.\n"
    "[4] M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. The arcade learning environment. Journal of "
    "Artificial Intelligence Research, 47:253-279, 2013.\n"
)


def test_a_reference_list_is_not_prose_but_a_text_full_of_citations_is():
    assert not_prose(REFERENCES)
    assert not not_prose(PROSE)
    chronology = ("The intervals are 243, 483, 502, 486, 491 years. B.C. 585, Eclipse of Thales. A.D. 30, The "
                  "crucifixion. A.D. 529, Closing of Athenian schools. A.D. 1125, Rise of the Universities. A.D. 1543, "
                  "Publication of the De Revolutionibus of Copernicus. From these figures no conclusion can be drawn.")
    assert not not_prose(chronology)  # "A.D." and "B.C." are no authors' initials
    styles = "Schaul, Tom, Saxton, David, and Munos, Remi. Unifying count-based exploration. In NIPS, 2016. " * 3
    assert not not_prose(styles) and not_prose(styles, after_references=True)


def test_number_lines_of_a_figure_are_left_out_but_short_sentences_stay():
    text = "Figure 9 shows the source lightning.\n-3\n 0.5\n\u00d7 1000 km\n10\nDunedin - Source Lightning\nIt is.\nSo it is."
    assert without_number_lines(text).split("\n") == [
        "Figure 9 shows the source lightning.", "", "Dunedin - Source Lightning", "It is.", "So it is."]


def test_the_library_never_offers_a_reference_list(tmp_path: Path):
    """A 7B judge accepted a reference ("Nunn, D. and Smith, A.J.: 1996, ...") as a confirmation."""
    papers = tmp_path / "papers"
    papers.mkdir()
    authors = "V. Shastun, O. Agapitov, A. B. Smith, C. D. Jones, E. F. Brown, G. H. White, I. J. Green, K. L. Black"
    body = "\n\n".join([
        f"Electron density from whistler waves\n{authors}\nAbstract\n{PROSE}",
        *[PROSE.replace("Whistlers", f"In section {i}, whistlers") for i in range(2, 8)],
        "In conclusion, the method recovers the density of the plasmasphere from the ratio of the wave fields.",
        "References\n" + REFERENCES * 4,
        "Appendix A. Implementation details\n" + "The density solver iterates until the wave fields agree with "
        "the model, which takes a few seconds for each whistler that the station records. " * 6,
    ])
    (papers / "paper.md").write_text(f"---\ntitle: Electron density from whistler waves\n---\n{body}\n", encoding="utf-8")
    library = LexicalLibrary([(papers, "papers")], chunk_chars=600, overlap_chars=100)
    assert library.refresh() > 5
    texts = [c.text for c in library._chunks]
    assert not any("Thorne, R.M." in t or "Bellemare" in t for t in texts)
    assert any("In conclusion, the method recovers" in t for t in texts)  # the text before the heading stays
    assert any("The density solver iterates" in t for t in texts)  # and the appendix after the list
    assert authors in texts[0]  # a title page full of initials is the start of the paper, not a reference list
    found = library.search("numerical simulation whistler-triggered VLF emissions Antarctica scattering loss", k=5)
    assert all("Nunn, D." not in o.text for o in found)
