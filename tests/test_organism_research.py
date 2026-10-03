"""Researcher mode, the researcher's papers, the outside exam and the research map."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.exam import exam_notes, load_exam, run_exam
from curiosity_ai.organism.papers import clean_pdf_text, ingest_papers, pdf_text
from curiosity_ai.organism.research_map import render_research_map

from .organism_fakes import ScriptedLLM

REPO = Path(__file__).resolve().parent.parent
LINES = [
    "Lithium-ion cells lose capacity as the solid electrolyte interphase grows on the anode.",
    "Each charge cycle consumes a little lithium in side reactions at the interphase.",
    "Fast charging at low temperature causes lithium plating, which speeds up capacity fade.",
    "Calendar aging continues even when the cell rests, faster at high state of charge.",
]


def minimal_pdf(lines: list[str], title: str = "") -> bytes:
    """A one-page PDF with real text, built by hand so the test needs no PDF writer."""
    text_ops = " ".join(f"({line}) Tj T*" for line in lines)
    content = f"BT /F1 10 Tf 72 720 Td 14 TL {text_ops} ET".encode()
    info = f"<< /Title ({title}) >>".encode() if title else None
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ] + ([info] if info else [])
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    trailer_info = b" /Info 6 0 R" if info else b""
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, trailer_info, xref)
    return out


@pytest.fixture
def research_config(config):
    config.organism.research.topic = "How do lithium-ion batteries age?"
    return config


@pytest.fixture
def papers(tmp_path: Path) -> Path:
    src = tmp_path / "my_papers"
    src.mkdir()
    (src / "aging.pdf").write_bytes(minimal_pdf(LINES, title="Mechanisms of battery aging"))
    (src / "notes.md").write_text("---\ntitle: My lab notes\nauthor: Me\n---\n\n" + " ".join(LINES) * 2, encoding="utf-8")
    (src / "scan.pdf").write_bytes(minimal_pdf([]))
    return src


# -- papers --------------------------------------------------------------------------


def test_pdf_text_keeps_page_markers_and_the_title():
    meta, text = pdf_text_from_bytes(minimal_pdf(LINES, title="Mechanisms of battery aging"))
    assert meta["title"] == "Mechanisms of battery aging" and meta["pages"] == "1"
    assert text.startswith("[page 1]") and "solid electrolyte interphase" in text


def pdf_text_from_bytes(data: bytes):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.pdf"
        path.write_bytes(data)
        return pdf_text(path)


def test_broken_words_are_rejoined():
    assert clean_pdf_text("infor-\nmation  gap theory") == "information gap theory"


def test_papers_are_ingested_once_and_scans_are_reported(papers: Path, tmp_path: Path):
    dest = tmp_path / "home" / "papers"
    report = ingest_papers(papers, dest)
    assert sorted(report.added) == ["aging.pdf", "notes.md"]
    assert report.failed == [("scan.pdf", "no text found (a scanned PDF needs OCR first)")]
    again = ingest_papers(papers, dest)
    assert again.added == [] and sorted(again.already) == ["aging.pdf", "notes.md"]
    converted = next(p for p in dest.glob("aging_*.md"))
    assert "source_file:" in converted.read_text(encoding="utf-8")


# -- researcher mode ---------------------------------------------------------------------


def test_a_researcher_is_born_with_its_own_questions_and_key_terms(research_config):
    llm = ScriptedLLM()
    org = CuriosityOrganism(research_config, llm=llm)
    st = org.state
    assert org.research and st.topic.mode == "research" and st.topic.title == "Battery aging"
    assert "capacity fade" in st.topic.keywords
    assert [q.text for q in st.questions.values()][0].startswith("Why does the solid electrolyte interphase")
    assert llm.calls == ["TOPIC"]
    assert "research assistant" in org._sys("ANTICIPATE") and "Battery aging" in org._sys("ANTICIPATE")
    from curiosity_ai.organism import prompts as P

    assert "research assistant" not in P.JUDGE and "curiosity-driven" not in P.JUDGE  # the judge is nobody's inner voice


def test_a_researcher_reads_its_papers_not_the_philosophy_corpus(research_config, papers):
    org = CuriosityOrganism(research_config, llm=ScriptedLLM())
    ingest_papers(papers, org.papers_dir)
    org.senses.notice_new_material()
    hits = org.senses.library.search("solid electrolyte interphase capacity", k=3)
    assert hits and all(h.kind == "papers" for h in hits)
    assert hits[0].citation.startswith("[DOC:")
    assert all(h.kind != "corpus" for h in org.senses.library.search("wonder is the feeling of a philosopher", k=3))


def test_a_researcher_heartbeat_speaks_of_papers(research_config, papers):
    llm = ScriptedLLM()
    org = CuriosityOrganism(research_config, llm=llm)
    ingest_papers(papers, org.papers_dir)
    org.senses.notice_new_material()
    ep = org.heartbeat()
    assert ep is not None and "the papers and notes in your library" in llm.prompts["ANTICIPATE"]
    assert "Main topic: Battery aging" in llm.prompts["JUDGE"]  # new questions are rated against the topic


def test_a_failed_topic_setup_still_gives_a_question(config):
    config.organism.research.topic = "Graphene membranes for desalination"
    org = CuriosityOrganism(config, llm=ScriptedLLM(fail_steps=("TOPIC",)))
    assert [q.text for q in org.state.questions.values()] == [
        "What is known about Graphene membranes for desalination, and what is still unknown?"
    ]
    assert org.state.topic.keywords  # taken from the topic's own words


def test_the_research_map_shows_questions_hypotheses_and_reading(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM(contradict_for_real=True))
    for _ in range(2):
        org.heartbeat()
    text = render_research_map(org)
    for heading in ("# Research map: the philosophy of curiosity", "## Open questions", "## Surprises", "## Possible gaps", "## Reading list"):
        assert heading in text
    assert "I expected" in text  # the scripted contradiction


# -- the exam -------------------------------------------------------------------------------


@pytest.fixture
def exam_file(tmp_path: Path) -> Path:
    path = tmp_path / "exam.json"
    path.write_text(json.dumps({"title": "Mini exam", "questions": [
        {"id": "X1", "question": "What feeling marks the philosopher, according to Plato?", "reference": "Wonder.", "source": "Theaetetus"},
        {"id": "X2", "question": "How long should onions sweat in butter?", "reference": "Gently, before the stock.", "source": "Soups"},
    ]}), encoding="utf-8")
    return path


def test_the_default_exam_is_well_formed():
    title, questions = load_exam(REPO / "data" / "exams" / "philosophy_of_curiosity.json")
    assert len(questions) == 21 and all(q.reference and q.source for q in questions)
    assert len({q.id for q in questions}) == 21


def test_without_notes_nothing_can_be_answered(config, exam_file):
    llm = ScriptedLLM()
    org = CuriosityOrganism(config, llm=llm)
    result = run_exam(org, exam_file, label="before")
    assert result.score == 0.0 and result.answered == 0
    assert "EXAM" not in llm.calls and "GRADE" not in llm.calls  # no notes, no questions to the model
    assert result.path is not None and result.path.exists()


def test_the_exam_counts_what_it_learned_by_reading(config, exam_file):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    org.heartbeat()
    notes = exam_notes(org, "What feeling marks the philosopher, according to Plato?")
    assert notes and notes[0][2]  # its grounded belief about Plato comes first
    result = run_exam(org, exam_file, label="after")
    first = result.answers[0]
    assert first.grade == "correct" and first.grounded and first.notes_used
    assert result.answers[1].grade == "unanswered"
    assert result.score == pytest.approx(0.5) and result.grounded_score == pytest.approx(0.5)


def test_a_session_takes_the_exam_before_and_after(config, exam_file):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    report = org.run_session(2, exam_path=exam_file)
    m = report.metrics
    assert m["exam_before"] == 0.0 and m["exam_after"] == pytest.approx(0.5) and m["exam_gain"] == pytest.approx(0.5)
    assert m["heartbeats"] == 2
    assert "**Exam:** before 0%" in report.markdown
    assert len(list((org.home / "exams").glob("*.json"))) == 2


# -- checking the judge by hand -----------------------------------------------------------------


def test_the_judge_can_be_checked_by_hand(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    report = org.run_session(2)
    path = report.directory / "judge_check.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows and set(rows[0]) >= {"claim", "quote", "organism_said", "judge_said", "your_verdict"}
    for row in rows:
        row["your_verdict"] = row["judge_said"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "judge_agreement.py"), str(path)], capture_output=True, text=True, check=True)
    assert "The judge agreed with you:" in out.stdout and "100%" in out.stdout
    assert report.metrics["judge_pairs"] > 0 and report.metrics["judge_agreement"] is not None
