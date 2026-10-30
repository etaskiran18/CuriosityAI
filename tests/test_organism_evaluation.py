"""Evaluation tools for the records architecture: forking a life, blind rating sheets, the model without the loop."""
from __future__ import annotations

import csv
import json

import pytest

from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.fork import fork, latest_deferred
from curiosity_ai.organism.organism import _parse_record
from curiosity_ai.organism.plain import plain_records
from curiosity_ai.organism.rating import SCORES, collect, make_sheet, score
from curiosity_ai.organism.state import load_mind

from .organism_fakes import RECORD, ScriptedLLM


@pytest.fixture
def records(config):
    config.organism.curiosity.architecture = "records"
    return config


def maker(**kwargs):
    return lambda cfg: CuriosityOrganism(cfg, llm=ScriptedLLM(**kwargs))


def _with_a_waiting_curiosity(config) -> tuple[CuriosityOrganism, str]:
    org = CuriosityOrganism(config, llm=ScriptedLLM(recognize="no"))
    org.heartbeat()
    q = org.state.add_question("Which feeling marks a philosopher?", trigger="gap", importance=0.5)
    org._apply_record(q, dict(_parse_record(RECORD), unknown="the feeling that marks a philosopher and with which philosophy begins"))
    org._set_status(q, "deferred", "I want to come back to it later")
    org.save()
    return org, q.id


def test_forking_lives_the_same_moment_both_ways(records, tmp_path):
    org, qid = _with_a_waiting_curiosity(records)
    assert latest_deferred(org) == qid
    before = org.state.heartbeat
    got, branches, verdict = fork(records, org.home, tmp_path / "fork", heartbeats=1, make_organism=maker())
    assert got == qid
    waited, pursued = branches
    assert pursued.name == "pursued" and pursued.satisfied and pursued.looks == 1
    assert waited.name == "as decided" and waited.looks == 0
    assert verdict  # one of the four verdicts
    assert (tmp_path / "fork" / "fork_result.md").read_text(encoding="utf-8").startswith(f"# What if it had pursued {qid}?")
    assert load_mind(org.mind_path).heartbeat == before  # the life itself is untouched


def test_a_life_without_a_deferred_curiosity_cannot_be_forked(records, tmp_path):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    with pytest.raises(ValueError):
        fork(records, org.home, tmp_path / "fork", heartbeats=1, make_organism=maker())


def _run(tmp_path, name: str, questions: list[tuple[str, str]]):
    home = tmp_path / name
    home.mkdir()
    qs = {
        f"Q{i}": {"id": f"Q{i}", "text": text, "trigger": "gap" if i > 1 else "seed", "unknown": unknown,
                  "known": ["a fact"], "answer_form": "a text saying it" if unknown else "", "interest": "it matters",
                  "decision_reason": "my texts speak of it"}
        for i, (text, unknown) in enumerate(questions, start=1)
    }
    (home / "mind.json").write_text(json.dumps({"questions": qs}), encoding="utf-8")
    return home


def test_a_blind_sheet_hides_where_each_question_came_from(tmp_path):
    a = _run(tmp_path, "a", [("Seed question?", "x"), ("What makes wonder last?", "the thing that makes wonder last"),
                             ("Why does doubt end?", "the cause that ends doubt")])
    b = _run(tmp_path, "b", [("Seed question?", ""), ("Is curiosity a vice?", ""), ("What is boredom?", "the nature of boredom")])
    assert [i["question"] for i in collect(a)] == ["What makes wonder last?", "Why does doubt end?"]  # seeds are left out

    n = make_sheet({"records": a, "drive": b}, tmp_path / "sheet.csv", tmp_path / "key.csv", n=5, level=1)
    assert n == 4
    rows = list(csv.reader((tmp_path / "sheet.csv").open(encoding="utf-8")))
    assert rows[0] == ["item", "question", "intellectual_curiosity", "worth_asking", "notes"]
    assert "records" not in (tmp_path / "sheet.csv").read_text(encoding="utf-8")
    key = {row["item"]: row["run"] for row in csv.DictReader((tmp_path / "key.csv").open(encoding="utf-8"))}
    assert sorted(key.values()) == ["drive", "drive", "records", "records"]

    assert make_sheet({"records": a, "drive": b}, tmp_path / "s2.csv", tmp_path / "k2.csv", n=5, level=2) == 3  # records only
    header = next(csv.reader((tmp_path / "s2.csv").open(encoding="utf-8")))
    assert header[:7] == ["item", "question", "unknown", "known", "answer_form", "interest", "reason"] and set(SCORES) <= set(header)


def test_filled_sheets_are_scored_by_run_with_the_raters_agreement(tmp_path):
    a = _run(tmp_path, "a", [("Seed?", ""), ("What makes wonder last?", "w"), ("Why does doubt end?", "d")])
    b = _run(tmp_path, "b", [("Seed?", ""), ("Is curiosity a vice?", ""), ("What is boredom?", "")])
    make_sheet({"records": a, "drive": b}, tmp_path / "sheet.csv", tmp_path / "key.csv", n=5, level=1)
    key = {row["item"]: row["run"] for row in csv.DictReader((tmp_path / "key.csv").open(encoding="utf-8"))}

    def fill(path, flip: bool):
        rows = list(csv.DictReader((tmp_path / "sheet.csv").open(encoding="utf-8")))
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                if row["item"].isdigit():
                    yes = key[row["item"]] == "records"
                    row["intellectual_curiosity"] = "yes" if yes != (flip and row["item"] == "1") else "no"
                    row["worth_asking"] = "yes"
                writer.writerow(row)

    fill(tmp_path / "r1.csv", flip=False)
    fill(tmp_path / "r2.csv", flip=True)
    result = score([tmp_path / "r1.csv", tmp_path / "r2.csv"], tmp_path / "key.csv")
    assert result["runs"]["drive"]["worth_asking"] == 1.0
    assert result["runs"]["records"]["intellectual_curiosity"] > result["runs"]["drive"]["intellectual_curiosity"]
    assert result["agreement"]["items_both_rated"] == 4 and result["agreement"]["percent_agreement"] == 0.75


def test_the_model_without_the_loop_reads_the_same_library(records):
    org = CuriosityOrganism(records, llm=ScriptedLLM())
    items = plain_records(org, n=3)
    assert len(items) == 3
    assert all(i["question"] and i["unknown"] == RECORD["unknown"] and i["source"] for i in items)
    assert org.state.heartbeat == 0  # it did not live
