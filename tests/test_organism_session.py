from __future__ import annotations

import json

import pytest

from curiosity_ai.config import BodyConfig
from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.body import Body
from curiosity_ai.organism.state import Visit

from .organism_fakes import ScriptedLLM


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def timed_organism(config, llm: ScriptedLLM, *, seconds_per_thought: float = 10.0, **body) -> tuple[CuriosityOrganism, FakeTime]:
    """An organism whose thinking takes fake time, so time limits can be tested instantly."""
    t = FakeTime()
    llm.on_call = lambda step: t.sleep(seconds_per_thought)
    body_cfg = BodyConfig(enabled=bool(body), **body) if body else BodyConfig(enabled=False)
    org = CuriosityOrganism(config, llm=llm, body=Body(body_cfg, sleep=t.sleep, clock=t.clock))
    return org, t


def test_a_session_writes_a_report_metrics_and_config(config):
    config.organism.reflect_every = 2
    org = CuriosityOrganism(config, llm=ScriptedLLM(contradict_for_real=True))
    report = org.run_session(3, label="first try")
    assert report.directory.name.endswith("-first-try")
    assert {p.name for p in report.directory.iterdir()} == {"report.md", "metrics.json", "config.json", "judge_check.csv"}
    assert (org.home / "research_map.md").exists()
    m = json.loads((report.directory / "metrics.json").read_text(encoding="utf-8"))
    assert m["heartbeats"] == 3 and m["policy"] == "curiosity" and m["web"] is False
    assert m["beliefs_grounded"] <= m["beliefs_new"]
    assert 0.0 <= m["fabrication_rate"] <= 1.0 and m["quotes_rejected"] > 0
    assert m["diagnoses"] == {"healthy_wonder": 1}
    config_snapshot = json.loads((report.directory / "config.json").read_text(encoding="utf-8"))
    assert config_snapshot["organism"]["policy"] == "curiosity"
    text = report.markdown
    assert "## Summary" in text and "## Heartbeat by heartbeat" in text and "## Its theory of curiosity" in text


def test_a_time_limit_stops_living(config):
    org, t = timed_organism(config, ScriptedLLM(), seconds_per_thought=10)  # about a minute per heartbeat
    report = org.run_session(minutes=3)
    assert 2 <= report.metrics["heartbeats"] <= 4
    assert t.now <= 4 * 60


def test_rests_count_toward_the_time_limit(config):
    org, t = timed_organism(config, ScriptedLLM(), seconds_per_thought=10, breath_seconds=60, work_minutes=0, rest_minutes=0,
                            gpu_temperature_guard=False, pause_on_battery=False)
    report = org.run_session(minutes=5)
    assert report.metrics["heartbeats"] < 5  # each heartbeat is about a minute plus a one-minute breath


def test_the_report_is_written_even_when_interrupted(config):
    llm = ScriptedLLM()
    org = CuriosityOrganism(config, llm=llm)

    def stop_on_second_settle(step):
        if step == "SETTLE" and llm.calls.count("SETTLE") == 2:
            raise KeyboardInterrupt

    llm.on_call = stop_on_second_settle
    with pytest.raises(KeyboardInterrupt):
        org.run_session(forever=True)
    assert org.last_report is not None
    assert org.last_report.metrics["heartbeats"] == 1
    assert (org.last_report.directory / "report.md").exists()


def _organism_with_three_questions(config, policy: str) -> CuriosityOrganism:
    config.organism.policy = policy
    config.organism.seed_questions = [
        "Plato says philosophy begins in wonder: what exactly is wonder, and is it the same thing as curiosity?",
        "Is doubt the engine of inquiry or its enemy, according to Peirce and Descartes?",
        "When does curiosity become a vice, according to Augustine and Heidegger?",
    ]
    org = CuriosityOrganism(config, llm=ScriptedLLM())
    q1, q2 = org.state.questions["Q1"], org.state.questions["Q2"]
    for q, n in ((q1, 3), (q2, 1)):  # Q3 is the least visited
        q.visits = [Visit(heartbeat=i, confidence_before=0.3, confidence_after=0.3, prediction_error=0.5, informativeness=0.5) for i in range(n)]
    return org


def test_novelty_policy_picks_the_least_visited_question(config):
    org = _organism_with_three_questions(config, "novelty")
    ep = org.heartbeat()
    assert ep.question_id == "Q3" and ep.policy == "novelty"


def test_random_policy_is_reproducible_with_a_seed(config):
    picks = []
    for _ in range(2):
        org = _organism_with_three_questions(config, "random")
        picks.append(org._choose(org._read_drives()).question_id)
        CuriosityOrganism.archive(config)
    assert picks[0] == picks[1]


def test_it_does_not_start_a_heartbeat_it_cannot_finish(config):
    # Nine thoughts per heartbeat (predict, compare, judge, three voices, settle, judge the beliefs,
    # rate the new question): 90 s each.
    org, t = timed_organism(config, ScriptedLLM(), seconds_per_thought=10)
    report = org.run_session(minutes=3.25)  # 195 s: room for two heartbeats, not a third
    assert report.metrics["heartbeats"] == 2
    assert t.now <= 195


def test_no_brier_score_without_an_addressed_prediction(config):
    org = CuriosityOrganism(config, llm=ScriptedLLM(judge="neither"))
    report = org.run_session(1)
    assert report.metrics["predictions_confirmed"] == 0 and report.metrics["mean_brier"] is None
    assert "no Brier score yet" in report.markdown
