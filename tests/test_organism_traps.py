"""Trap tests (curiosity_ai/organism/traps.py), with the scripted model: the traps are set and judged correctly."""
from __future__ import annotations

import random

from curiosity_ai.organism import CuriosityOrganism
from curiosity_ai.organism.traps import NOISE_AUTHOR, run_trap, scramble, write_results
from curiosity_ai.organism.textutil import token_set

from .organism_fakes import RECORD, ScriptedLLM


def maker(**kwargs):
    return lambda cfg: CuriosityOrganism(cfg, llm=ScriptedLLM(**kwargs))


def test_scrambled_text_keeps_its_words_but_not_their_order():
    text = "Wonder is the feeling of a philosopher, and philosophy begins in wonder. He was not a bad genealogist."
    noise = scramble(text, random.Random(1))
    assert token_set(noise) == token_set(text) and noise != text


def test_the_noise_trap_puts_scrambled_texts_in_the_library_and_counts_what_came_of_them(config, tmp_path):
    result = run_trap("noise", config, tmp_path / "noise", heartbeats=2, make_organism=maker(recognize="no"))
    assert result.trap == "noise" and result.passed in (True, False, None)
    assert set(result.details) >= {"noise_reads", "real_reads", "curiosities_from_noise", "curiosities_from_real_texts"}
    assert list((tmp_path / "noise" / "library").glob("noise_*.md"))
    assert f"author: {NOISE_AUTHOR}" in (tmp_path / "noise" / "library" / "noise_1.md").read_text(encoding="utf-8")


def test_the_unreachable_trap_fails_when_it_believes_a_text_answered_the_question(config, tmp_path):
    result = run_trap("unreachable", config, tmp_path / "u", heartbeats=1, make_organism=maker())
    assert result.passed is False and "believed a text answered it" in result.evidence


def test_the_unreachable_trap_passes_when_it_stops_looking(config, tmp_path):
    config.organism.curiosity.pursuit_limit = 1
    result = run_trap("unreachable", config, tmp_path / "u", heartbeats=2, make_organism=maker(recognize="no"))
    assert result.passed is True and "it stopped after" in result.evidence


def test_the_hidden_answer_is_recognized_while_it_reads_for_another_question(config, tmp_path):
    result = run_trap("hidden", config, tmp_path / "h", heartbeats=1, make_organism=maker())
    assert result.passed is True, result.evidence
    assert "while it was pursuing something else" in result.evidence and "Iris" in result.evidence


def test_the_old_answer_is_not_taken_up_again(config, tmp_path):
    result = run_trap("old", config, tmp_path / "o", heartbeats=1, after_heartbeats=2, make_organism=maker())
    assert result.passed is True, result.evidence


def test_a_trap_can_be_set_in_a_copy_of_an_existing_life(config, tmp_path):
    config.organism.curiosity.architecture = "records"
    original = CuriosityOrganism(config, llm=ScriptedLLM(recognize="no"))
    original.heartbeat()
    copy_home = tmp_path / "copy"
    result = run_trap("unreachable", config, copy_home, heartbeats=1, from_home=original.home, make_organism=maker(recognize="no"))
    assert result.trap == "unreachable"
    assert (copy_home / "diary.md").read_text(encoding="utf-8").startswith((original.home / "diary.md").read_text(encoding="utf-8")[:40])
    assert original.state.heartbeat == 1  # the original life is untouched


def test_results_are_written_as_a_table(tmp_path):
    from curiosity_ai.organism.traps import TrapResult

    md = write_results([TrapResult("noise", True, "ok"), TrapResult("old", None, "no curiosity ended")], tmp_path / "trap_results.md")
    text = md.read_text(encoding="utf-8")
    assert "| noise | passed | ok |" in text and "| old | not walked into |" in text
    assert (tmp_path / "trap_results.json").exists()
