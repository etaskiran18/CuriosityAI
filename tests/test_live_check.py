from __future__ import annotations

import live


class Reply:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


def ps(monkeypatch, models):
    monkeypatch.setattr(live.requests, "get", lambda url, timeout=5: Reply({"models": models}))


def test_gpu_share_reads_ollamas_report(monkeypatch):
    ps(monkeypatch, [{"name": "mistral:7b-instruct", "size": 100, "size_vram": 100}])
    assert live._gpu_share("http://x", "mistral:7b-instruct") == ("ok", "GPU use", "Ollama reports 100% of the model on the GPU")
    ps(monkeypatch, [{"name": "mistral:7b-instruct", "size": 100, "size_vram": 40}])
    assert "40% on the GPU" in live._gpu_share("http://x", "mistral:7b-instruct")[2]
    ps(monkeypatch, [{"name": "mistral:7b-instruct", "size": 100, "size_vram": 0}])
    assert "CPU only" in live._gpu_share("http://x", "mistral:7b-instruct")[2]
    ps(monkeypatch, [{"name": "qwen2.5:3b", "size": 100, "size_vram": 100}])
    assert "not loaded" in live._gpu_share("http://x", "mistral:7b-instruct")[2]


def test_model_names_match_with_or_without_a_tag():
    assert live._is_model({"mistral:latest"}, "mistral")
    assert live._is_model({"mistral:7b-instruct"}, "mistral")
    assert live._is_model({"mistral:7b-instruct"}, "mistral:7b-instruct")
    assert not live._is_model({"qwen2.5:3b"}, "mistral:7b-instruct")


import os
from pathlib import Path

import pytest

from curiosity_ai.config import AppConfig
from curiosity_ai.organism.papers import ingest_papers

from .test_organism_research import LINES, minimal_pdf


@pytest.mark.skipif(os.name == "nt", reason="WSL and Linux paths")
@pytest.mark.parametrize("typed, expected", [
    (r"C:\Users\emre\papers\article", "/mnt/c/Users/emre/papers/article"),
    ("'D:/data/papers'", "/mnt/d/data/papers"),
    ("/home/emre/papers", "/home/emre/papers"),
])
def test_windows_style_paths_work_in_wsl(typed, expected):
    assert live.local_path(typed) == Path(expected)


def test_backslashes_in_a_linux_path_are_forgiven(tmp_path):
    (tmp_path / "paper" / "article").mkdir(parents=True)
    typed = str(tmp_path) + r"\paper\article"
    assert live.local_path(typed) == tmp_path / "paper" / "article"


def test_a_missing_papers_folder_gets_a_hint(tmp_path, capsys):
    config = AppConfig()
    config.organism.home = str(tmp_path / "home")
    assert live.ingest(config, str(tmp_path / "nope")) == 1
    out = capsys.readouterr().out
    assert "There is no folder" in out and "wslpath" in out


def test_your_own_article_can_be_shared_as_a_pdf_and_is_not_read_twice(tmp_path):
    folder = tmp_path / "article"
    folder.mkdir()
    (folder / "main.pdf").write_bytes(minimal_pdf(LINES, title="Lightning and the inner magnetosphere"))
    (folder / "other.pdf").write_bytes(minimal_pdf(LINES[::-1], title="Whistler waves"))
    title, text = live.read_shared_file(folder / "main.pdf")
    assert title == "Lightning and the inner magnetosphere" and "[page 1]" in text
    report = ingest_papers(folder, tmp_path / "home" / "papers", skip={folder / "main.pdf"})
    assert report.added == ["other.pdf"] and report.skipped == ["main.pdf"]


class Tags(Reply):
    def raise_for_status(self):
        pass


def test_a_judge_model_that_is_not_installed_stops_the_run_at_once(monkeypatch, config):
    monkeypatch.setattr(live.requests, "get", lambda url, timeout=5: Tags({"models": [{"name": "mistral:7b-instruct"}]}))
    config.llm.model = "mistral:7b-instruct"
    assert live.check_ollama(config)
    config.organism.judge.model = "qwen2.5:14b"
    assert not live.check_ollama(config)  # without its judge nothing would count as evidence
    monkeypatch.setattr(live.requests, "get", lambda url, timeout=5: Tags({"models": [{"name": "mistral:7b-instruct"}, {"name": "qwen2.5:14b"}]}))
    assert live.check_ollama(config)

