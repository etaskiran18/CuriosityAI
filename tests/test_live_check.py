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
