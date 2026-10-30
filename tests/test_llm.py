from __future__ import annotations

import pytest
import requests

from curiosity_ai import llm as llm_module
from curiosity_ai.config import LLMConfig
from curiosity_ai.llm import OllamaClient


class Reply:
    def __init__(self, status: int, content: str = ""):
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def json(self):
        return {"message": {"content": self.content}}


def scripted(monkeypatch, replies):
    calls = []

    def post(url, json, timeout):
        calls.append(json["model"])
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm_module.requests, "post", post)
    monkeypatch.setattr(OllamaClient, "wait_seconds", 0.0)
    return calls


def test_a_server_error_is_tried_again(monkeypatch):
    """Ollama answered 500 once when the process running the model was killed; it restarts it by itself."""
    calls = scripted(monkeypatch, [Reply(500), Reply(200, "fine")])
    assert OllamaClient(LLMConfig()).chat("system", "user") == "fine"
    assert len(calls) == 2


def test_a_dropped_connection_is_tried_again(monkeypatch):
    calls = scripted(monkeypatch, [requests.ConnectionError("reset"), Reply(200, "fine")])
    assert OllamaClient(LLMConfig()).chat("system", "user") == "fine"
    assert len(calls) == 2


def test_a_lasting_failure_still_fails(monkeypatch):
    calls = scripted(monkeypatch, [Reply(500), Reply(500), Reply(500)])
    with pytest.raises(requests.HTTPError):
        OllamaClient(LLMConfig()).chat("system", "user")
    assert len(calls) == 3


def test_a_bad_request_and_a_slow_model_are_not_tried_again(monkeypatch):
    calls = scripted(monkeypatch, [Reply(404)])
    with pytest.raises(requests.HTTPError):
        OllamaClient(LLMConfig()).chat("system", "user")  # a model that is not installed: trying again cannot help
    calls = scripted(monkeypatch, [requests.ReadTimeout("slow")])
    with pytest.raises(requests.ReadTimeout):
        OllamaClient(LLMConfig()).chat("system", "user")  # waiting the whole timeout again would only double it
    assert len(calls) == 1
