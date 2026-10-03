from __future__ import annotations

import json
import time
from typing import Any
import requests
from pydantic import BaseModel

from .config import LLMConfig
from .utils import extract_json_object


class OllamaClient:
    """Small Ollama chat client for local models.

    Uses the local /api/chat endpoint and expects Ollama to be running.
    """

    attempts = 3  # a server error or a dropped connection is tried again, twice
    wait_seconds = 5.0

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature if temperature is None else temperature,
                "top_p": self.config.top_p,
                "num_ctx": self.config.num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        for attempt in range(self.attempts):
            last = attempt == self.attempts - 1
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
            except requests.ConnectionError:
                if last:
                    raise
            else:
                # Ollama answers 500 when the process that runs the model died (out of memory, for example) and
                # starts it again by itself: a second try a few seconds later usually works.
                if resp.status_code < 500 or last:
                    break
            time.sleep(self.wait_seconds * (attempt + 1))
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def json_chat(
        self,
        system: str,
        user: str,
        schema_hint: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        forced_user = (
            user
            + "\n\nReturn ONLY valid JSON. Do not include markdown. JSON schema hint:\n"
            + schema_hint
        )
        text = self.chat(system, forced_user, temperature=temperature, json_mode=True, max_tokens=max_tokens)
        try:
            return extract_json_object(text)
        except Exception:
            # Retry without Ollama's JSON mode, sometimes smaller local models do better with explicit repair.
            repair_prompt = (
                "The previous response was not valid JSON. Convert it into valid JSON only. "
                "Do not add explanations.\n\nPrevious response:\n"
                + text
                + "\n\nSchema hint:\n"
                + schema_hint
            )
            repaired = self.chat(
                "You repair invalid JSON into valid JSON only.",
                repair_prompt,
                temperature=0.0,
                json_mode=True,
                max_tokens=max_tokens,
            )
            return extract_json_object(repaired)


def as_model(model_cls: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    return model_cls.model_validate(data)
