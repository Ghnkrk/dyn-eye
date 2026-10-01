"""
Single entry point for every TEXT LLM call in DYN-EYE.

All reasoning/advisory calls (training advisor, retrain decision, dynamic
prompt writer) go through Groq with the model configured in
``config.LLM_MODEL_ID``.  The VLM (image → bboxes) is the only component
that talks to Google GenAI and lives in ``vlm_annotation.py``.

Callers get:
  - one lazily-built, cached client (no key needed at import time)
  - reasoning-model tolerance (<think> blocks and code fences are stripped)
  - a small retry on transient errors
  - a single exception type (LLMUnavailable) to catch for "no key / no SDK"
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import config as cfg
from src.utils.logger import get_logger

log = get_logger("llm")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_client: Any = None
_client_lock = threading.Lock()


class LLMUnavailable(RuntimeError):
    """No API key configured or the provider SDK is not installed."""


def llm_available() -> bool:
    """True if a Groq key is configured (does not validate the key)."""
    return bool(cfg.GROQ_API_KEY)


def _get_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            if not cfg.GROQ_API_KEY:
                raise LLMUnavailable("GROQ_API_KEY is not set (add it to .env).")
            try:
                from groq import Groq
            except ImportError as e:  # pragma: no cover
                raise LLMUnavailable("The 'groq' package is not installed.") from e
            _client = Groq(api_key=cfg.GROQ_API_KEY)
    return _client


def clean_reply(text: str | None) -> str:
    """Strip reasoning blocks and markdown fences from a model reply."""
    text = _THINK_RE.sub("", text or "").strip()
    return _FENCE_RE.sub("", text).strip()


def parse_json(text: str | None) -> dict:
    """Parse a JSON object from a reply, tolerating surrounding prose."""
    text = clean_reply(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def chat(
    prompt: str,
    *,
    system: str | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retries: int = 2,
) -> str:
    """Send one prompt to the configured LLM and return the cleaned reply text."""
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": cfg.LLM_MODEL_ID,
        "messages": messages,
        "temperature": cfg.LLM_TEMPERATURE if temperature is None else temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            return clean_reply(response.choices[0].message.content)
        except Exception as e:  # network / rate limit / 5xx
            last_err = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise last_err  # type: ignore[misc]


def chat_json(prompt: str, *, system: str | None = None, **kw) -> dict:
    """Like chat() but returns a parsed JSON object."""
    return parse_json(chat(prompt, system=system, json_mode=True, **kw))
