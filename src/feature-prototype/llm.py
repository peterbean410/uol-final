"""Optional local-LLM narration for the advisor chatbot (Template 4.2).

Turns the structured recommendation into plain prose and answers free-form
questions, grounded in the *real* recommendation so the model cannot invent
signals. Talks to any **OpenAI-compatible** chat endpoint, e.g. the project's
served Gemma-4 vLLM (`scripts/port-forward-gemma4.sh` -> localhost:8080) or a
local Ollama (`ollama serve`, base ``http://localhost:11434/v1``).

Config (env or sibling ``.env``); when ``LLM_BASE_URL`` is unset the chatbot
degrades gracefully to the structured reply:

    LLM_BASE_URL   e.g. http://localhost:8080/v1
    LLM_MODEL      e.g. google/gemma-4-31b-it   (or  llama3.2  for Ollama)
    LLM_API_KEY    optional (vLLM/Ollama accept any value; default "EMPTY")
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a concise USD/JPY trading-advisor assistant. Answer the user using "
    "ONLY the recommendation below, do not invent prices, signals, or numbers. "
    "If the user asks something the recommendation does not cover, say so briefly. "
    "Two or three sentences. Always end with: 'Educational/research only, not "
    "financial advice.'"
)


def base_url() -> str | None:
    url = os.environ.get("LLM_BASE_URL", "").strip()
    return url.rstrip("/") or None


def available() -> bool:
    return base_url() is not None


def _chat(messages: list[dict], *, temperature: float = 0.3, timeout: int = 45) -> str | None:
    url = base_url()
    if not url:
        return None
    model = os.environ.get("LLM_MODEL", "").strip() or "google/gemma-4-31b-it"
    key = os.environ.get("LLM_API_KEY", "").strip() or "EMPTY"
    try:
        r = requests.post(
            f"{url}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": 300},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001 - graceful fallback to structured reply
        logger.warning("LLM call failed (%s); falling back to structured reply", exc)
        return None


def narrate(recommendation: str, user_message: str | None = None) -> str | None:
    """Return an LLM answer grounded in ``recommendation``, or ``None`` if no LLM."""
    if not available():
        return None
    question = (user_message or "").strip() or "Explain the current recommendation in plain language."
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Recommendation:\n{recommendation}\n\nUser: {question}"},
    ]
    return _chat(messages)
