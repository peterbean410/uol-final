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

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a USD/JPY trading-advisor assistant for a system that pairs a "
    "reinforcement-learning policy with a probabilistic forecaster. Use this model "
    "of how it works to explain decisions accurately:\n"
    "- The forecaster predicts the next 5-minute move as a mean 'mu' (expected "
    "return, in basis points) and a standard deviation 'sigma' (its uncertainty). "
    "Larger sigma = less confident = a 'high-uncertainty regime'.\n"
    "- An uncertainty screen sits in front of the policy: in high-uncertainty "
    "regimes it can veto a trade (forcing HOLD) or throttle exposure via a daily "
    "risk budget, and it can also veto a trade whose direction conflicts with mu. "
    "A screen reason of 'pass' means the trade was allowed; 'budget_exhausted' or "
    "'directional_conflict' means it was held back.\n"
    "- A profitability gate continuously checks whether the screen has been "
    "earning its keep; when 'active' the screen is trusted, when 'bypassed' the "
    "policy trades unscreened.\n"
    "- Actions: HOLD (no new position), BUY/SELL (open long/short).\n\n"
    "Answer the user's question by explaining the CURRENT recommendation in these "
    "terms, why this action follows from the mu, sigma, regime, screen reason and "
    "gate state shown, and what would change it. Use ONLY the numbers and the "
    "decision in the recommendation below; never invent prices, levels or figures. "
    "If the recommendation does not cover what is asked, say so. Be substantive but "
    "not rambling (about 3-5 sentences). Always end with exactly: "
    "'Educational/research only, not financial advice.'"
)


def base_url() -> str | None:
    url = os.environ.get("LLM_BASE_URL", "").strip()
    return url.rstrip("/") or None


def available() -> bool:
    return base_url() is not None


def _post(messages: list[dict], *, tools: list[dict] | None = None,
          temperature: float = 0.3, timeout: int = 45) -> dict | None:
    """POST one chat completion; return the assistant `message` object (which may
    carry `tool_calls`), or None on failure/no-LLM."""
    url = base_url()
    if not url:
        return None
    model = os.environ.get("LLM_MODEL", "").strip() or "google/gemma-4-31b-it"
    key = os.environ.get("LLM_API_KEY", "").strip() or "EMPTY"
    payload: dict = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": 450}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    try:
        r = requests.post(
            f"{url}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]
    except Exception as exc:  # noqa: BLE001 - graceful fallback to structured reply
        logger.warning("LLM call failed (%s); falling back to structured reply", exc)
        return None


def _chat(messages: list[dict], **kw) -> str | None:
    msg = _post(messages, **kw)
    if not msg:
        return None
    return (msg.get("content") or "").strip() or None


_TOOL_HINT = (
    "\n\nYou can also call the `send_price_chart` tool to show the user a chart of "
    "recent USD/JPY price movement. Call it whenever they want to SEE the price, a "
    "chart/graph, the trend, or how USD/JPY is moving; after it is sent, add a short "
    "sentence of context. Do not describe candles/levels you cannot see, just send the chart."
)


def narrate(recommendation: str, user_message: str | None = None,
            tools: list[dict] | None = None, execute_tool=None) -> str | None:
    """Return an LLM answer grounded in ``recommendation``, or ``None`` if no LLM.

    If ``tools`` (OpenAI function specs) and ``execute_tool(name, args) -> str`` are
    given, the model may emit tool calls; each is executed and its result fed back
    until the model produces a final text answer (bounded loop).
    """
    if not available():
        return None
    question = (user_message or "").strip() or "Explain the current recommendation in plain language."
    system = _SYSTEM + (_TOOL_HINT if tools else "")
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Recommendation:\n{recommendation}\n\nUser: {question}"},
    ]
    if not tools:
        return _chat(messages)

    for _ in range(3):  # bounded tool-call loop
        msg = _post(messages, tools=tools)
        if msg is None:
            return None
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return (msg.get("content") or "").strip() or None
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            result = "tool executor unavailable"
            if execute_tool is not None:
                try:
                    result = execute_tool(name, args)
                except Exception as exc:  # noqa: BLE001 - a tool error must not kill the reply
                    logger.exception("tool %s failed", name)
                    result = f"tool {name} failed: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": str(result)})
    return _chat(messages)  # final text summary after tool use
