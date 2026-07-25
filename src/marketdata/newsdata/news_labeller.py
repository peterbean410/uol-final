"""
Label FX news headlines with BOJ policy / yen intervention flags using an
OpenAI-compatible LLM endpoint (in-cluster Gemma via LiteLLM or vLLM directly).

Usage:
    from marketdata.newsdata.news_labeller import label_headlines, LABEL_COLUMNS

    labels = label_headlines(["USD/JPY slips as BoJ holds rates"], endpoint, model)
    # -> [{"boj_policy": True, "jpy_intervention": False}]

The two flags are deliberately independent. `boj_policy` is about the Bank of
Japan as a monetary authority; `jpy_intervention` is about Japanese authorities
acting on the exchange rate - which is legally the Ministry of Finance's call,
with the BOJ acting only as its agent. Headlines routinely conflate the two, so
a single "is this BOJ" flag forces an arbitrary call on the intervention
cluster; two flags let each headline set both, neither, or one.

Labelling is best-effort: any headline the endpoint cannot label comes back as
None so callers can persist a null rather than failing a data pipeline.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger(__name__)

LABEL_COLUMNS = ("boj_policy", "jpy_intervention")

DEFAULT_MODEL = "gemma-4-31b-it"
DEFAULT_MAX_WORKERS = 16
DEFAULT_TIMEOUT_SECONDS = 90
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1
MAX_TOKENS = 40

SYSTEM_PROMPT = (
    "You label forex news headlines with two independent flags.\n\n"
    "boj_policy = the headline concerns Bank of Japan monetary policy: rate decisions, policy "
    "meetings, minutes, statements, policy tools (YCC, QQE, bond purchases), or the Governor/board "
    "members speaking on policy.\n\n"
    "jpy_intervention = the headline concerns actual or threatened intervention in the yen by "
    "Japanese authorities: MOF or BOJ intervention, 'rate checks', official verbal warnings or "
    "jawboning about currency moves, or explicit speculation about imminent intervention.\n\n"
    "Both can be true. Both can be false. Generic price or technical analysis with only a passing "
    "mention of an upcoming event is false for both.\n\n"
    'Reply with JSON only: {"boj_policy": true|false, "jpy_intervention": true|false}'
)


def _parse_label_response(content: str) -> dict[str, bool]:
    """Parse the model's JSON reply into a flag dict.

    Tolerates markdown code fences, which some models emit despite the
    JSON-only instruction.
    """
    cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    payload = json.loads(cleaned.strip())
    return {column: bool(payload[column]) for column in LABEL_COLUMNS}


def label_headline(
    title: str,
    endpoint: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = MAX_RETRIES,
) -> dict[str, bool] | None:
    """Label a single headline, or return None if it cannot be labelled.

    Args:
        title: The news headline.
        endpoint: Base URL of an OpenAI-compatible API (no trailing /v1).
        model: Model name to request.
        api_key: Bearer token, if the endpoint requires one.
        timeout: Per-request timeout in seconds.
        max_retries: Attempts before giving up on this headline.

    Returns:
        A dict with a bool per LABEL_COLUMNS entry, or None on persistent failure.
    """
    if not title or not title.strip():
        return None

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": title},
        ],
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{endpoint.rstrip('/')}/v1/chat/completions",
                json=body,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _parse_label_response(content)
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(INITIAL_BACKOFF_SECONDS * (2 ** attempt))

    logger.warning("Could not label headline after %d attempts: %s", max_retries, last_error)
    return None


def label_headlines(
    titles: list[str],
    endpoint: str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, bool] | None]:
    """Label many headlines concurrently, preserving input order.

    Duplicate titles are labelled once and the result reused, which matters on
    the first pass over a cumulative snapshot where syndicated headlines repeat.
    """
    if not titles:
        return []

    unique = list(dict.fromkeys(titles))
    logger.info("Labelling %d unique headlines (%d total)", len(unique), len(titles))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(
            pool.map(
                lambda t: label_headline(t, endpoint, model, api_key, timeout),
                unique,
            )
        )

    by_title = dict(zip(unique, results))
    return [by_title[title] for title in titles]
