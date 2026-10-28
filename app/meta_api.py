"""Gateway for structured Meta Model API calls: retries, a spend cap and a response cache.

- Retries follow the API's error guide: 429, 500 and 503 are retried with exponential backoff and jitter, honouring
  Retry-After; 504 is not (the same request would time out again). An intermittent 404 `model_not_found` is retried
  too: it was seen on the first day and the same call succeeded seconds later.
- Every answer is cached under a hash of its request, so reruns (tests, benchmarks, the demo) replay it for free.
  MODEL_MODE=replay forbids network calls: a cache miss is an error. Use it offline and in CI.
- USD_CAP stops a process that has spent that much (default $2), so a loop can't drain the credits. Every paid call is
  appended to the spend ledger.

Plain httpx rather than LiteLLM, because LiteLLM's `drop_params` can silently strip `response_format`.
"""

import base64
import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.meta.ai/v1"
SPARK = "muse-spark-1.3"
SAM = "sam-3.1"
# USD per million tokens: input, cached input, output (reasoning is billed as output). dev.meta.ai pricing, 29 Oct 2026.
PRICE_PER_M = {SPARK: (1.25, 0.15, 4.25)}
SAM_USD_PER_IMAGE = 2.50 / 1000  # SAM is billed per image, not per token
RETRY_STATUS = {429, 500, 503}
MAX_ATTEMPTS = 4
TIMEOUT_S = 120.0
MAX_WAIT_S = 60.0

CACHE_DIR = Path(os.getenv("DATA_DIR", "data")) / "cache" / "meta"
LEDGER = CACHE_DIR / "spend.jsonl"


class ModelCallError(RuntimeError):
    pass


class SpendCapReached(ModelCallError):
    pass


@dataclass(frozen=True)
class Answer:
    data: dict  # the parsed JSON the model returned
    usd: float  # what the original call cost; replaying it from the cache costs nothing
    cached: bool
    model: str


@dataclass(frozen=True)
class Segment:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in the image's pixels
    confidence: float


_spent_usd = 0.0
_spend_lock = threading.Lock()


def spark_json(messages: list[dict], schema: dict, name: str, effort: str = "low", max_tokens: int = 6000) -> Answer:
    """Ask Muse Spark for JSON that matches `schema` (strict structured output) and return it parsed."""
    body = {
        "model": SPARK,
        "messages": messages,
        "reasoning_effort": effort,  # set explicitly: reasoning tokens are billed as output
        "max_tokens": max_tokens,  # covers reasoning and the visible answer together
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
    }
    record = _cached_or_call("/chat/completions", body)
    choice = record["response"]["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ModelCallError("answer cut off at max_tokens; raise it or lower the reasoning effort")
    try:
        data = json.loads(choice["message"]["content"])
    except (TypeError, json.JSONDecodeError) as e:
        raise ModelCallError(f"answer was not JSON: {e}") from e
    return Answer(data, record["usd"], record["cached"], SPARK)


def sam_find(image_jpeg: bytes, concept: str) -> tuple[list[Segment], float, bool]:
    """Every instance of `concept` SAM 3.1 finds in the image, with its confidence; also what the call cost and whether
    it came from the cache. Masks are requested in the compact encoding and ignored: boxes are enough to search."""
    body = {
        "model": SAM,
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": concept},
            {"type": "input_image", "image_url": jpeg_data_url(image_jpeg)}]}],
    }
    record = _cached_or_call("/responses", body)
    text = "".join(part.get("text", "") for item in record["response"].get("output", [])
                   for part in item.get("content", []))
    return [_segment(fields) for fields in re.findall(r"<\|box;([^|]*)\|>", text)], record["usd"], record["cached"]


def _segment(fields: str) -> Segment:
    """One box record of SAM's output grammar, e.g. `x1=12;y1=40;x2=30;y2=71;w=1000;h=1000;c=0.83`."""
    v = dict(pair.split("=", 1) for pair in fields.split(";"))
    return Segment((float(v["x1"]), float(v["y1"]), float(v["x2"]), float(v["y2"])), float(v.get("c", 1.0)))


def jpeg_data_url(image_jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()


def spent_usd() -> float:
    """Money spent by this process so far (cache hits are free)."""
    return _spent_usd


def _cached_or_call(path: str, body: dict) -> dict:
    key = hashlib.sha256(json.dumps({"path": path, **body}, sort_keys=True).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return {**json.loads(cache_file.read_text(encoding="utf-8")), "cached": True}
    if os.getenv("MODEL_MODE", "live") == "replay":
        raise ModelCallError(f"MODEL_MODE=replay and no cached answer for request {key[:12]}")
    response = _post(path, body)
    usd = _cost(body["model"], response.get("usage") or {})
    _charge(usd, body["model"], key)
    record = {"request_sha256": key, "usd": usd, "response": response}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return {**record, "cached": False}


def _post(path: str, body: dict) -> dict:
    key = os.getenv("MODEL_API_KEY")
    if not key:
        raise ModelCallError("MODEL_API_KEY is not set")
    cap = float(os.getenv("USD_CAP", "2"))
    if _spent_usd >= cap:
        raise SpendCapReached(f"spent ${_spent_usd:.2f} of the USD_CAP ${cap:.2f} for this process")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = httpx.post(BASE_URL + path, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT_S)
        except httpx.TransportError as e:  # connection reset, DNS, read timeout
            if attempt == MAX_ATTEMPTS:
                raise ModelCallError(f"network error after {attempt} attempts: {e!r}") from e
            time.sleep(_backoff(attempt, None))
            continue
        if r.status_code == 200:
            return r.json()
        if attempt < MAX_ATTEMPTS and _retryable(r):
            time.sleep(_backoff(attempt, r.headers.get("retry-after")))
            continue
        raise ModelCallError(f"HTTP {r.status_code}: {r.text[:300]}")
    raise AssertionError("unreachable")


def _retryable(r: httpx.Response) -> bool:
    return r.status_code in RETRY_STATUS or (r.status_code == 404 and "model_not_found" in r.text)


def _backoff(attempt: int, retry_after: str | None) -> float:
    """Retry-After if the server sent one, else 1, 2, 4 ... seconds, plus up to 1 s of jitter; never above a minute."""
    if retry_after and retry_after.replace(".", "", 1).isdigit():
        return min(float(retry_after), MAX_WAIT_S)
    return min(2 ** (attempt - 1) + random.random(), MAX_WAIT_S)


def _cost(model: str, usage: dict) -> float:
    if model == SAM:
        return usage.get("images_processed", 1) * SAM_USD_PER_IMAGE
    price_in, price_cached, price_out = PRICE_PER_M[model]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    fresh = usage.get("prompt_tokens", 0) - cached
    return (fresh * price_in + cached * price_cached + usage.get("completion_tokens", 0) * price_out) / 1e6


def _charge(usd: float, model: str, key: str) -> None:
    global _spent_usd
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _spend_lock, LEDGER.open("a", encoding="utf-8") as f:
        _spent_usd += usd
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "usd": round(usd, 5),
                            "request_sha256": key}) + "\n")
