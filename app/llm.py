"""Every model call goes through here. Switch provider by changing MODEL in .env.

MODEL uses LiteLLM's "provider/model" format, e.g.
  meta/muse-spark-1.3           (MODEL_API_KEY) Meta Model API, OpenAI-compatible
  openai/<model>                (OPENAI_API_KEY)
  gemini/<model>                (GEMINI_API_KEY)
  openrouter/<vendor>/<model>   (OPENROUTER_API_KEY) - one key, most providers
  mock                          no key needed; echoes back, for UI work and demos offline
Full list: https://docs.litellm.ai/docs/providers
"""

import asyncio
import os
from collections.abc import AsyncIterator

import litellm

litellm.drop_params = True  # silently drop params a provider doesn't support

META_API_BASE = "https://api.meta.ai/v1"

# Spend guard: hard stop after this many model calls per server process, so a
# looping agent can't burn the credits. Raise it in .env when you mean to.
MAX_CALLS = int(os.getenv("MAX_LLM_CALLS", "500"))
_calls = 0


def current_model() -> str:
    return os.getenv("MODEL", "mock")


def _route(model: str) -> dict:
    """Map our MODEL name to litellm kwargs. meta/<id> goes to the Meta Model API."""
    global _calls
    _calls += 1
    if _calls > MAX_CALLS:
        raise RuntimeError(f"MAX_LLM_CALLS={MAX_CALLS} reached; restart the server or raise the limit")
    if model.startswith("meta/"):
        return {
            "model": "openai/" + model.removeprefix("meta/"),
            "api_base": META_API_BASE,
            "api_key": os.getenv("MODEL_API_KEY"),
        }
    return {"model": model}


def _mock(messages: list[dict]) -> str:
    last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return f"[mock model] You said: {last}. Set MODEL in .env to use a real provider."


async def complete(messages: list[dict], **kwargs) -> str:
    """One-shot completion. kwargs pass straight to litellm (tools, response_format, ...)."""
    model = current_model()
    if model == "mock":
        return _mock(messages)
    resp = await litellm.acompletion(messages=messages, **_route(model), **kwargs)
    return resp.choices[0].message.content or ""


async def stream(messages: list[dict], **kwargs) -> AsyncIterator[str]:
    """Yield text chunks as they arrive."""
    model = current_model()
    if model == "mock":
        for word in _mock(messages).split(" "):
            yield word + " "
            await asyncio.sleep(0.03)
        return
    resp = await litellm.acompletion(messages=messages, stream=True, **_route(model), **kwargs)
    async for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
