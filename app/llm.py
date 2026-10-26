"""Every model call goes through here. Switch provider by changing MODEL in .env.

MODEL uses LiteLLM's "provider/model" format, e.g.
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


def current_model() -> str:
    return os.getenv("MODEL", "mock")


def _mock(messages: list[dict]) -> str:
    last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return f"[mock model] You said: {last}. Set MODEL in .env to use a real provider."


async def complete(messages: list[dict], **kwargs) -> str:
    """One-shot completion. kwargs pass straight to litellm (tools, response_format, ...)."""
    model = current_model()
    if model == "mock":
        return _mock(messages)
    resp = await litellm.acompletion(model=model, messages=messages, **kwargs)
    return resp.choices[0].message.content or ""


async def stream(messages: list[dict], **kwargs) -> AsyncIterator[str]:
    """Yield text chunks as they arrive."""
    model = current_model()
    if model == "mock":
        for word in _mock(messages).split(" "):
            yield word + " "
            await asyncio.sleep(0.03)
        return
    resp = await litellm.acompletion(model=model, messages=messages, stream=True, **kwargs)
    async for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
