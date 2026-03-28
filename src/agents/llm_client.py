"""LLM client utility shared across agents."""
from __future__ import annotations

import os
from typing import Any


def get_llm_client():
    """Return a configured OpenAI client (or compatible endpoint)."""
    from openai import OpenAI

    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def chat_complete(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    """Call the LLM and return the response content string."""
    client = get_llm_client()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
