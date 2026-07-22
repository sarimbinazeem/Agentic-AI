"""OpenAI-compatible LLM client.

Works with:
- OpenAI
- DigitalOcean GenAI
- OpenRouter
- Groq
- Together AI
- Any OpenAI-compatible endpoint

Just change OPENAI_BASE_URL and OPENAI_API_KEY in .env.
"""

from __future__ import annotations

import os
from functools import lru_cache

from openai import OpenAI


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Build the OpenAI client on first use."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is unset. Add it to .env and restart."
        )

    if not base_url:
        raise RuntimeError(
            "OPENAI_BASE_URL is unset. Add it to .env and restart."
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def chat(system: str, user: str) -> str:
    """Single-turn chat. Returns the model's reply text."""

    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        temperature=0.7,
    )

    return (response.choices[0].message.content or "").strip()