"""
LLM which is compatible to OpenAI

it can be any provider

"""

from __future__ import annotations

import os
from functools import lru_cache  #so that the fuynction is created once and reused 

from openai import OpenAI


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Build the OpenAI client on first use, when .env has been loaded."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:31415/v1")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is unset. Add it to .env (and restart uvicorn)."
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def chat(system: str, user: str) -> str:
    """Single-turn chat. Returns the model's reply text."""

    model = os.environ.get("OPENAI_MODEL", "auto")
    resp = _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()