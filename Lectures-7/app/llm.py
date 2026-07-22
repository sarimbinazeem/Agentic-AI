"""
We create chat function

this helps us to chat with any OpenAI- compatible provider 

"""

from __future__ import annotations
from functools import lru_cache #it creates a function one time and reuses it so that there isnt waste of resources
from openai import OpenAI
from app.core.config import get_settings #this gives the configuration file instead of os.getenv()


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """
    Build OpenAI client 
    when loading .env
    """

    settings = get_settings()
    if not settings.OPEN_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is unset. Add it to .env (and restart uvicorn).")

    return OpenAI(base_url=settings.OPENAI_BASE_URL, api_key=settings.OPENAI_API_KEY)

def chat(system: str, user: str, model: str | None = None) -> str:
    """
    Single turn chat 

    it automatically picks up model.
    we can pass a specific model to stop it from automatically picking it
    
    """

    settings = get_settings()
    chosen = model or settings.OPENAI_MODEL

    response = _get_client().chat.completions.create(
        model = chosen,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},            
        ]

    )
    return (response.choices[0].message.content or "").strip()