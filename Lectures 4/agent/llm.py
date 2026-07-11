"""

Step 3
llm.py acts as a middle layer between the agent and the LLM.

Today it forwards requests to mock_llm.
Later we can replace mock_llm with OpenAI, Claude, Gemini, etc.

The rest of the project always calls llm.chat(),
so only this file needs to change when switching LLMs.
"""

from __future__ import annotations

from agent import mock_llm
from agent.memory import Message

def chat(messages: list[Message],system:str="")->str:
    return mock_llm.chat(messages,system = system)

