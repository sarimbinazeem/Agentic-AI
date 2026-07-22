"""
State is a shared note book

It is shared by every ndoe, it is updated and vieewed by nodes.

it also stores provider and persona name

"""

from typing import TypedDict,Literal

# Allowed persona values. Add new personas here and in app/personas.py.
Persona = Literal["resume", "services", "personal"]



class State(TypedDict):
    """
    we get one message and one reply

    Provider chooses which LLM to run

    Persona choses the system prompt 

    classify fills persona
    generate fills reply
    """
    #user message
    message:str

    #bot reply what the llm will generate
    reply: str

    provider: Literal["claude", "gpt", "free"]

    persona: Persona

# Provider → human label for logging.
PROVIDER_LABEL = {
    "claude": "Anthropic",
    "gpt": "FreeLLMAPI (gpt)",
    "free": "FreeLLMAPI (auto)",
}