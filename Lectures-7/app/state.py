"""
State is a shared note book

It is shared by every ndoe, it is updated and vieewed by nodes.

it also stores provider and persona name

Every iteration will take this file
"""

from typing import TypedDict,Literal,Any

# Allowed persona values. Add new personas here and in app/personas.py.
Persona = Literal["resume", "services", "personal"]

# A single message in the conversation. We use plain dicts (not
# LangChain's BaseMessage) so the schema stays stdlib-only.
ChatMessage = dict[str, Any]  # {"role": "user"|"assistant", "content": str}


class State(TypedDict):
    """
    we get one message and one reply

    Provider chooses which LLM to run

    Persona choses the system prompt 
     Memory
    stores the conversation history.

    classify fills persona
    generate fills reply
    """
    #user message
    message:str

    #bot reply what the llm will generate
    reply: str

    provider: Literal["claude", "gpt", "free"]

    persona: Persona

    #Full conversation history is store in dict
    messages: list[ChatMessage]

# Provider → human label for logging.
PROVIDER_LABEL = {
    "claude": "Anthropic",
    "gpt": "FreeLLMAPI (gpt)",
    "free": "FreeLLMAPI (auto)",
}