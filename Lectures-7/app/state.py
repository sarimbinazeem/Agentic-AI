"""
State is a shared note book

It is shared by every ndoe, it is updated and vieewed by nodes.


"""

from typing import TypedDict,Literal

class State(TypedDict):
    """
    we get one message and one reply

    Provider chooses which LLM to run
    """
    #user message
    message:str

    #bot reply what the llm will generate
    reply: str

    provider: Literal["claude", "gpt", "free"]

# Provider → human label for logging.
PROVIDER_LABEL = {
    "claude": "Anthropic",
    "gpt": "FreeLLMAPI (gpt)",
    "free": "FreeLLMAPI (auto)",
}