"""
Reply Graph
1) Define State
2) Write a node function (Takes state and return state)
3) Write in stategraph

It also picks what provider it want to generate answer 

"""

from langgraph.graph import END,START,StateGraph
from app.state import State
from app.openAI import chat as open_chat
from app.llm import chat as free_chat

SYSTEM_PROMPT = (
    "You are a helpful WhatsApp assistant. "
    "Reply concisely (1–3 sentences). "
    "Mirror the user's language. "
    "If the user writes in English, reply in English. "
    "If they write in Urdu/Hindi, reply in the same script."
)

_GPT_MODEL = "gpt-oss-120b"

def generate(state:State) -> dict:
    """
    it picks llm from state["provider"] 
    Takes message and replies with LLM

    Provider values (set by main.py based on the message prefix):
      - "claude" → Anthropic Messages API via the configured proxy
      - "gpt"    → FreeLLMAPI, pinned to a GPT-style model
      - "free"   → FreeLLMAPI router (auto-picks the best available)
    """
    user_msg = state["message"]

    provider = state.get("provider", "free")

    if provider == "gpt":
        text = free_chat(SYSTEM_PROMPT, user_msg, model=_GPT_MODEL)

    elif provider == "openai":
        text= open_chat(SYSTEM_PROMPT,user_msg)
    
    else:
        text = free_chat(SYSTEM_PROMPT, user_msg)

    return {"reply": text}

#Building the graph Start to edge (echo) then edge to end

_builder = StateGraph(State)
_builder.add_node("generate", generate)
_builder.add_edge(START, "generate")
_builder.add_edge("generate", END)

graph = _builder.compile()