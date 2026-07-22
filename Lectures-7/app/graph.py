"""
Reply Graph
1) Define State
2) Write a node function (Takes state and return state)
3) Write in stategraph


"""

from langgraph.graph import END,START,StateGraph
from app.state import State
from app.llm import chat

SYSTEM_PROMPT = (
    "You are a helpful WhatsApp assistant. "
    "Reply concisely (1–3 sentences). "
    "Mirror the user's language. "
    "If the user writes in English, reply in English. "
    "If they write in Urdu/Hindi, reply in the same script."
)



def generate(state:State) -> dict:
    """
    Takes message and replies with LLM
    """

    text = chat(SYSTEM_PROMPT,state["message"])
    return {"reply": text}

#Building the graph Start to edge (echo) then edge to end

_builder = StateGraph(State)
_builder.add_node("generate", generate)
_builder.add_edge(START, "generate")
_builder.add_edge("generate", END)

graph = _builder.compile()