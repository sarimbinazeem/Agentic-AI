"""
Reply Graph
1) Define State
2) Write a node function (Takes state and return state)
3) Write in stategraph

It also picks what provider it want to generate answer 

It also picks persona 
classify -> generate -> END

classify picks persona
generates generates reply according to provider
"""

import logging
from langgraph.graph import END,START,StateGraph
from app.state import Persona,State
from app.openAI import chat as open_chat
from app.llm import chat as free_chat
from app.personas import CLASSIFY_PROMPT, DEFAULT_PERSONA, PERSONAS

log = logging.getLogger("app.graph")
_GPT_MODEL = "gpt-oss-120b"

#to check for automatic model
_CLASSIFY_MODEL = "auto"

def classify(state:State) -> dict:
    """
    Pick a persona 

    if main.py sets it then pass
    otherwise ask llm to pick from the personas
    we do defauly persona if something unexpected comes
    
    """

    existing = state.get("persona")

    if existing:
        log.info("persona already set by slash prefix: %s", existing)
        return {}

    #ask for persona
    raw = free_chat(
        "You are a routing assistant.",
        CLASSIFY_PROMPT.format(message=state["message"]),
        model=_CLASSIFY_MODEL,
    ).strip().lower()

    if raw in ("resume", "services", "personal"):
        log.info("classify picked persona=%s", raw)
        return {"persona": raw}

    log.warning("classify got %r, defaulting to %s", raw, DEFAULT_PERSONA)
    return {"persona": DEFAULT_PERSONA}
    

def generate(state:State) -> dict:
    """
    it picks llm from state["provider"] 
    Takes message and replies with LLM

    it answers the prompt according to the persona specified

    Provider values (set by main.py based on the message prefix):
      - "claude" → Anthropic Messages API via the configured proxy
      - "gpt"    → FreeLLMAPI, pinned to a GPT-style model
      - "free"   → FreeLLMAPI router (auto-picks the best available)
    """

    persona: Persona = state.get("persona", DEFAULT_PERSONA)
    system_prompt = PERSONAS.get(persona, PERSONAS[DEFAULT_PERSONA])

    user_msg = state["message"]

    provider = state.get("provider", "free")

    if provider == "gpt":
        text = free_chat(system_prompt, user_msg, model=_GPT_MODEL)

    elif provider == "openai":
        text= open_chat(system_prompt,user_msg)
    
    else:
        text = free_chat(system_prompt, user_msg)

    return {"reply": text}

#Building the graph Start to edge (echo) then edge to end

_builder = StateGraph(State)
_builder.add_node("generate", generate)
_builder.add_node("classify", classify)

def _route_from_start(state:State) -> str:
    """
    if main.py set persona via slash then we skip classify
    otherwise we run it so that LLM picks one
    
    """
    #run generate function if persona was set inside main.py otherwise run classify first
    return "generate" if state.get("persona") else "classify"

#we put the function inside the edges and it calls the function according to condition
_builder.add_conditional_edges(START,_route_from_start,{
    "classify": "classify",
    "generate": "generate",
})
_builder.add_edge("classify","generate")
_builder.add_edge("generate",END)

graph = _builder.compile()