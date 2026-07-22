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

we save conversational memory trhough SqliteSaver chekpointer. 

Each chat have a session_id the bot is reminded through that ID

we append the reply to previous chat again and agan

we pass the session Id so the checkpointer know which conversation to load


"""

import logging
import os
from langgraph.graph import END,START,StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.state import Persona,State
from app.openAI import chat as open_chat
from app.llm import chat as free_chat, _get_client as free_client
from app.personas import CLASSIFY_PROMPT, DEFAULT_PERSONA, PERSONAS

log = logging.getLogger("app.graph")
_GPT_MODEL = "gpt-oss-120b"

#to check for automatic model
_CLASSIFY_MODEL = "auto"

# Where the SQLite checkpointer stores conversation state. One DB for
# the whole app — thread_id inside the DB separates conversations.
_CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", "data/checkpoints.sqlite")


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
    It also reply according to message history

    it answers the prompt according to the persona specified

    Provider values (set by main.py based on the message prefix):
      - "claude" → Anthropic Messages API via the configured proxy
      - "gpt"    → FreeLLMAPI, pinned to a GPT-style model
      - "free"   → FreeLLMAPI router (auto-picks the best available)

      we append the replies to state['messages'] so that we can keep it up with the history
    """

    persona: Persona = state.get("persona", DEFAULT_PERSONA)
    system_prompt = PERSONAS.get(persona, PERSONAS[DEFAULT_PERSONA])

    user_msg = state["message"]

    provider = state.get("provider", "free")

    #get the messages
    history: list[dict] = state.get("messages", []) or []

    # Build the full message list: system, prior turns, current user.
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(history) #put the hsitory before the new message
    msgs.append({"role": "user", "content": user_msg})

    if provider == "gpt":
        text = _openai_multi_turn(msgs, model=_GPT_MODEL)
    else:
        text = _free_multi_turn(msgs, model="auto")

    new_history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": text},
    ]

    return {"reply": text, "messages": new_history}


def _free_multi_turn(msgs: list[dict], model: str |None) -> str:
    """OpenAI-style multi-turn call using FreeLLM."""
    resp = free_client().chat.completions.create(
        model=model or "auto",
        messages=msgs,
    )
    return (resp.choices[0].message.content or "").strip()

def _openai_multi_turn(msgs: list[dict], model: str | None = None) -> str:
    """OpenAI multi-turn conversation."""

    from app.openAI import _get_client as openai_client

    resp = openai_client().chat.completions.create(
        model=model or "gpt-4o",
        messages=msgs,
    )

    return (resp.choices[0].message.content or "").strip()

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

"""
We build the graph using AsyncSplitSaver

from normal .compiler() the graph was created EVREYTIME it was invoked

AsncSplitSaver creates a graph immediately when fast api starts

"""

import os as _os
_dir = _os.path.dirname(_CHECKPOINT_DB)
if _dir:
    _os.makedirs(_dir, exist_ok=True)

#temporality storing nothing in grpah until Fast API strts
graph = None 

async def build_graph():
    """
    this calls the FAST API and bulds the graph


    Open the async checkpointer and compile the graph.

    Caller owns the returned object's lifetime — the AsyncSqliteSaver
    inside stays bound to the event loop that compiled it.
    """

    saver_cm = AsyncSqliteSaver.from_conn_string(_CHECKPOINT_DB) #for database

    saver = await saver_cm.__aenter__()

    #after every node it saves
    compiled = _builder.compile(checkpointer=saver)
    # Hold the context manager alive so __aexit__ never runs while
    # the graph is in use. 
    compiled._saver_cm = saver_cm  # type: ignore[attr-defined]
    return compiled    