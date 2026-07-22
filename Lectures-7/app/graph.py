"""
Echo Graph
1) Define State
2) Write a node function (Takes state and return state)
3) Write in stategraph


"""

from langgraph.graph import END,START,StateGraph
from app.state import State

def echo(state:State) -> dict:
    """
    Takes state and uppercases it sends back

    it changes the state and gives
    
    """

    return {"reply":state["message"].upper()}

#Building the graph Start to edge (echo) then edge to end

_builder = StateGraph(State)
_builder.add_node("echo",echo)
_builder.add_edge(START,"echo")
_builder.add_edge("echo",END)

graph = _builder.compile()