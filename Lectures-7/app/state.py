"""
State is a shared note book

It is shared by every ndoe, it is updated and vieewed by nodes.


"""

from typing import TypedDict

class State(TypedDict):
    #user message
    message:str

    #bot reply
    reply: str

