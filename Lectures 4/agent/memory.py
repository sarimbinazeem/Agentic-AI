"""
Step 2
To Store Short Term Memory 

By sending history everytime a new llm is called
"""

from __future__ import annotations

from typing import TypedDict #Through this we can type check the data type of keys in a dictionary

#Message class that have role and content only in it.It represents a single message in the conversation
class Message(TypedDict):
    role:str
    content:str
    
#Function to Initialize History with a system prompt FIRST
def init_history(system:str)-> list[Message]:
    return [{"role":"system","content":system}]

#Function append user content in the history
def add_user(history:list[Message],text:str) ->None:
    history.append({"role":"user","content":text})
    
#Function append assistant replies in the history
def add_assistant(history:list[Message],text:str) ->None:
    history.append({"role":"assistant","content":text})

#Helper Function that extracts last messsage from the user
def last_user(messages: list[Message]) ->str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content","")
        
    return ""