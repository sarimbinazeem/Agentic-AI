"""
Mock_llm gives a sample of llms replies
step 1
"""

from __future__ import annotations #for Effective Hints
import random #to give randon replies from list of mock llms

#Mock llms list that is meant only for internal use only (Leading Underscorre)
_MOCK_REPLIES = [
    "Understood. What Do You Want Me To Do?",
    "Interesting. Please tell me everything in detail.",
    "I am an agent only capable to communicate. I currently do not have any tools with me yet.",
    "Noted. What is your next question?",
    "Alright, Please tell me more.",
]

def chat(messages: list[dict], system: str="") ->str:
    """
    here Messages is a list of dictionaries 
    system is a string for system prompt (That is not used for mock_llm but for real_llms)
    
    """
    
    #We get what last user said from the messages list
    last_user =""
    #We doing reverse because we want to see what the user said LAST!
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user=m.get("content","") #it returns "" if we dont find any content (it is just for error handling)
            break #break when we retrieve last message from the user
        
    #Now we generate answer according to the message.
    
    #If the message is not empty then execute this
    if not last_user.strip():
        reply = "What Do You Want To Reply?"
        
    #If the message is NOT A QUESTION
    elif "?" not in last_user:
        reply = f"'{last_user}' — got it. Aur kuch?"
        
    #otherwise return a random reply
    else:
        reply = random.choice(_MOCK_REPLIES)
        
    return reply