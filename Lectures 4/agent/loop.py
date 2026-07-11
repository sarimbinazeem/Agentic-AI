"""
step 4
loop.py act as REPL (Read Evaluate Print Loop)
"""

from __future__ import annotations
import sys
from typing import Callable #For Type Testing

from agent import llm 
from agent import memory as mem #just created an alias for memory

#wE MAKE SYSTEM PROMPTS
SYSTEM_PROMPT = (
    "You are a helpful agent. "
    "Reply briefly in the user's language. "
    "If tools are available, use them. If not, just chat."
)

#Main function that does all the tasks
def run(input_fn: Callable[[str],str] = input,  output_fn:Callable[[str],None] = print):
    
    #We put default input function into input_fn variable works like input function that TAKES STR and RETURNS STR
    #We put default input function into output_fn variable works like print function that TAKES STR and RETURNS NOTHING
    
    #Initilazing History with system prompt
    history =  mem.init_history(SYSTEM_PROMPT)
    
    output_fn("╔════════════════════════════════════════════╗")
    output_fn("║   Demo Harness — Checkpoint 1             ║")
    output_fn("║   (Mock LLM active — no API calls yet)    ║")
    output_fn("║   Type 'quit' or 'exit' to leave.         ║")
    output_fn("╚════════════════════════════════════════════╝")
    output_fn("")
    
    #Inifnite looping
    
    while True:
        #user input
        try:
            user_prompt = input_fn("What Do You Want To Ask: ")
        except (EOFError,KeyboardInterrupt):
            #If there is EOF error OR Ctrl + C or Ctrl+D we stop the program
            output_fn("\n Bye.")
            return
        
        user_prompt = user_prompt.strip()
        #If User Doesnt Write Anything
        if not user_prompt: 
            continue
        
        #If the user write quit
        if user_prompt.lower() in ("quit", "exit", "q"):
            output_fn("Bye!")
            return
            
        #Updating Memory
        mem.add_user(history,user_prompt)
        
        #LLM thinks
        try:
            reply=llm.chat(history)
        except Exception as e:
            output_fn(f"[Agent error] {e}")
            continue
        
        #Save assitant reply
        mem.add_assistant(history,reply)
        
        #Printing
        output_fn(f"Agent: {reply}")
        sys.stdout.flush()