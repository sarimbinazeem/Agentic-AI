import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DO_API_KEY"),
    base_url=os.getenv("DO_BASE_URL"),
)

persona = "Your a rapper that raps in coding genre only."  # change this!
history = [
    {
        "role": "system", 
        "content": persona
    }
]

print(f"Persona: {persona}\nType 'quit' to exit.\n")

def chat(history:list)-> None:
    while True:
        prompt = input("What Do You Want To Ask Today?")
        
        if prompt.lower().strip() == "quit":
            break
        
        if prompt.strip() == "":
            continue
        
        history.append(
            {
                "role":"user",
                "content":prompt,
            }      
        )
        
        response = client.chat.completions.create(
            model = os.getenv("MODEL"),
            
            messages= history,
            
            stream=True
        )
        
        assistant_message=""
        
        for chunk in response:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta.content or ""
            print(delta,end="",flush=True)
            
            assistant_message += delta
            
        print("\n")
        
        history.append(
            {
                "role":"assistant",
                "content":assistant_message,
            }
        )
            

chat(history)
    
