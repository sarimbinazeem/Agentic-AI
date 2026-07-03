import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding ="utf-8")
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DO_API_KEY"),
    base_url=os.getenv("DO_BASE_URL"),
)

history = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
]
 
 
def chat(history:list) -> str:
    while True:
        prompt = input("What do you want to ask today: ")

        #if the user enters quit we break the loop 
        if prompt.lower().strip() == "quit":
            break
        
        #if the user accidently enters a blank space we ignore and continue
        if prompt.strip() == "":
            continue
        
        history.append(
                       {
                           "role":"user",
                           "content":prompt
                       })
        
        response = client.chat.completions.create(
            model = os.getenv("MODEL"),
            
            messages=history,
            stream = True,
            
        )
        
        #since the response is on chunks so we apply loop to obtain response in a variable
        assistant_message = ""
    
        for chunk in response:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta.content or ""
            print(delta,end="",flush=True)
            
            assistant_message += delta
            
        #to get in next line
        print("\n")
        
        history.append(
            {
                "role":"assistant",
                "content": assistant_message,
            }
        )
        
        
            
    
print("Type 'quit' to exit.\n")
    
chat(history)
    





