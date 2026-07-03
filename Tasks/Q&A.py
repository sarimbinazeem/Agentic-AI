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

print("Type 'quit' to exit.\n")
 
while True:
    prompt = input("What do you want to ask today: ")

    #if the user enters quit we break the loop 
    if prompt.lower().strip() == "quit":
        break
    
    #if the user accidently enters a blank space we ignore and continue
    if prompt.strip() == "":
        continue
    
    
    response = client.chat.completions.create(
        model = os.getenv("MODEL"),
        
        messages=[
            {
                "role":"user",
                "content": prompt,
            }
        ]
        
    )
    
    print(response.choices[0].message.content)  
    
    

    





