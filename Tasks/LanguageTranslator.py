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

language = input("To which language you want to translate to? ")
sentence = input("Enter A Sentence You Want To Translate: ")

history = [
    {
        "role":"system",
        "content": f"You are a language translator that translates an english sentence to {language}. Return only the translation please"
        
    },
    
    {
        "role":"user",
        "content":sentence,
    }
]

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=history,
    
    stream=True,
)

for chunk in response:
    if not chunk.choices:
        continue
    
    delta=chunk.choices[0].delta.content or ""
    
    print(delta,end="",flush=True)
    
    
    
    
    