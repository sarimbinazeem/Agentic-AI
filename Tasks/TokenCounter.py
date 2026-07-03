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

questions = ["Hi.","Explain what Python is in 2 sentences.","Write a detailed explanation of how the internet works, including DNS, TCP/IP, HTTP, and servers.", ]

for prompt in questions:
    
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[
            {
                "role":"user",
                "content":prompt,
            }
        ],
    )
    
    print("Question: ",prompt)
    print("Response: ",response.choices[0].message.content)
    
    print("Tokens Used: ",response.usage.prompt_tokens) #Tokens it take from user to AI
    print("Completion Tokens: ",response.usage.completion_tokens) #Tokens it take from AI to user
    
    print("\n")