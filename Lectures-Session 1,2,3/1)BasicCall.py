import os
import sys;
from openai import OpenAI
from dotenv import load_dotenv

# To laod text correctyl
sys.stdout.reconfigure(encoding = "utf-8")
load_dotenv(); 


client = OpenAI(
    api_key= os.getenv("DO_API_KEY"),
    base_url= os.getenv("DO_BASE_URL"),
)

# To generate response from OPEN AI
response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=[
        {
            "role": "user",
            "content": "What is 2 + 2? Answer in one sentence.",
        }
    ],
)

print("Raw  Response: ",response,"\n\n---\n\n")

text = response.choices[0].message.content
print("Model Response: ",text)

# pRINTING TOKEN USAGE
print("\nToken usage:", response.usage)