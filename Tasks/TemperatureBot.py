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

prompt = "Write a one-sentence motivational quote."
temperatures = [0.0, 0.7, 1.5]

for temp in temperatures:
    
    print(f"\n====Temperature {temp}====")
    response = client.chat.completions.create(
        model = os.getenv("MODEL"),
        
        messages=[
            {
                "role":"user",
                "content":prompt,
            }
        ],
        
        temperature= temp,
    )

    print(response.choices[0].message.content)

