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


def chat(history: list , user_message: str) -> str:
    history.append({
        "role":"user",
        "content": user_message,
    })

    response = client.chat.completions.create(
        model = os.getenv("MODEL"),

        messages = history,
    )

    assistant_message = response.choices[0].message

    history.append(
        {
            "role" : "assistant",
            "content" : assistant_message.content,
        }

    )

    return assistant_message.content
   

history = [
    {
        "role" : "system",
        "content" : "your a helpful assistant. Keep answers short",
    }
]


reply1 = chat(history, "My name is Sarim")
print("Turn 1: ",reply1)
reply2 = chat(history, "What is The Capital of Pakistan")
print("Turn 2: ",reply2)
reply3 = chat(history, "What is my name?")
print("Turn 3: ",reply3)


import json
print("\n Full History Sent On Last Call");
print(json.dumps(history, indent=2))


