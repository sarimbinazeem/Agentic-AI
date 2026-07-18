"""
OPENAI API CALLS

raw HTTPS vs SDK method to call via Digital Ocean

instead of giving the actual url of openAi , we point openAI SDK at Digital Ocean endpoint.
By this we get a difference which are base_url and api_key which arent in real OpenAi

through this SDK dont know it isnt talking to openAI

(easy wording: digital ocean ghave made its url look like openAI which is why we need a base_url for digital ocean)

"""

import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("DIGITAL_OCEAN_MODEL_ACCESS_KEY")
BASE_URL = "https://inference.do-ai.run/v1/"
MODEL = "llama3-8b-instruct"   # see client.models.list() for the full catalog
QUESTION = "What is the OpenAI chat-completions format famous for? One sentence."

#HTTP raw call
print("CUSTOM: raw requests.post()")


response = requests.post(
    f"{BASE_URL}chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": QUESTION}],
    },
)
data = response.json()
print(data["choices"][0]["message"]["content"])

#Official openAI sdkm pointed at DIgital Ocean
print("USUAL: openai.OpenAI() SDK, base_url swapped to DigitalOcean")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY) 
completion = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": QUESTION}],
)
print(completion.choices[0].message.content)
