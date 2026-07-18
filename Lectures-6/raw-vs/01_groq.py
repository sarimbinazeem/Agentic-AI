"""
Groq API calls

two ways to call the groq api.
first is to send the raw HTTPS 
second is to use the SDK which is the direct way
"""

import os
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"
QUESTION = "What is Groq famous for? Answer in one sentence."

# Raw HTTPS call
print("CUSTOM: raw requests.post()")

response = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",

    #to authorize that im able to use the api , and to clarify that the file type is json
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },

    json={
        "model":MODEL,
        "messages": [{"role": "user", "content": QUESTION}],
    },

)

# Convert the HTTP response into a Python dictionary
data = response.json()
print(data["choices"][0]["message"]["content"])    

#Direct SDK way
# SDK automatically sends the request and parses the response
print("USUAL: groq.Groq() SDK")
client = Groq(api_key=API_KEY)
completion = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": QUESTION}],
)
print(completion.choices[0].message.content)