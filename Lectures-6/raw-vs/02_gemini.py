"""
Gemini Api CALL

there are two ways
raw http request positng or
direct SDK way that automatically sends rqeust to api in less line of codes

it calles messages -> parts

"""


import os
import requests
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("")
MODEL = "gemini-2.5-flash"
QUESTION = "What is Gemini famous for? Answer in one sentence."

#Raw HTTPS requests

print("CUSTOM: raw requests.post()")

response = requests.post(
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
    headers={
        "x-goog-api-key": API_KEY,
        "Content-Type": "application/json",
    },
    json={"contents": [{"parts": [{"text": QUESTION}]}]},
)
data = response.json()
print(data["candidates"][0]["content"]["parts"][0]["text"])  #data.candidates[0].content.parts[0].text

#Using offical sdk for generation
print("USUAL: google.genai.Client() SDK")

client = genai.Client(api_key=API_KEY)
result = client.models.generate_content(model=MODEL, contents=QUESTION)
print(result.text)


