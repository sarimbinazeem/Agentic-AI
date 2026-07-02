import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding = "utf-8")
load_dotenv()

client = OpenAI(
    api_key = os.getenv("DO_API_KEY"),
    base_url=os.getenv("DO_BASE_URL"),
)

PROMPT = "Write a slogan for a coffee shop."


def streaming(response):
    for chunk in response:
        if not chunk.choices:
            continue #continue if the chunk in the stream is empty

        delta = chunk.choices[0].delta.content or ""
        #delta here tells what next text chunk is there to be printed

        print(delta,end="",flush=True) #end="" ignores new line, flush buffers output


#temperature prompt -> tells how creative it should be 
print("\n====Temperature 0.0====");
cold = client.chat.completions.create(
    model = os.getenv("MODEL"),
    messages = [
        {
            "role":"user",
            "content":PROMPT,
        }
    ],

    stream = True,

    temperature=0.0,
)

streaming(cold)


print("\n====Temperature 1.5====");
hot = client.chat.completions.create(
    model = os.getenv("MODEL"),
    messages = [
        {
            "role":"user",
            "content":PROMPT,
        }
    ],

    stream = True,

    temperature=1.5,
)

streaming(hot)

#max tokens
print("\n=== max_tokens: 10 (truncated) ===")
short = client.chat.completions.create(
    model = os.getenv("MODEL"),
    messages=[
        {
            "role":"user",
            "content":"Tell me about our solar system",
        }
    ],


    max_tokens=30,
)

print(short.choices[0].message.content)
print("Reason of Unfinished: ",short.choices[0].finish_reason)


#system persona
print("\n=== System Persona: Pirate ===")
pirate = client.chat.completions.create(
    model = os.getenv("MODEL"),
    messages=[
        {
            "role":"system",
            "content":"You are a Pirate. Always respond as a pirate"
        },
        {
            "role":"user",
            "content":"What time is it?"
        }

        
    ],

    stream = True,

)

streaming(pirate)
