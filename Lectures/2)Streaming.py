import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding = "utf-8")
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DO_API_KEY"),
    base_url=os.getenv("DO_BASE_URL"),
)

stream = client.chat.completions.create(
    model=os.getenv("MODEL"),

    messages=[
        {
            "role":"user",
            "content":"Count from 1 to 20, one per line",
        }
    ],

    stream = True,

)

for chunk in stream:
    if not chunk.choices:
        continue #continue if the chunk in the stream is empty

    delta = chunk.choices[0].delta.content or ""
    #delta here tells what next text chunk is there to be printed

    print(delta,end="",flush=True) #end="" ignores new line, flush buffers output