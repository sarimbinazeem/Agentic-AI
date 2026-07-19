"""
Langchain reduces the boilerplate of calling the API

we dont need to write os.getenv(), data.choices[0].message.content and more complex syntax.
Instead of writing provider-specific code every time,
it gives every model the same interface such as

invoke()
stream()
batch()

it act as standard wrapper
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

#loading API 
load_dotenv()

MODEL = "llama-3.3-70b-versatile"

#calling llm in langchain way
llm = ChatGroq(model=MODEL,temperature=0.7)

#Asking question
print("DEMO 1: Ask a plain question")

#invoking question
response = llm.invoke("What is LangChain, in one simple sentence?")
print(response.content) # .content is the direct string of the text

#Giving Sysem and Human Role
print("DEMO 2: Give it a role with system + human messages")

response=llm.invoke(
    [
        ("system", "You are a grumpy pirate. Answer everything like a pirate."),
        ("human","What is the Capital of France")
    ]
)

print(response.content)

#DEMO 3 -> Streaming he response token by token
for chunk in llm.stream("Count from 1 to 5, one number per line."):
    print(chunk.content,end="",flush=True) 

print()