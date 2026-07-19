"""
Chaining the runnables through the pipe operator ( | )

Langchain piece like runnables which are prompt model parser retiever can be chained with | operator.

Data flows from left to right sequentially ina strict chainning manner.

"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser  

#StrOutputParser parses the response object sent by LLM into a readable string that can be printed directly 

load_dotenv()
MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(model=MODEL, temperature=0.7)

#creating prompt templates
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer in exactly one short sentence."),
        ("human", "{question}"),        
    ]
)

#DEMO 1 : Chaining prompt and llm
print("DEMO 1: prompt | llm")


chain = prompt | llm 
result = chain.invoke({"question":"Why is the sky blue?"})
print(result.content)

#DEMO 2:  StrOutputParser
print("DEMO 2: prompt | llm | StrOutputParser")

#we dont need to print result.content anymore now. because of StOutputParser() we can print result directly
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "Why is the sky blue?"})
print(result)


#DEMO 3: .batch() function that takes many questions at a time
print("DEMO 3: .batch() - many questions, one call")

questions=[  
    {"question": "Why is the sky blue?"},
    {"question": "Why do we dream?"},
    {"question": "Why is the ocean salty?"},
]

answers = chain.batch(questions)

#the answers and questions array are same index so we print both question and answer at the same time
for q,a in zip(questions,answers):
    print(f"Q: {q['question']} \n A: {a}\n")
