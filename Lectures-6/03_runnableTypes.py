"""
Runnable are of types that are chainned with pipe operator ( | )

Runnable Sequence -> Normal chainning with | that runs sequentially from left to Right

Runnable Parallel -> Runs several runnables at the same time and give an output of dictionary of results

RunnableLambda -> turns a normal python function into a runnable so that our function can chain as well

RunnablePassthrough -> it does nothing. it passes the input striaght through. 

"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

load_dotenv()
MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(model=MODEL, temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer in exactly one short sentence.{aaj ke halaat kia he, 14-7-2026} asdfjakldsfjas;dfasdjf(5000 tokens)zxcsdf"), #random gibberish
    ("human", "{question}"),
])


#RUNNABLE SEQUENCE
print("DEMO 1: RunnableSequence (prompt | llm | parser)")

#it is normal chainning with pipe operator
chain = prompt | llm | StrOutputParser()

# formatted_prompt = prompt.format(input_variable="value")  #it formats the prompt like it FILLS IN THE BLANK of the template prompt
# response = llm.invoke(formatted_prompt)
# output = StrOutputParser().parse(response.content)

# print(type(chain).__name__)      # <- RunnableSequence

print(chain.invoke({"question": "Why is the sky blue?"}))

# RUNNABLE LAMBA THAT plugs plain functuion with |
print("DEMO 2: RunnableLambda - your own function in the chain")

shout = RunnableLambda(lambda text: text.upper() + "!!!")

chain = prompt | llm | StrOutputParser() |  shout
print(chain.invoke({"question": "Why is the sky blue?"}))

#RUNNABLE PARALLEL that runs all the steps side by side
print("DEMO 3: RunnableParallel - same input, multiple branches at once")

#It gives a dictionary of resultys of all the step that are running paralelly
parallel = RunnableParallel(
    answer= prompt | llm | StrOutputParser(),
    shouted = prompt | llm | StrOutputParser() | shout,
)

result= parallel.invoke({"question": "Why is the sky blue?"})
print(result) # <- {"answer": "...", "shouted": "...!!!"}

#RUNNABLE PASS TRHOUGH -> Keep the orginal input mainly during runnableparallel

print("DEMO 4: RunnablePassthrough - pass input through untouched")
parallel_passthrough = RunnableParallel(
    question=RunnablePassthrough(), #it keeps the original questions
    answer= prompt | llm | StrOutputParser(),

)

result = parallel_passthrough.invoke({"question","Why is sky blue?"})
print(result)

