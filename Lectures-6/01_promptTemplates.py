"""
Prompt templates

Basically we should use the skills of prompt engineering to write the prompt

By using templates we put blanks inside the prompts to fill in later.

By this, we can create many other prompt in the future by only filling the blanks.

We use ChatPromptTemplate
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

#loading API keys and also calling LLM
load_dotenv()
MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(model=MODEL, temperature=0.7)

#BUILDING MESSAGES
print("DEMO 1: A template is just a message builder")
#We create template by making blanks {blank} like this
template = ChatPromptTemplate.from_messages([
    ("system","You are friendly {role}."),
    ("human","{question}")
])

#now later we can invoke the template and fill the variables of role and question
filled= template.invoke(
    {
        "role":"Math Teacher",
        "question":"What is 12*12?"
    }
)

print(filled.to_messages()) # converts the prompt into a list of Human/System messages

#Demo 2-> One template can make many prompts
print("DEMO 2: Reuse the same template with different inputs")

#by chainning the langhcain will automatically call the (template.invoke()) first then it calls the llm (CHatGroq) automatically without explicitly writing
chain= template | llm

questions=[
    {"role":"Math Teacher","question":"What is 12*12"},
    {"role": "pirate", "question": "What is 12 * 12?"},
    {"role": "poet", "question": "What is 12 * 12?"},    
]
#Looping through questions  and printing

for q in questions:
    result = chain.invoke(q)  #doing chain (template | llm) on each query
    print(f"[{q['role']}] {result.content}")


#Giving prompt templates within prompt templates
print("DEMO 3: Few-shot - teach by example")

few_templates =ChatPromptTemplate.from_messages([
    ("system", "Convert product names into short, punchy taglines. Follow the style below."),
    ("human", "Product: wireless earbuds"),
    ("ai", "Cut the cord. Keep the beat."),
    ("human", "Product: standing desk"),
    ("ai", "Sit less. Stand tall."),
    ("human","Product: {product}"),    
])

chain = few_templates | llm
result = chain.invoke({"product":"noise-cancelling headphones"})
print(result.content)

