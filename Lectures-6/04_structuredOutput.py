"""
without langchain we explicity had to ask the model to send the response in JSON
we also had to clean up the mess by coding more and more

Langchain skip this mess
.with_structured_output() -> it gives real python output that is structured

it automatically parses JSON
"""

import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

#BaseModel and Feild are used for structured data validation

load_dotenv()
MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(model=MODEL, temperature=0)   # temperature=0 -> stay precise for data extraction


#RECIPE OBJECT
print("DEMO 1: Get a real Python object back")

#we are making class because we are the AI that we want output in this strcutrue ( the output should have the attributes that we mention in the class)

#We passes BaseModel in class because  it raises an error if the data types are incorrect and if the convetrsion is IMPOSSIBLE

#the description in Field helps LLM understand
class Recipe(BaseModel):
    name: str = Field(description="name of the dish")
    ingredients: list[str] = Field(description="short list of ingredients")
    minutes: int = Field(description="time to cook, in minutes")   

#Here we pass the class that the output schema should be of RECIPE CLASS STRUCTURER
structured_llm = llm.with_structured_output(Recipe)
result= structured_llm.invoke("Give me a simple recipe for pancakes.")

#result is a recipe object
print(result)
print("Just the name:", result.name)
print("Ingredient count:", len(result.ingredients))

#PULLING STRUCTURED OUTPUT FROM MESSY TEXT
print("DEMO 2: Extract structured data from messy text")

class Contact(BaseModel):
    name: str | None = Field(description="person's name, or null if not mentioned")
    email: str | None = Field(description="email address, or null if not mentioned")
    phone: str | None = Field(description="phone number, or null if not mentioned")

structured_llm= llm.with_structured_output(Contact)
messy_text = """
hey it's Ali again, sorry for the late reply!! you can reach me at
ali.dev99@gmail.com if the form doesn't work, no phone for now
"""

#Extracting impotant info from messy text
result=structured_llm.invoke(f"Extract Contact Info from the message: {messy_text}")
print(result)

