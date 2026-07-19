"""
Without langchain, for calling tools we need to do the following:
1) Send tool list to model
2) mODEL requests the tool
3) our system executes the tool
4)we send the result of the tool bck
5) model give final answer

with langchain this process become automatic

with create_agent() function we can do it automatically by just giving the tools + model. It runs the loop by itself
it can call more than one tool also if needed


"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()
MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(model=MODEL, temperature=0)

#MAKING OUR NORMAL FUNCTIONS INTO TOOLS BY @tool
@tool
def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    print(f"   [tool running] get_weather(city='{city}')")

    # real weather api key call
    fake_weather = {"karachi": "34 C, sunny", "london": "18 C, rainy", "tokyo": "25 C, cloudy"}
    return fake_weather.get(city.lower(), "22 C, clear sky")


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts an amount of money from one currency to another."""
    print(f"   [tool running] convert_currency({amount}, '{from_currency}', '{to_currency}')")
    rates = {"USD": 1.0, "PKR": 278.0, "EUR": 0.9}     # fake rates for the demo
    result = amount / rates[from_currency.upper()] * rates[to_currency.upper()]
    return f"{amount} {from_currency} = {result:.0f} {to_currency}"



#Creating an agent and giving the agent tools amd model
agent = create_agent(
    model = llm,
    tools=[get_weather,convert_currency],
    system_prompt="You are a helpful assistant. Use tools whenever it is necessary to answer the specific question, and if they are not needed, answer the question directly.",
)

def ask(question:str):
    """Small helper function that run the agent,check if tool calsl exists then call tools if exist and print result"""

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    for msg in result['messages']:
        #if tool calls exist then print the tool calls
        if getattr(msg,"tool_calls",None):
            for call in msg.tool_calls:
                print(f"   [agent decided to call] {call['name']}({call['args']})")    

        print("Final answer:", result["messages"][-1].content)


print("\n\n DEMO 1: Single tool call")
ask("What is the weather in Karachi?")


print("DEMO 2: The agent chains multiple tool calls on its own")
ask("What is the weather in Tokyo? Also, how much is 100 USD in PKR?")



              