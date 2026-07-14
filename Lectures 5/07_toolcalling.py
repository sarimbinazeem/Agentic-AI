"""
AGENTS cant call tool by itself

It tells us which function to call and we run it for them.

Steps:
1)We tell model whcih function exists
2) model tell which to run
3) we run them for it   
4) we send result bck to model
5 ) model write final asnwer
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"

#Normal tools  -> we use docstring so that the model can read it
def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    print(f"   [tool running] get_weather(city='{city}')")
    fake_weather = {"karachi": "34 C, sunny", "london": "18 C, rainy", "tokyo": "25 C, cloudy"}
    return fake_weather.get(city.lower(), "22 C, clear sky")


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts an amount of money from one currency to another."""
    print(f"   [tool running] convert_currency({amount}, '{from_currency}', '{to_currency}')")
    rates = {"USD": 1.0, "PKR": 278.0, "EUR": 0.9}     # fake rates for the demo
    result = amount / rates[from_currency.upper()] * rates[to_currency.upper()]
    return f"{amount} {from_currency} = {result:.0f} {to_currency}"


#MANUAL TOOL CALLING
print("DEMO 1: Manual - see what the model asks for")

config = type.GenerateContentConfig(
    tools=[get_weather],
    #turning off the automatic 
    automatic_function_calling= types.AutomaticFunctionCallingConfig(disable=True),
)

conversation=[
    types.Content(role="user" ,parts=[types.Part(text="What is weather of karachi today?")])
]

#now the model asks us to call function
response= client.models.generate_content(model=MODEL,contents=conversation,config=config)
tool_call= response.function_calls[0]
print("Model asks us to run:", tool_call.name, "with arguments:", dict(tool_call.args))

result = get_weather(**tool_call.args)
print("Our function returned:", result)

#Sending the result
conversation.append(response.candidates[0].content) #tool request
conversation.append(
    types.Content(
        role="user",
        parts=[types.Part.from_function_response(name=tool_call.name, response={"result": result})],
    )
)


# Step 5: now the model writes the final answer
response = client.models.generate_content(model=MODEL, contents=conversation, config=config)
print("Final answer:", response.text)

#AUTOMATIC 

print("DEMO 2: Automatic - the SDK runs the tools for us")

config = types.GenerateContentConfig(tools=[get_weather,convert_currency])

response=client.models.generate_content(
    model=MODEL,
    contents="What is the weather in Tokyo? Also, how much is 100 USD in PKR?",
    config=config
)

print("Final answer:", response.text)