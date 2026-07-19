"""
Before langchain we used to append to the history everytime there was a conversation progress
when there was a user query we APPENDED IT , when the llm responed WE APPNEDED IT

with langchain we dont have to do it by hand.

RunnableWithMessageHistory do this hard work for us. It rremembers each session we have to just give it a session ID

"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


#Message Placeholder -> we use this placehodler inside the TEMPLATE so that whenever a prompt encounters a keyword "history" it automatically retrieves the chat history
#InMemoryChatHistory()  -> it returns the session's chat history 
# RunnableWithMessageHistroy() -> it is the wrrapper that takes other runnables  with memory
# 1st arg -> chain , 2nd arg -> histroy, 3rd arg-> input key , 4th arg -> keywrod to retrieve history

load_dotenv()
MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(model=MODEL, temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep answers short."),
    MessagesPlaceholder("history"), #the past messages are placed here when there is a keyword history found (the keyword here is histroy)  
    ("human", "{input}"),      
])

chain = prompt | llm

#Memory Storing -> session_id  per histroy 
store = {}  #here we get the session conversation history. We append every conversation here during each session that have a unique session ID
#In real wordl applications the store is a big database

#function to get Session history if exist or make a session ID and store the Message session history inside the store array
def get_session_history(session_id:str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()  #STORES THE CHAT HISTORY of one session
    return store[session_id]

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

#DEMO of session history
print("DEMO: session 'ali' - tell it a name, then ask for it back")

#we give our own session id it can be anything
config = {"configurable":{"session_id":"ali"}}

resp1 = chatbot.invoke({"input":"Hi! My Name Is Ali."},config=config)

print("Bot:", resp1.content)

r2 = chatbot.invoke({"input": "What is my name?"}, config=config)
print("Bot:", r2.content)   # <- remembers, same session_id

#If we change the session ID it wont remember the name 
print("DEMO: session 'sara' - a totally different, empty conversation")

other_config= {"configurable":{"session_id":"sara"}}
r3 = chatbot.invoke({"input": "What is my name?"}, config=other_config)
print("Bot:", r3.content) 