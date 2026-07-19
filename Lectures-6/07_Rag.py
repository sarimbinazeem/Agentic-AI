"""
We used to create by hand the chunks, embeddings, stroing into vbector database

Langchainprovides ready made pieces for the same idea


"""


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()
MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(model=MODEL, temperature=0)

#Knowledge base

bootcamp_facts = [
    "The AI Season bootcamp runs live sessions every week covering practical AI engineering.",
    "Session 5 covered multimodal agents: vision, audio, video, OCR, and documents, using the Gemini API.",
    "Session 6 covers LangChain: prompts, chains, memory, tools, agents, and RAG, using the Groq API.",
    "Students must bring their own laptop with Python 3.10 or newer installed.",
    "Assignments are submitted by pushing code to each student's own GitHub repository.",
    "The instructor's office hours are on Sunday evenings for anyone stuck on an assignment.",
    "There is no final exam - the bootcamp is graded entirely on session projects.",
    "Free API keys are used throughout so no student has to pay for compute.",
]

#creating embeddings of the facts
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#storing it
vectorstore = FAISS.from_text(bootcamp_facts,embeddings=embeddings)
#gives top 2 facts when retriever is invoekd
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 


#DEMO 1: only retrieval
print("DEMO 1: Just retrieve - what does the search find?")
question = "What do I need to bring to class?"
docs = retriever.invoke(question)

for i, doc in enumerate(docs, start=1):
    print(f"{i}. {doc.page_content}")


#DEMO 2: Retrieval + answer (FULL RAG)
print("DEMO 2: Retrieve THEN answer using only that context")

#making prompt for llm
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using ONLY the context below. "
     "If the answer isn't in the context, say you don't know.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

answer_chain = rag_prompt | llm | StrOutputParser()

for question in [
    "What do I need to bring to class?",
    "How are students graded?",
    "What API does Session 6 use?",
]:
    
    docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)   
    answer = answer_chain.invoke({"context": context, "question": question})
    print(f"Q: {question}\nA: {answer}\n")    