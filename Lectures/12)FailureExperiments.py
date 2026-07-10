import os
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

sys.stdout.reconfigure()
load_dotenv()

client = OpenAI(
    api_key= os.getenv("DO_API_KEY"),
    base_url=os.getenv("DO_BASE_URL"),
)

PERSIST_DIR = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space":"cosine"},
)

def call_llm(prompt):
    
    history=[
        {
            "role": "system",
            "content": "You are a careful assistant. Answer only from the provided context."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    response=client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=history,
        
    )
    
    return response.choices[0].message.content

def run_query(query,top_k,style="normal"):
    retriever=db.as_retriever(
        search_kwargs={"k":top_k}
    )
    
    docs=retriever.invoke(query)
    
    context= "\n".join([doc.page_content for doc in docs])
    
    if style == "weak":
        prompt = f"""
                Context:
                {context}

                Question:
                {query}

                Answer:
                """
                
    else:
        prompt = f"""
        Use ONLY the context below.

        If the answer cannot be found, reply exactly:

        "I don't have enough information in the provided documents."

        Context:
        {context}

        Question:
        {query}
        """
        
    print(f"\nQuery: {query}")
    print(f"Top K = {top_k}")
    print(f"Prompt Style = {style}")

    answer = call_llm(prompt)
    
    print("\nAnswer:")
    print(answer)
    
print("===== Experiment 1 : Question not in documents =====")
run_query(
    "What is the capital of France?",
    top_k=3
)

print("\n===== Experiment 2 : Top K Comparison =====")
run_query(
    "Tell me about tuition and financial aid.",
    top_k=1
)

run_query(
    "Tell me about tuition and financial aid.",
    top_k=3
)

print("\n===== Experiment 3 : Weak vs Strong Prompt =====")
run_query(
    "Can alumni borrow books from the library?",
    top_k=2,
    prompt_style="weak"
)

run_query(
    "Can alumni borrow books from the library?",
    top_k=2,
    prompt_style="strong"
)
