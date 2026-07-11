import os
import sys;
from openai import OpenAI
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DO_API_KEY"),
    base_url=os.getenv("DO_BASE_URL"),
)

PERSIST_DIR = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

db = Chroma(
    persist_directory= PERSIST_DIR,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"},
)

query = "How do I request a refund?"

retriever = db.as_retriever(search_kwargs={"k":3})

docs=retriever.invoke(query)

print(f"User Query: {query}\n")
print("--- Context ---")
for i,doc in enumerate(docs,1):
    print(f"Document  {i}: \n {doc.page_content}\n")

#in it chr(10) generates \n through ASCI
combined_input = f"""Based on the following documents. Please answer this question {query}
Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])} 
Answer using ONLY the documents above. If the answer is not there, say:
"I don't have enough information in the provided documents."
"""

print("--- Prompt sent to LLM ---")
print(combined_input)

history = [
    {
        "role":"system",
        "content":"You are a helpful assistant that keeps the replies short and complete."
    },
    {
        "role":"user",
        "content":combined_input,
    },
]

response = client.chat.completions.create(
    model = os.getenv("MODEL"),
    messages = history,
    stream = True,
)

for chunk in response:
    if not chunk.choices:
        continue
    
    delta = chunk.choices[0].delta.content or ""

    print(delta,end="",flush=True)
