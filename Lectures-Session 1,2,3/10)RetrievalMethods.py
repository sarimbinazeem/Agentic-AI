from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PERSIST_DIR = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"},
)

query ="What is the in-state tuition?"
print(f"Query: {query}\n")

#Method 1 BAsic similiarity (gets top 3 similar -> even tho the third one is far away from the meaning)
retriever = db.as_retriever(search_kwargs={"k":3})
docs=retriever.invoke(query)
for i, doc in enumerate(docs, 1):
    print(f"Document {i}: {doc.page_content[:200]}...\n")

#Method 2 Score Threshold -> GETS MOST SIMILAR (MAY IT BE 1 2 3 )
retriever =db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":3,"score_threshold":0.3} 
)
docs=retriever.invoke(query)
for i, doc in enumerate(docs, 1):
    print(f"Document {i}: {doc.page_content[:200]}...\n")


#Method 3 Maximum Marginal Relevanve 
retriever = db.as_retriever(
    search_type = "mmr",
    search_kwargs={"k":3,"fetch_k":10,"lambda_mult":0.5}, #it takes 10 chunks then see top 3 most diverse and simialr (it takes both in balance becuase lamba_mult is 0.5)
)

docs=retriever.invoke(query)
for i, doc in enumerate(docs, 1):
    print(f"Document {i}: {doc.page_content[:200]}...\n")
