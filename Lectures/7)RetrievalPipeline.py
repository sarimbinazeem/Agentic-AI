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

query = "How long does a refund take to process"

retriever = db.as_retriever(search_kwargs={"k":3}) #give top 3 similar chunks

# retriever=db.as_retriever(
#     search_type ="similarity_score_threshold",
#     search_kwargs = {"k":3,"score_threshold":0.3}, #give simialr chunks who have simialirty level higher than 0.3
# )


docs = retriever.invoke(query)


print(f"User Query: {query}\n")
print("--- Retrieved Context ---")

for i,doc in enumerate(docs,1):
    print(f"Document {i}: ({doc.metadata.get('source','unknown')})") #default type is unknown
    print(doc.page_content)
    print()


