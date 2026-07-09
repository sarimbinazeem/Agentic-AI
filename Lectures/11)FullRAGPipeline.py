import os 
import sys
from openai import OpenAI
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

DOCS_PATH = "docs"
PERSIST_DIR = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

client = OpenAI(
    api_key = os.getenv("DO_API_KEY"),
    base_url=os.getenv("DO_BASE_URL"),
)


def create_vector_store():
     print("[1] Loading documents...")

     loader = DirectoryLoader(
          path=DOCS_PATH,
          glob="*.txt",
          loader_cls=TextLoader,
     )

     documents = loader.load()

     print("[2] Chunking documents...")
     splitters = RecursiveCharacterTextSplitter(
          separators=["\n\n","\n",". "," ",""],
          chunk_size=500, 
          chunk_overlap=50,
     )

     chunks = splitters.split_documents(documents)

     print("[3] Embedding CHunks...")
     embeddings= HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

     print("[4] Creating Vector Database...")
     return Chroma.from_documents(
          documents = chunks,
          embedding=embeddings,
          persist_directory= PERSIST_DIR,
          collection_metadata={"hnsw:space": "cosine"},
     )


def ask(db,question,top_k=3):
     print(f"\nQuestion: {question}")
     print("[5] Retrieving Query Vector Database...")
     
     retriever = db.as_retriever(search_kwargs={"k":top_k})
     docs = retriever.invoke(question)

     print(f"    Retrieved {len(docs)} chunk(s)")

     context = "\n".join([f"-{doc.page_content}" for doc in docs ])

     prompt = f"""Answer this question using ONLY the context below.

        Context:
        {context}

        Question: {question}

        If the answer is not in the context, say you do not have enough information.
        """
     
     
     print("[6] Calling LLM...")
     print("\n--- Prompt ---")
     print(prompt)

     message = [
          {
               "role":"system",
               "content":"You are a helpful assistant.",
          },
          {
               "role":"user",
               "content": prompt,
          },
     ]

     response = client.chat.completions.create(
          model = os.getenv("MODEL"),

          messages= message,
     )

     print("\n--- Answer ---")
     print(response.choices[0].message.content)

     print("\n--- Sources ---")
     for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.metadata.get('source')}")
        print(f"      {doc.page_content[:120]}...")





def main():
     print("=== Full RAG Pipeline ===\n")
     
     if os.path.exists(PERSIST_DIR):
          embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
          db = Chroma(
               persist_directory=PERSIST_DIR,
               embedding_function=embeddings,
               collection_metadata={"hnsw:space":"cosine"},
          )

          print(f"Loaded existing vector store ({db._collection.count()} chunks)\n")

     else:
          db = create_vector_store()
    

     ask(db, "What is the in-state tuition at Riverside University?")
     ask(db, "How do I request a refund from ACME store?")


if __name__ == "__main__":
    main()