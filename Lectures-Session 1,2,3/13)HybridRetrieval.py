from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

DOCS_PATH = "docs"
PERSIST_DIR = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

query = "financial aid FAFSA deadline"
print(f"Query: {query}\n")

loader = DirectoryLoader(
    path=DOCS_PATH,
    glob="*.txt",
    loader_cls=TextLoader
)
allDocuments = loader.load()

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space":"cosine"}
)
vector_retriever = db.as_retriever(search_kwargs={"k":3})


#BM25 finds those chunks only whose KEYWORDS are similar
bm25_retriever = BM25Retriever.from_documents(allDocuments)
bm25_retriever.k = 3

#Blend of both vector (70%) and bm25 (30%) 
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever,bm25_retriever],
    weights=[0.7,0.3]
)

print("\n=== Hybrid (70% vector + 30% BM25) ===")
hybrid = hybrid_retriever.invoke(query)

for i,doc in enumerate(hybrid,1):
    print(f"\nDocument {i}")
    print(doc.page_content[:120])