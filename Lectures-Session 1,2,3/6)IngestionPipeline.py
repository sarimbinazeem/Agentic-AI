import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

DOCS_PATH = "docs"
PERSIST_DIR = "db/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_docs(docPath = DOCS_PATH):
    print(f"Loading Documents from {docPath}...")
    
    if not os.path.exists(docPath):
     raise FileNotFoundError(f"Folder not found: {docPath}. Run 1_file_conversion.py first.")
 
    loader = DirectoryLoader(
        path =docPath,
        glob = "*.txt",
        loader_cls = TextLoader, #do with textloader function
    )
    
    documents = loader.load() #it have one object metadeta and page_Content array
    
    #if there are no documents 
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files in {docPath}")
    
    #we only show the first two documents
    for i,doc in enumerate(documents[:2]):
        print(f"\n====Document {i+1}====\n")
        print(f"Source: {doc.metadata['source']}")
        print(f"Length: {len(doc.page_content)} characters.")
        print(f"Preview: {doc.page_content[:120]}")
    
    return documents

def split_docs(documents,chunk_size = 500, chunk_overlap=50):
    print("\nSplitting documents into chunks...")
    
    splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    #show only the first three chunks
    for i,chunk in enumerate(chunks[:3]):
        print(f"\n===Chunk {i+1}===")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Text Preview: {chunk.page_content[:150]}...")       
    
    if len(chunks) > 3:
        print(f"\n... and {len(chunks) - 3} more chunks")
        
    return chunks

def create_vector_store(chunks,persist = PERSIST_DIR):
    print("\nCreating embeddings and storing in ChromaDB...")
    print("(First run downloads the local embedding model ~80MB)")
    
    embedding_model = HuggingFaceEmbeddings(model_name =EMBEDDING_MODEL)
    
    vectorStore = Chroma.from_documents(
        documents = chunks,
        embedding= embedding_model,
        persist_directory= persist,
        collection_metadata={"hnsw:space":"cosine"},
    )
    
    print(f"Vector Store Saved To {persist}")
    return vectorStore

def main():
    print("=== RAG Ingestion Pipeline ===\n")
    
    #if persist directory already exists
    
    if os.path.exists(PERSIST_DIR):
         print("Vector store already exists. Loading it instead of re-ingesting.\n")
         
         embedding_model = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
         
         vectorStore = Chroma(
             persist_directory=PERSIST_DIR,
             embedding_function=embedding_model,
             collection_metadata={"hnsw:space":"cosine"},
         )    
         print(f"Loaded {vectorStore._collection.count()} chunks from {PERSIST_DIR}")
         return vectorStore         
     
    documents = load_docs()
    chunks = split_docs(documents)
    vectors = create_vector_store(chunks)
    

    print("\nIngestion complete.")
    return vectors


if __name__ == "__main__":
    main()