import os
import pickle
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from common import config

DATA_PATH = config.DATA_PATH
DB_PATH = config.DB_PATH

def create_vector_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Directory '{DATA_PATH}' created. Please add PDF resumes there.")
        return

    print("Starting fresh ingestion for nomic-embed-text...")
    print("Loading documents...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if not documents:
        print("No documents found in 'input' folder. Please add some PDF resumes.")
        return

    # Enrich metadata with document info
    for i, doc in enumerate(documents):
        doc.metadata["doc_id"] = os.path.basename(doc.metadata.get("source", f"doc_{i}"))

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk IDs for tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('doc_id', 'unknown')}_ch_{i}"

    print(f"Processing {len(chunks)} chunks...")
    
    # Clean existing DB to start fresh (BEFORE saving chunks)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    
    # Create DB directory
    os.makedirs(DB_PATH, exist_ok=True)
    
    # Save chunks for BM25 (AFTER creating directory)
    CHUNKS_FILE = os.path.join(DB_PATH, "chunks.pkl")
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {CHUNKS_FILE}")
    
    # Initialize Embedding Model
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL_NAME)

    print("Creating vector store...")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print(f"Vector store created successfully in '{DB_PATH}'.")

if __name__ == "__main__":
    create_vector_db()
