import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "input"
DB_PATH = "vector_db"
MODEL_NAME = "llama3.2:3b"

def create_vector_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Directory '{DATA_PATH}' created. Please add PDF resumes there.")
        return

    print("Loading documents...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    if not documents:
        print("No documents found in 'input' folder. Please add some PDF resumes.")
        return

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    print(f"Processing {len(chunks)} chunks...")
    
    # Initialize Embedding Model
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    # Clean existing DB to start fresh (optional, but good for testing)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    print("Creating vector store...")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print("Vector store created successfully in 'vector_db'.")

if __name__ == "__main__":
    create_vector_db()
