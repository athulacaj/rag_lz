import os
import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb
from config import DATA_PATH, DB_PATH, MODEL_NAME, COLLECTION_NAME, EMBEDDING_MODEL_NAME

def create_vector_db():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Directory '{DATA_PATH}' created. Please add PDF resumes there.")
        return

    print(f"Loading documents from {DATA_PATH}...")
    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    
    if not documents:
        print("No documents found in 'input' folder. Please add some PDF resumes.")
        return

    print(f"Loaded {len(documents)} documents.")

    # --- LlamaIndex Settings ---
    print(f"Setting up Ollama ({MODEL_NAME}) and Embeddings ({EMBEDDING_MODEL_NAME})...")
    Settings.llm = Ollama(model=MODEL_NAME, request_timeout=360.0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)

    # --- Chroma Vector Store ---
    print(f"Initializing Chroma locally at '{DB_PATH}'...")
    
    # Chroma client (local mode)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Check if collection exists and delete if needed (for fresh ingest)
    # Using list_collections() to check existence
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        print(f"Deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    # Create Chroma collection
    chroma_collection = client.get_or_create_collection(COLLECTION_NAME)

    # Create Chroma vector store wrapper
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Creating index and pushing vectors to Chroma...")
    # This automatically handles chunking (default chunks) and embedding
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("Vector store created successfully.")

if __name__ == "__main__":
    create_vector_db()
