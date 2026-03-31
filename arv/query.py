import sys
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from config import DATA_PATH, DB_PATH, MODEL_NAME, COLLECTION_NAME, EMBEDDING_MODEL_NAME

def query_rag(query_text):
    # --- LlamaIndex Settings ---
    Settings.llm = Ollama(model=MODEL_NAME, request_timeout=360.0, context_window=4096)
    # Important: Must match the model used for ingestion!
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)

    # --- Connect to Chroma ---
    print(f"Connecting to Chroma at '{DB_PATH}'...")
    client = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Load index from vector store
    # Since we are loading an existing index, we use from_vector_store
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Create Query Engine
    print(f"Querying with {MODEL_NAME}...")
    query_engine = index.as_query_engine(similarity_top_k=5)
    
    # Run Query
    response = query_engine.query(query_text)
    
    print("\n" + "="*30)
    print("Response:")
    print("="*30)
    print(response)
    print("\n" + "="*30)
    print("Source Nodes:")
    print("="*30)
    
    for node in response.source_nodes:
        # Access metadata from valid nodes
        metadata = node.metadata
        print(f"- Source: {metadata.get('file_name', 'Unknown')}")
        print(f"  Score: {node.score:.4f}")
        # print(f"  Snippet: {node.text[:100]}...")
        print("-" * 10)

def main():
    # If using command line args:
    # if len(sys.argv) < 2:
    #     print("Usage: python query.py \"Your question here\"")
    #     return
    # query_text = sys.argv[1]
    
    query_text = input("What is your question?\n")
    print(f"Question: {query_text}")
    query_rag(query_text)

if __name__ == "__main__":
    main()
