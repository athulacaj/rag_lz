import pickle
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from common import config

DB_PATH = config.DB_PATH
CHUNKS_FILE = os.path.join(DB_PATH, "chunks.pkl")

PROMPT_TEMPLATE = """
Answer the question based only on the following context.
If the answer cannot be found, say "I cannot find this information in the provided resumes."

Context:
{context}

Question: {question}

Answer (include source filenames as evidence):
"""

def query_rag(query_text):
    # 1. Load chunks for BM25
    if not os.path.exists(CHUNKS_FILE):
        print(f"Chunks file not found at {CHUNKS_FILE}. Run ingest.py first.")
        return
    
    with open(CHUNKS_FILE, "rb") as f:
        all_chunks = pickle.load(f)
    
    # 2. BM25 Retrieval
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 10
    bm25_results = bm25_retriever.invoke(query_text)
    
    # 3. Vector Retrieval
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL_NAME)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    vector_results = db.similarity_search(query_text, k=10)
    
    # 4. Manual Merge (deduplicate by chunk content)
    seen = set()
    merged_docs = []
    for doc in bm25_results + vector_results:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            merged_docs.append(doc)
    
    # 5. Rerank using Cross-Encoder
    if len(merged_docs) == 0:
        print("No relevant documents found.")
        return
    
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    pairs = [[query_text, doc.page_content] for doc in merged_docs]
    scores = reranker.predict(pairs)
    
    # Sort by score and keep top 5
    ranked = sorted(zip(merged_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in ranked[:5]]
    
    # 6. Build context
    context_text = "\n\n---\n\n".join([
        f"SOURCE: {doc.metadata.get('source', 'Unknown')}\nCONTENT: {doc.page_content}"
        for doc in top_docs
    ])
    
    # 7. Generate answer
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    print(f"\nGenerating answer using {config.MODEL_NAME}...\n")
    model = ChatOllama(model=config.MODEL_NAME)
    response = model.invoke(prompt)
    
    print("Response:")
    print(response.content)
    print("\nSources:")
    for doc in top_docs:
        print(f"- {doc.metadata.get('source', 'Unknown')}")

def main():
    query_text = "Pick freshers who recenltly passed out"
    query_rag(query_text)

if __name__ == "__main__":
    main()
