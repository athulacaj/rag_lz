from functions.query_utils import (
    load_bm25_chunks,
    get_bm25_results,
    get_vector_results,
    merge_and_deduplicate,
    rerank_documents,
    generate_answer
)

def query_rag(query_text):
    """Main RAG pipeline."""
    # 1. BM25 Retrieval
    chunks = load_bm25_chunks()
    if chunks is None: return
    # bm25_docs = get_bm25_results(chunks, query_text)
    bm25_docs = []
    
    # 2. Vector Retrieval
    vector_docs = get_vector_results(query_text)
    
    # 3. Merge & Deduplicate
    merged_docs = merge_and_deduplicate(bm25_docs, vector_docs)
    
    if not merged_docs:
        print("No relevant documents found.")
        return

    # 4. Rerank
    top_docs = rerank_documents(query_text, merged_docs)
    
    # 5. Generate Answer
    answer = generate_answer(query_text, top_docs)
    
    print("Response:")
    print(answer)
    print("\nSources:")
    for doc in top_docs:
        print(f"- {doc.metadata.get('source', 'Unknown')}")

def main():
    query_text = "Which candidates show an interest in sports or athletic activities in their resumes?"
    query_rag(query_text)

if __name__ == "__main__":
    main()
