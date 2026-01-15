import pickle
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
# Assumes 'common' is in sys.path when running query.py
from config import DATA_PATH, DB_PATH, EMBEDDING_MODEL_NAME, MODEL_NAME,COLLECTION_NAME

CHUNKS_FILE = os.path.join(DB_PATH, "chunks.pkl")

PROMPT_TEMPLATE = """
Answer the question based only on the following context.
If the answer cannot be found, say "I cannot find this information in the provided resumes."

Context:
{context}

Question: {question}

Answer (include source filenames as evidence):
"""

def load_bm25_chunks():
    """Lengths and loads chunks for BM25 retrieval."""
    if not os.path.exists(CHUNKS_FILE):
        print(f"Chunks file not found at {CHUNKS_FILE}. Run ingest.py first.")
        return None
    
    with open(CHUNKS_FILE, "rb") as f:
        return pickle.load(f)

def get_bm25_results(chunks, query_text):
    """Retrieves documents using BM25."""
    if not chunks: 
        return []
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 10
    return retriever.invoke(query_text)

def get_vector_results(query_text):
    """Retrieves documents using vector similarity."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    # use NER to get the section
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings, collection_name=COLLECTION_NAME)
    results = db.similarity_search_with_score(query_text, k=10,filter={"section": "interests"})
    # Filter by score
    return [doc for doc, score in results if score > 0.75]

def merge_and_deduplicate(bm25_docs, vector_docs):
    """Merges and deduplicates documents by content."""
    seen = set()
    merged = []
    for doc in bm25_docs + vector_docs:
        h = hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            merged.append(doc)
    return merged

def rerank_documents(query_text, docs):
    """Reranks documents using CrossEncoder."""
    if not docs:
        return []
        
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    pairs = [[query_text, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    
    # Sort and take top 5
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:5]]

def merge_same_source(docs):
    """Merges documents with the same source."""
    merge_dict={}
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        if(merge_dict[source]):
            merge_dict[source].page_content += "\n\n" + doc.page_content
        else:
            merge_dict[source] = doc
    return list(merge_dict.values())

def generate_answer(query_text, context_docs):
    """Generates answer using LLM."""
    context_text = "\n\n---\n\n".join([
        f"SOURCE: {doc.metadata.get('source', 'Unknown')}\nCONTENT: {doc.page_content}"
        for doc in context_docs
    ])
    
    template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = template.format(context=context_text, question=query_text)
    
    print(f"\nGenerating answer using {MODEL_NAME}...\n")
    model = ChatOllama(model=MODEL_NAME)
    response = model.invoke(prompt)
    
    return response.content
