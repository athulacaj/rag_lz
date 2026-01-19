from functions.query_utils import (
    load_bm25_chunks,
    get_bm25_results,
    get_vector_results,
    merge_and_deduplicate,
    rerank_documents,
    generate_answer,
    get_section_using_llm,
    polish_question
)
from functions.make_section import CV_HEADING_PATTERNS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from config import MODEL_NAME,DB_NAME
import json
import functions.database_utils as db_utils
import logging

# Configure logger
logger = logging.getLogger('rag_logger')

def get_connection():
    return db_utils.get_db_connection(DB_NAME)


def query_rag(query_text):
    """Main RAG pipeline."""
    logger.info(f"Starting RAG query for: {query_text}")
    # 1. BM25 Retrieval
    # chunks = load_bm25_chunks()
    # if chunks is None: return
    # bm25_docs = get_bm25_results(chunks, query_text)
    # bm25_docs = []
    
   
    question_dict=polish_question(query_text)
    logger.info(f"Polished question: {question_dict}")
    names=question_dict["names"]
    emails=question_dict["emails"]
    polished_question=question_dict["polished_question"]
    db_results=[]
    if(len(emails)>0):
        with get_connection() as conn:
            sql_data=db_utils.get_data_by_email(conn,emails)
            if sql_data:
                db_results.append({
                    "name":sql_data[0]["general"]["name"],
                    "email":sql_data[0]["general"]["email"],
                })
    elif(len(names)>0):
        with get_connection() as conn:
            sql_data=db_utils.get_data_by_name(conn,names)
            for data in sql_data:
                db_results.append({
                    "name":data["general"]["name"],
                    "email":data["general"]["email"],
                })
    
    if polished_question=="not related":
        logger.info("Question not related to context.")
        return "I can only answer questions related to the resume/context."

    logger.info(f"Polished question: {polished_question}")
    section=get_section_using_llm(polished_question)
    # 2. Vector Retrieval
    section_names=section["sections"]
    logger.info(f"Identified sections: {section_names}")

    chunk_ids=[]
    if(len(db_results)>0):
        for data in db_results:
            for section in section_names:
                chunk_ids.append(data["email"]+"_"+section)
    

    vector_docs = get_vector_results(polished_question,section_names,chunk_ids)
    vector_ids=[]
    for doc in vector_docs:
        vector_ids.append(doc.id)
    logger.info(f"Vector docs id: {vector_ids}")

    # 3. Merge & Deduplicate
    # merged_docs = merge_and_deduplicate(bm25_docs, vector_docs)
    merged_docs = vector_docs
    
    if not merged_docs:
        logger.info("No relevant documents found.")
        return "No relevant documents found."

    # 4. Rerank
    # top_docs = rerank_documents(query_text, merged_docs)
    top_docs = merged_docs
    
    # 5. Generate Answer
    answer = generate_answer(query_text, top_docs,section_names)
    
    logger.info("Answer generated successfully.")
    
    result = answer + "\n\nSources:\n"
    for doc in top_docs:
        result += f"- {doc.metadata.get('source', 'Unknown')}\n"
    
    return result

def main():
    # Setup basic logging for CLI usage
    logging.basicConfig(level=logging.INFO)
    query_text = "is athul and nihal interested in sports"
    response = query_rag(query_text)
    print(response)
    

if __name__ == "__main__":
    main()
