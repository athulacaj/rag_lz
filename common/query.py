from functions.query_utils import (
    load_bm25_chunks,
    get_bm25_results,
    get_vector_results,
    merge_and_deduplicate,
    rerank_documents,
    generate_answer,
    get_section_using_llm,
    polish_question,
    get_sql_using_llm,
    check_need_more_context_needed
)
from functions.make_section import CV_HEADING_PATTERNS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from config import MODEL_NAME,DB_NAME,PARSER,EMBEDDING_MODEL_NAME
import json
import functions.database_utils as db_utils
import logging

# Configure logger
logger = logging.getLogger('rag_logger')

def get_connection(db_name=None):
    return db_utils.get_db_connection(db_name or DB_NAME)


def query_rag(query_text, model_name=None, embedding_model=None, parser=None, db_name=None):
    """Main RAG pipeline."""
    current_model = model_name or MODEL_NAME
    current_parser = parser or PARSER
    current_embedding = embedding_model or EMBEDDING_MODEL_NAME
    current_db = db_name or DB_NAME

    logger.info(f"Starting RAG query for: {query_text}")
    logger.info(f"LLM Model: {current_model}")
    logger.info(f"Used PARSER: {current_parser}")
    logger.info(f"Embedding Model: {current_embedding}")
    logger.info(f"DB Name: {current_db}")

    # 1. BM25 Retrieval
    # chunks = load_bm25_chunks()
    # if chunks is None: return
    # bm25_docs = get_bm25_results(chunks, query_text)
    # bm25_docs = []
    
   
    question_dict=polish_question(query_text, model_name=current_model)
    
    names=question_dict["names"]
    emails=question_dict["emails"]
    polished_question=question_dict["polished_question"]
    logger.info(f"Polished question: {polished_question}")
    logger.info(f"Polished question: {question_dict}")
    db_results=[]
    sql_data_str=""


    if(len(emails)>0):
        with get_connection(current_db) as conn:
            sql_data=db_utils.get_data_by_email(conn,emails)
            if sql_data:
                db_results.append({
                    "name":sql_data[0]["general"]["name"],
                    "email":sql_data[0]["general"]["email"],
                })
    elif(len(names)>0):
        with get_connection(current_db) as conn:
            sql_data=db_utils.get_data_by_name(conn,names)
            for data in sql_data:
                db_results.append({
                    "name":data["general"]["name"],
                    "email":data["general"]["email"],
                })
    
    if polished_question.lower()=="not related":
        logger.info("Question not related to context.")
        return "I can only answer questions related to the resume/context.","no context"


    top_docs = []
    section_names = []
    section=get_section_using_llm(polished_question, model_name=current_model)
    # 2. Vector Retrieval
    section_names=section["sections"]
    logger.info(f"Identified sections: {section_names}")
    need_more_context=True
    if len(section_names)==1:
        if section_names[0] in ["general","skills","experience"]:
                # get sql query and data from db
            with get_connection(current_db) as conn:
                schema=db_utils.get_schema(conn)
                schema_text=db_utils.schema_to_text(schema)
                section=get_sql_using_llm(polished_question,schema_text)
                sql_query=section["query"]
                logger.info(f"Sql result is based on: {section["format_result"]}")
                logger.info(f"SQL Query: {sql_query}")
                if(sql_query):
                    with get_connection(current_db) as conn:
                        sql_data=db_utils.get_data_by_sql(conn,sql_query)
                        logger.info(f"SQL Data: {sql_data}")
                        if sql_data:
                            sql_data_str+="\n\n# start of SQL Data"
                            sql_data_str+="\n##"+ section["format_result"]
                            sql_data_str+=": in csv format:\n"
                            sql_data_str+=",".join(section["headers"])+"\n"
                            sql_data_str+="\n".join(
                                [
                                    "" if x is None
                                    else ",".join("" if i is None else str(i) for i in x)
                                    if isinstance(x, tuple)
                                    else str(x)
                                    for x in sql_data
                                ]
                            )
                            sql_data_str+="\n# end of SQL Data\n"
            if(sql_data_str is not None and sql_data_str!=""):
                need_more_context_dict=check_need_more_context_needed(polished_question,sql_data_str)
                need_more_context=need_more_context_dict["need_more_context"]=="True"

    if(need_more_context):

        chunk_ids=[]
        if(len(db_results)>0):
            for data in db_results:
                for section in section_names:
                    chunk_ids.append(data["email"]+"_"+section)
        

        logger.info(f"Need more context: {need_more_context}")
        vector_docs = get_vector_results(polished_question,section_names,chunk_ids, embedding_model_name=current_embedding)
        
        vector_ids=[]
        for doc in vector_docs:
            vector_ids.append(doc.id)
        logger.info(f"Vector docs id: {vector_ids}")

        # 3. Merge & Deduplicate
        # merged_docs = merge_and_deduplicate(bm25_docs, vector_docs)
        merged_docs = vector_docs
        
        if not merged_docs:
            logger.info("No relevant documents found.")
            return "No relevant documents found.","no context"

        # 4. Rerank
        # top_docs = rerank_documents(query_text, merged_docs)
        top_docs = merged_docs

    else:
        logger.info("No need for more context.")
        
    # 5. Generate Answer
    answer,context_text = generate_answer(query_text, top_docs,section_names, model_name=current_model,context=sql_data_str)
    
    logger.info("Answer generated successfully.")
    
    result = answer + "\n\nSources:\n"
    for doc in top_docs:
        result += f"- {doc.metadata.get('source', 'Unknown')}\n"
    
    return result, context_text

def main():
    # Setup basic logging for CLI usage
    logging.basicConfig(level=logging.INFO)
    query_text = "is athul and nihal interested in sports"
    response, _ = query_rag(query_text)
    print(response)
    

if __name__ == "__main__":
    main()
