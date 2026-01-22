import pickle
import os
import sys
import re
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
# Ensure 'common' directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH, DB_PATH, EMBEDDING_MODEL_NAME, MODEL_NAME,COLLECTION_NAME,DB_NAME,SQL_MODEL   
import functions.database_utils as db_utils
from functions.gemini_utils import get_gemini_json_response,get_gemini_response
import json
from datetime import datetime


CHUNKS_FILE = os.path.join(DB_PATH, "chunks.pkl")

PROMPT_TEMPLATE = """
Answer the question based only on the following context.
If the answer cannot be found, say "I cannot find this information in the provided resumes."
IF you cannot find the answer from the context, explain why.
Explain your answer in detail with proper reasoning.
Context:
{context}

Question: {question}

Answer:
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

def get_vector_results(query_text,section_list=[],chunk_ids=[], embedding_model_name=None,context=""):
    target_embedding_model = embedding_model_name or EMBEDDING_MODEL_NAME
    if "gemini" in target_embedding_model:
        return get_vector_results_gemini(query_text,section_list,chunk_ids, embedding_model_name=target_embedding_model)

    """Retrieves documents using vector similarity."""
    embeddings = OllamaEmbeddings(model=target_embedding_model)
    # use NER to get the section
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings, collection_name=COLLECTION_NAME)
    lst=[{"section": x} for x in section_list]
    filter=None
    if len(section_list)==1:
        filter=lst[0]
    elif len(section_list)>1:
        filter={
            "$or": lst
        }
    
    results = []
    if(len(chunk_ids)>0):
        return db.get_by_ids(chunk_ids)
    else:
        results=db.similarity_search_with_score(
            query_text,
            k=10,
            filter=filter
        )  
    return [doc for doc, score in results]

def get_vector_results_gemini(query_text,section_list=[],chunk_ids=[], embedding_model_name=None):
    """Retrieves documents using Gemini vector similarity."""
    target_embedding_model = embedding_model_name or EMBEDDING_MODEL_NAME
    # embeddings = OllamaEmbeddings(model=target_embedding_model)
    api_key = os.getenv("GEMINI_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model=target_embedding_model, google_api_key=api_key)
    # use NER to get the section
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings, collection_name=COLLECTION_NAME)
    lst=[{"section": x} for x in section_list]
    filter=None
    if len(section_list)==1:
        filter=lst[0]
    elif len(section_list)>1:
        filter={
            "$or": lst
        }
    
    results = []
    if(len(chunk_ids)>0):
        return db.get_by_ids(chunk_ids)
    else:
        results=db.similarity_search_with_score(
            query_text,
            k=10,
            filter=filter
        )  
    return [doc for doc, score in results]


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



def get_connection():
    return db_utils.get_db_connection(DB_NAME)

def get_genearl_context():
    todays_date=datetime.now().strftime("%Y-%m-%d")
    return f"\n\nToday's date is {todays_date} in dd-mm-yyyy format \n\n"


def generate_answer(query_text, context_docs,section_list, model_name=None,context=""):
    """Generates answer using LLM."""
    target_model_name = model_name or MODEL_NAME
    context_index_dict={
        0:[]
    }
    maximum_cv=10
    for i in range(len(section_list)):
        context_index_dict[i+1]=[]
    context_list = []
    email_group_content_dict={}
    for doc in context_docs:
        # check if email is already in the dictionary
        if doc.metadata.get("email", "Unknown") in email_group_content_dict:
            email_group_content_dict[doc.metadata.get("email", "Unknown")].append(doc)
        else:
            email_group_content_dict[doc.metadata.get("email", "Unknown")]=[doc]

    with get_connection() as conn:
        for email in email_group_content_dict:
            sql_data=db_utils.get_data_by_email(conn,email)

            candidate_data = {
                "personal_information": {
                    "name": sql_data[0]['general']['name'],
                    "email": sql_data[0]['general']['email']
                },
                "sections": []
            }
            
            for doc in email_group_content_dict[email]:
                candidate_data["sections"].append({
                    "section": doc.metadata.get('section', 'contents'),
                    "content": doc.page_content
                })
            
            context_index_dict[len(email_group_content_dict[email])].append(candidate_data)

        for keys in context_index_dict:
            if len(context_list) >= maximum_cv:
                    break
            for cv_content in context_index_dict[keys]:
                context_list.append(cv_content)
                if len(context_list) >= maximum_cv:
                    break
    
    context_text=get_genearl_context()
    context_text+="\n\n"+context+"\n\n"
    result={}
    result["candidate_list"]=context_list
    context_text += json.dumps(result, indent=4)
   
    if target_model_name=="gemini":
        content=get_data_using_gemini(query_text,PROMPT_TEMPLATE,context_text,is_json=False)
        return  content,context_text
    template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = template.format(context=context_text, question=query_text)
    
    print(f"\nGenerating answer using {target_model_name}...\n")
    model = ChatOllama(model=target_model_name)
    response = model.invoke(prompt)
    content=response.content
     #  write to a log file
    with open("log.txt", "a") as f:
        f.write(f"Query: {query_text}\n")
        f.write(f"Context: {context_text}\n")
        f.write(f"Answer: {content}\n")
    
    return content,context_text


def get_section_using_llm(question, model_name=None):
    target_model_name = model_name or MODEL_NAME
    TEMPLATE = """
    You are an expert CV analyzer.

    Your task is to determine which CV section(s) are most relevant to answer a given user question.

    Available CV sections:
    - skills : programming languages, tools, technologies
    - experience : worked at, employed, job history
    - education : education history
    - projects : built, developed, implemented, worked on a product
    - certifications : certifications obtained
    - interests : sports, hobbies, extracurricular activities
    - languages : languages known
    - general : general information like name,email,phone,place and other personal information
    - summary : summary should be from  all sections. skills, experience, education, projects, certifications, interests, languages, general
    

    Rules:
    1. Choose the MOST RELEVANT section(s).
    2. You may return multiple sections if needed.
    3. Do NOT invent new sections.
    4. filter_query → eg is interests:contains('sports') AND ((skills:contains('Android') OR experience:contains('mobile development')))
    5. Return format:
    {{
    "sections": ["section1", "section2"],
    "confidence": "high | medium | low",
    "reason": "short explanation"
    "filter_query": "section1 AND (section2 OR section3)"
    }}

    Input question:

    {question}

    before answering check this question do this question needs sections skills,experience,interest,projects,education,general information.
    """
    if target_model_name=="gemini":
        res_dict=get_data_using_gemini(question,TEMPLATE,"")
        return  res_dict
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    model = ChatOllama(model=target_model_name, format="json")
    chain = prompt | model
    response = chain.invoke({"question": question})
    content = response.content
    cleaned_content = content.strip()
    try:
        json_data = json.loads(cleaned_content)
        print(json_data)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON {e}")
        return None

def get_sql_using_llm(question,schema_text):
    TEMPLATE = """
    You are a Text-to-SQL assistant.
    Do NOT hallucinate or invent new tables or columns or try to answer if the question is not clear or not applicable to this context.

    ##Database schema:
    {context}

    ##input question:
    {question}

    Rules:
    - Use ONLY the tables and columns provided.
    - Generate ONLY valid SQLite SELECT queries.
    - Do NOT use INSERT, UPDATE, DELETE, DROP.
    - Do NOT explain anything.
    - Return ONLY the SQL query.
    - use only like operator for string matching
    - The query should match both case (Case-Insensitive)
    - Return NA if the question is not related to the database or if the question is ambiguous or cannot be answered using the database
    - split_query_list give you the query with only one condition. if the question is complex then split_query_list will have more than one query.
    -The response should be in the following format:
    - try to use select * if possible   
    
    {{
    "query": "select * from table_name where condition",
    "headers": "list of headers of the table that is selected (always should be a list)",
    "format_result":"respond with what data will it have "
    }}
    eg: {{
    "query": "SELECT u.name, u.email FROM users AS u JOIN experience AS e ON u.email = e.user_email WHERE e.company_name  LIKE '%abc%'",
    "headers": ["name","email"],
    "format_result":"This data will have the name and email of the user who has worked at abc"
    }}
    """
    if SQL_MODEL=="gemini":
        res_dict=get_data_using_gemini(question,TEMPLATE,schema_text)
        return  res_dict
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    model = ChatOllama(model=SQL_MODEL, format="json")
    chain = prompt | model
    response = chain.invoke({"question": question,"schema_text":schema_text})
    content = response.content
    cleaned_content = content.strip()
    try:
        json_data = json.loads(cleaned_content)
        print(json_data)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON {e}")

def get_data_using_llm(question,TEMPLATE,context="", model_name=None):
    target_model_name = model_name or MODEL_NAME
    if target_model_name=="gemini":
        data=get_data_using_gemini(question,TEMPLATE,context)
        return data
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    model = ChatOllama(model=target_model_name, format="json",temperature=0.0)
    chain = prompt | model
    response = chain.invoke({"question": question,"context":context})
    content = response.content
    cleaned_content = content.strip()
    try:
        json_data = json.loads(cleaned_content)
        print(json_data)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON {e}")
        return None

def get_data_using_gemini(question,TEMPLATE,context="",**args):
    is_json=args.get("is_json",True)
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    formatted_prompt = prompt.format(question=question,context=context)
    content = get_gemini_json_response(formatted_prompt) if is_json else get_gemini_response(formatted_prompt)
    
    if not content:
        return None

    cleaned_content = content.strip()
    try:
        if is_json:
            json_data = json.loads(cleaned_content)
            print(json_data)
            return json_data
        return cleaned_content
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON {e}")
        return None


def polish_question(question, model_name=None):
    target_model_name = model_name or MODEL_NAME
    TEMPLATE="""
    ## context:
    - skills : programming languages, tools, technologies
    - experience : worked at, employed, job history
    - education : education history
    - projects : built, developed, implemented, worked on a product
    - certifications : certifications obtained
    - interests : sports, hobbies, extracurricular activities
    - languages : languages known
    - general : general information like name,email,phone,place and other personal information

    You are an expert Technical Recruiter and Search Query Optimizer.
    First Find the quetsion is related to this context if not return "not related".
    If the question have name, email add it to the response.
    Your goal is to transform a raw user question into an optimized search query for a RAG (Retrieval-Augmented Generation) system that searches candidate CVs.

    Rewrite the question ONLY to make the intent explicit.
    Do not add new meaning.
    Do not remove constraints.
    Fix the grammar and spelling.
    response should be in json format with key "polished_question","names","emails", "short_description".
    Rules:
    - If the user asked about himself return "not related".
    CRITICAL RULE:
    - If the question contains a specific person name, DO NOT generalize it.
    - Preserve the person name exactly as given.
    - NEVER replace a named person with "candidates", "people", or "users".
    SELF-REFERENCE RULE:
    - If the question contains first-person references (I, me, my, myself), return "not related".
    - If the question contains a third-person name, treat it as a candidate query.
    - if the user is asking  hi,hello,how are you, what are you doing, where are you, etc return "not related"
    ENTITY EXTRACTION RULE:
    - If a person name is present, extract it into the "names" list.
    - If a person email is present, extract it into the "emails" list.
    - Don't guess name from the email.
    - Do not remove or rewrite the name from the polished question.
    Rewrite the question ONLY to improve clarity.
    - Keep the same subject.
    - Keep the same scope.
    - Do NOT generalize.
    - Do NOT pluralize.
    - Who,what, where, when, why, how are not names 
    Example:
    Input: "is steve interested in sports"
    Output:
    {{
        "polished_question": "Is steve interested in sports?",
        "names": ["steve"],
        "emails": [],
        "short_description": "Check whether steve has sports or hobby interests."
        "intents": ["sports","hobby"]
    }}
    Example:
    Input: "is steve and jhonson interested in sports"
    Output:
    {{
        "polished_question": "Is steve and jhonson interested in sports?",
        "names": ["steve","jhonson"],
        "emails": [],
        "short_description": "Check whether steve and jhonson has sports or hobby interests."
        "intents": ["sports","hobby"]
    }}
    Example:
    Input: "is steve@gmail.com interested in sports"
    Output:
    {{
        "polished_question": "Is steve@gmail.com interested in sports?",
        "names": [],
        "emails": ["steve@gmail.com"],
        "short_description": "Check whether steve@gmail.com has sports or hobby interests."
        "intents": ["sports","hobby"]
    }}
    Example:
    Input: "hi steve@gmail.com"
    Output:
    {{
        "polished_question": "not related" // if not related to the context
        "names": [],
        "emails": [],
        "short_description": "not related, it is a greeting",
        "intents": []
    }}
    Before responding, verify:
    - Is the question related to the context?. Or just a general question.
    - if general question return "not related"
    - Determine if the user is responding with a greeting.
    - if greeting return "not related"
    - The subject of the polished question matches the original subject.
    - If not, correct it.
    ##input question:
    {question}
    """
    question_dict=get_data_using_llm(question,TEMPLATE,"", model_name=target_model_name)
    names=question_dict["names"]
    emails=question_dict["emails"]
    polished_question=question_dict["polished_question"]
    # check the names and emails are present in the question
    # by seracrhing it
    names = [name for name in names if name.lower() in question.lower()]
    emails= [email for email in emails if email.lower() in question.lower()]

    question_dict["names"]=names
    question_dict["emails"]=emails
    


    return question_dict



# Example usage:
# formatted_prompt = RECRUITER_PROMPT_TEMPLATE.format(
#     context="Candidate: John Doe, Email: john@example.com", 
#     query="Give me John's email"
# )
    question_dict=get_data_using_llm(question,RECRUITER_PROMPT_TEMPLATE,"")
    names=question_dict["names"]
    emails=question_dict["emails"]
    polished_question=question_dict["polished_question"]
    # check the names and emails are present in the question
    # by seracrhing it
    # names= [name for name in names if name in question]
    # emails= [email for email in emails if email in question]

    # question_dict["names"]=names
    # question_dict["emails"]=emails
    # short_description = question_dict.get("short_description", "").lower()


    return question_dict


def check_need_more_context_needed(question,context):
    TEMPLATE = """
    You are a question analyzer.
    Your task is to determine if the question needs more context to be answered.
    ##context:
    {context}
    ##input question:
    {question}
    
    Rules:
    - if the question is clear and can be answered using the database return "False"
    - if the question is not clear or cannot be answered using the database return "True"
    - The response should be in the following format:
    {{
    "need_more_context": "True | False"
    }}
    """
    question_dict=get_data_using_llm(question,TEMPLATE,context, model_name=MODEL_NAME)
    return question_dict

if __name__ == "__main__":
    pass