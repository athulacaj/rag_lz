import pickle
import os
import sys
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
# Ensure 'common' directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH, DB_PATH, EMBEDDING_MODEL_NAME, MODEL_NAME,COLLECTION_NAME,DB_NAME,SQL_MODEL   
import functions.database_utils as db_utils
import json


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

def get_vector_results(query_text,section_list=[],chunk_ids=[]):
    """Retrieves documents using vector similarity."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
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


def generate_answer(query_text, context_docs,section_list):
    """Generates answer using LLM."""
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
            result=f"""
            This is the cv of {sql_data[0]["general"]["name"]}
            # Personal information
             Name: {sql_data[0]["general"]["name"]}
             Email: {sql_data[0]["general"]["email"]}
            """
            for doc in email_group_content_dict[email]:
                result+=f"\n\n# {doc.metadata.get("section", "contents")}\n\n{doc.page_content}"
            context_index_dict[len(email_group_content_dict[email])].append(result)
            # context_list.append(result)
        for keys in context_index_dict:
            if len(context_list) >= maximum_cv:
                    break
            for cv_content in context_index_dict[keys]:
                context_list.append(cv_content)
                if len(context_list) >= maximum_cv:
                    break
    

    context_text = "\n\n---\n\n".join(context_list)

   
    
    template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = template.format(context=context_text, question=query_text)
    
    print(f"\nGenerating answer using {MODEL_NAME}...\n")
    model = ChatOllama(model=MODEL_NAME)
    response = model.invoke(prompt)
    content=response.content
     #  write to a log file
    with open("log.txt", "a") as f:
        f.write(f"Query: {query_text}\n")
        f.write(f"Context: {context_text}\n")
        f.write(f"Answer: {content}\n")
    
    return content


def get_section_using_llm(question):
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
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    model = ChatOllama(model=MODEL_NAME, format="json")
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
    {schema_text}

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
    
    {{
    "query": "sql query",
    "reason": "short explanation",
    "query_object":give json object of the query. for eg:'query_object': {{'table': 'users', 'columns': ['name'], 'conditions': {{'column': 'skills', 'operator': 'LIKE', 'value': '%web app%'}}}}
    }}
    """
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

def get_data_using_llm(question,TEMPLATE,context=""):
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    model = ChatOllama(model=MODEL_NAME, format="json")
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


def polish_question(question):
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
    Example:
    Input: "is athul interested in sports"
    Output:
    {{
        "polished_question": "Is Amal interested in sports?",
        "names": ["Amal"],
        "emails": [],
        "short_description": "Check whether Amal has sports or hobby interests."
    }}
        Example:
    Input: "is athul@gmail.com interested in sports"
    Output:
    {{
        "polished_question": "Is athul@gmail.com interested in sports?",
        "names": [],
        "emails": ["athul@gmail.com"],
        "short_description": "Check whether athul@gmail.com has sports or hobby interests."
    }}
    Before responding, verify:
    - The subject of the polished question matches the original subject.
    - If not, correct it.
    ##input question:
    {question}
    """
    result=get_data_using_llm(question,TEMPLATE,"")
    print(result)
    return result

if __name__ == "__main__":
    pass