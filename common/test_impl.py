from functions.query_utils import (
    get_section_using_llm,get_sql_using_llm,get_data_using_llm,
    get_vector_results,polish_question
)
import functions.database_utils as db_utils
from config import MODEL_NAME,DB_NAME

import json

def get_connection():
    return db_utils.get_db_connection(DB_NAME)

QUESTIONS_FOR_LLM=[
    "who can develop android app",
    "who is intrested in sports",
    "who can play football",
    "who worked on lambdazen",
    "any android developer who is intrested in sports",
    "Which of the candidates with an interest in sports have the ability to develop an Android app",
    "can athul develop web app"
]
def get_sections():
    result=[]
    for ques in QUESTIONS_FOR_LLM:
        section=get_section_using_llm(ques)
        result.append({
            "question":ques,
            "section":section
        })
    with open("test_results.json", "w") as f:
        json.dump(result, f, indent=4)
    print(result)

def get_sql(ques):
    result=[]
    with get_connection() as conn:
        schema=db_utils.get_schema(conn)
        schema_text=db_utils.schema_to_text(schema)
        section=get_sql_using_llm(ques,schema_text)
        print(section["query"])


def query_sql():
    with get_connection() as conn:
       d= db_utils.read_db_by_sql(conn,"SELECT u.name FROM users AS u JOIN experience AS e ON u.email = e.user_email WHERE e.company_name  LIKE '%LambdaZen%'")
       print(d)


def query_vector(query_text):
    section_names=["interest"]
    chunk_ids=["athul9040@gmail.com_interests"]
    results =get_vector_results(query_text,section_names,chunk_ids)
    print(results)

if __name__ == "__main__":
    # query_sql()
    # ques="who worked on lambdazen"
    # get_sql(ques)
    # query_vector("is athul and nihal interested in sports")
    r=polish_question("is athul and nihal interested in sports")
    print(r)
