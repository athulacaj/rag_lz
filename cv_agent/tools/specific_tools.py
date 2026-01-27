import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from langchain.tools import tool
from common.functions.planner_utils import get_tool_schema, ToolsGroup, to_llm_json
from typing import List, Union, Dict, Any, Literal
import json
import common.functions.database_utils as db_utils
from common.config import DB_NAME
from pydantic import BaseModel, Field
import pandas as pd
from datetime import datetime

def get_connection(db_name=None):
    return db_utils.get_db_connection(db_name)


# ============================================================================
# TOOLS DEFINITION
# ============================================================================


@tool
def get_candidate_data_by_name(candidate_name: str) -> Dict[str, Any]:
    """Returns complete candidate data for a specific candidate.
    
    
    Args:
        candidate_name (str): The name of the candidate (case-insensitive).        
    Returns:
        Returns the values like name, email, skills
        Dict[str, Any]
    Examples:
        >>> get_candidate_data_by_name("Athul")
        {"name": "Athul", "email": "athul9040@gmail.com",  "skills": ["python", "java", "flutter", "c++"]}
    """

    normalized_name = candidate_name.lower().strip()

    with get_connection(DB_NAME) as conn:
        sql_query="select * from users where name like '%{}%'".format(normalized_name)
        sql_data=db_utils.get_data_by_sql(conn,sql_query,is_dict=True)
        if(len(sql_data)>1):
            #  handle ambiguity
            return {"error": "Multiple candidates found with name: {}".format(candidate_name)}
        elif(len(sql_data)==0):
            return {"error": "No candidate found with name: {}".format(candidate_name)}
        else:
            return dict(sql_data[0])

@tool
def get_candidate_data_by_email(candidate_email: str) -> Dict[str, Any]:
    """
    Rules:
    1. Use this tool only when the email is given in the query.
    """
    """Returns complete candidate data for a specific candidate.
    
    
    Args:
        candidate_email (str): The email of the candidate (case-insensitive).
    Returns:
        Dict[str, Any]
    Examples:
        >>> get_candidate_data_by_email("athul9040@gmail.com")
        {"name": "Athul", "email": "athul9040@gmail.com",  "skills": ["python", "java", "flutter", "c++"]}
    """

    normalized_email = candidate_email.lower().strip()

    with get_connection(DB_NAME) as conn:
        sql_query="select * from users where email = {}".format(normalized_email)
        sql_data=db_utils.get_data_by_sql(conn,sql_query)
        if(len(sql_data)>1):
            #  handle ambiguity
            return {"error": "Multiple candidates found with email: {}".format(candidate_email)}
        elif(len(sql_data)==0):
            return {"error": "No candidate found with email: {}".format(candidate_email)}
        else:
            return sql_data[0]

@tool
def get_candidate_experience_from_candiate_dict(candidate_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Returns complete candidate data for a specific candidate.
    
    
    Args:
        candidate_dict (Dict[str, Any]): The candidate dictionary.
    Returns:
        List[Dict[str, Any]]: List of experience data or error message.
    """
    today = datetime.now()
    def handle_present(date_str):
        if str(date_str).strip().lower() == 'present':
            return today
        else:
            # Use 'mixed' to handle the various formats we saw earlier
            return pd.to_datetime(date_str, format='mixed')
    def format_duration(days):
        years = days // 365
        months = (days % 365) // 30
        return f"{years}y {months}m"
    normalized_email = candidate_dict.get("email").lower().strip()
    with get_connection(DB_NAME) as conn:
        # SELECT * FROM "experience" where user_email ="athul9040@gmail.com"
        sql_query="select * from experience where user_email = '{}'".format(normalized_email)
        sql_data=db_utils.get_data_by_sql(conn,sql_query,is_dict=True)
        result=[]
        if(len(sql_data)==0):
            return {"error": "No experience found with email: {}".format(normalized_email)}
        else:
            data= [dict(item) for item in sql_data]
            for item in sql_data:
                dict_item = dict(item)
                df = pd.DataFrame([dict_item])
                parsed_start_date = df['start_date'].apply(handle_present)
                parsed_end_date = df['end_date'].apply(handle_present)
                df['days_diff']=(parsed_end_date - parsed_start_date).dt.days
                df['tenure'] = df['days_diff'].apply(format_duration)
                result.append({
                    "company":dict_item['company_name'],
                    "position":dict_item['position'],
                    "tenure":df['tenure'].values[0],
                    "start_date":parsed_start_date[0].strftime('%b %Y'),
                    "end_date":parsed_end_date[0].strftime('%b %Y') if  dict_item['end_date'].lower() != 'present' else 'still working',
                })
            return result

@tool
def get_candidate_skills_from_candiate_dict(candidate_dict: Dict[str, Any]) -> List[str]:
    """Returns the skills of a candidate from the candidate dictionary.
    
    Args:
        candidate_dict (Dict[str, Any]): The candidate dictionary.
    Returns:
        List[str]: List of skills or error message.
    """
    return candidate_dict.get("skills", [])

@tool
def get_rag_data(question: str,section: str) -> Dict[str, Any]:
    """Returns the RAG data for a specific question.
    If the question is not able to handle by specific tools then use this tool.

    
    Args:
        question (str): The question to answer.
        section (str): The section of the cv to search in.Available sections are ["experience","skills","education","achievements","languages","interests","references","profile_summary"]
    Returns:
        Dict[str, Any]: The RAG data or error message.
    """
    return {"type":"rag","question":question,"section":section}

# Initialize tools group
cv_specific_tools = ToolsGroup()
cv_specific_tools.add_tool(get_candidate_data_by_email)
cv_specific_tools.add_tool(get_candidate_data_by_name)
cv_specific_tools.add_tool(get_candidate_experience_from_candiate_dict)
cv_specific_tools.add_tool(get_candidate_skills_from_candiate_dict)
# cv_specific_tools.add_tool(get_rag_data)


if __name__ == "__main__":
    # print(get_tool_schema(get_candidate_data))
    # print(get_candidate_data_by_name.invoke({"candidate_name": "athul"}))
    print(get_candidate_experience_from_candiate_dict.invoke({"candidate_dict": {"email": "athul9040@gmail.com"}}))