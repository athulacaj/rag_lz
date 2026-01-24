
import re
import os
from dotenv import load_dotenv
import sys
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain.tools import tool
from common.functions.planner_utils import get_tool_schema
from common.functions.query_utils import (
    get_data_using_llm
)
import json
from common.functions.planner_utils import ToolsGroup

CANDIDATE_DB={
        "athul": {
            "name": "Athul",
            "age": 25,
            "city": "Bangalore",
            "experience": "2 years",
            "skills": ["python", "java", "c++"]
        },
        "amal": {
            "name": "Amal",
            "age": 22,
            "city": "Kannur",
            "experience": "1 year",
            "skills": ["python", "java", "c++"]
        }
}


@tool
def get_candidate_data(candidate_name: str) -> str:
    """Returns the candidate data for a specific candidate.
        Args:
            candidate_name (str): The name of the candidate.
        Returns:
            str: The candidate data.
        Example:
            get_candidate_data("athul")
            Returns:
                {
                    "name": "Athul",
                    "age": 25,
                    "city": "Bangalore",
                    "experience": "2 years",
                    "skills": ["python", "java", "c++"]
                }
    """
  
    return CANDIDATE_DB.get(candidate_name, "")

@tool
def get_experience(candidate_name: str) -> str:
    """Returns the years of experience for a specific candidate.
    
    Args:
        candidate_name (str): The name of the candidate (case-insensitive).
    
    Returns:
        str: Experience string (e.g., "2 years") or "unknown" if not found.
    
    Example:
        get_experience("athul")
        Returns: "2 years"
    """
    normalized_name = candidate_name.lower().strip()
    candidate = CANDIDATE_DB.get(normalized_name)
    
    if not candidate:
        return "unknown"
    
    return candidate.get("experience", "unknown")



cv_tools=ToolsGroup()
cv_tools.add_tool(get_candidate_data)
cv_tools.add_tool(get_experience)





SCHEMA={
  "intent": "<string: what the user wants to know>",
  "entities": [
    {
      "name": "<entity name>",
      "type": "<person | company | skill | unknown>"
    }
  ],
  "plan": [
    {
      "step": 1,
      "tool": "<tool_name>",
      "input": {
        "<param_name>": "<value or reference>"
      },
      "output_key": "<state_key_name>",
      "thought": "<reasoning>"
    }
  ],
  "constraints": {
    "entity_isolation": True,
    "allow_inference": False,
    "missing_data_policy": "unknown"
  }
}


context=f"""
----------------------
AVAILABLE TOOLS
----------------------
{cv_tools.tools_llmm_schema()}

------------------------------

----------------------

JSON Schema:
{json.dumps(SCHEMA)}

------------------------------

"""




def execute_plan(planner_output):
    state = {}

    for step in planner_output["plan"]:
        tool = step["tool"]
        inputs = step["input"]
        result = cv_tools.get_tool_map()[tool].invoke(inputs)
        state[step["output_key"]] = result
    return state

TEMPLATE="""
You are a PLANNER AGENT.

Your task is to create a step-by-step plan using ONLY the tools defined below.
You must NOT answer the question directly.


PLANNING RULES
1. Never infer facts without retrieval
2. Always retrieve before answering
3. One candidate per retrieval
4. If retrieval returns empty, mark result as "no" or "unknown"
5. Output ONLY JSON
6: Reasoning Plan Define how each attribute will be derivedSpecify calculation steps if needed
7: Answer Assembly:Define how final response will be structured, Preserve entity isolation

{context}

---------------------------
Question:
{question}
"""


question="how many years of experience does athul and amal have"

PLANNER_OUTPUT = get_data_using_llm(question,TEMPLATE,context)

print(PLANNER_OUTPUT)

r=execute_plan(PLANNER_OUTPUT)

print("--------------------------- answer is ---------------------------")
print(r)



