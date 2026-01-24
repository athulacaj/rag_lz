
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


@tool
def get_weather(city: str) -> str:
    """Returns the weather for a specific city.
    Args:
    city (str): The name of the city.
    Example:
    get_weather("new york")
    
    Returns:
    The weather in new york is sunny.
    """
    return f"The weather in {city} is sunny."

@tool
def get_sum(a: int, b: int) -> int:
    """Returns the sum of two numbers.
    Example:
    sum(1, 2)
    
    Returns:
    3
    """
    return a + b

def text_lookup(context: str, entity: str, keywords: list[str]):
    matches = []

    for line in context.splitlines():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", line, re.IGNORECASE):
                matches.append(line.strip())

    return matches


def rule_evaluator(matches: list[str]):
    if matches:
        return "yes"
    return "no"


def answer_formatter(entity: str, result: str):
    if result == "yes":
        return f"Yes, {entity} has experience in Flutter."
    elif result == "no":
        return f"No, {entity} does not have experience in Flutter."
    return f"{entity}'s Flutter experience is unknown."








tools_list = [
    get_weather,
    get_sum
]

tools_map = {t.name: t for t in tools_list}


def execute_plan(planner_output):
    state = {}

    for step in planner_output["plan"]:
        tool = step["tool"]
        inputs = step["input"]
        result = tools_map[tool].invoke(inputs)
        state[step["output_key"]] = result

        # if tool == "get_weather":
        #     result = get_weather.invoke({
        #         "city": inputs["city"]
        #     })
        #     state["get_weather_result"] = result

    return state



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






ctx=f"""
----------------------
AVAILABLE TOOLS
----------------------
{
    "\n\n".join([json.dumps(get_tool_schema(tool)) for tool in tools_list])
}
----------------------

JSON Schema:
{json.dumps(SCHEMA)}

"""


Template="""
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

# question="what is the weather in bangalore"
question="what is 2+6"

PLANNER_OUTPUT = get_data_using_llm(question,Template,ctx)


final_state = execute_plan(PLANNER_OUTPUT)

print(final_state)
