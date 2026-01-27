import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()
from common.functions.query_utils import get_data_using_llm
from langchain.tools import tool
from common.functions.planner_utils import get_tool_schema, ToolsGroup, to_llm_json
import json
from typing import Dict, List, Any
from cv_agent.tools.specific_tools import cv_specific_tools
from cv_agent.ner import NERAgent


# ============================================================================
# DATABASE
# ============================================================================
CANDIDATE_DB = {
    "athul": {
        "name": "Athul",
        "age": 25,
        "city": "Bangalore",
        "experience": "2 years",
        "skills": ["python", "java","flutter", "c++"],
        "manager": "rahul"
    },
    "amal": {
        "name": "Amal",
        "age": 22,
        "city": "Kannur",
        "experience": "1 year",
        "skills": ["python", "java", "c++"],
        "manager": "priya"
    },
    "rahul": {
        "name": "Rahul",
        "age": 35,
        "city": "Bangalore",
        "experience": "10 years",
        "skills": ["python", "leadership", "architecture"],
        "role": "Engineering Manager"
    },
    "priya": {
        "name": "Priya",
        "age": 32,
        "city": "Mumbai",
        "experience": "8 years",
        "skills": ["java", "leadership", "agile","flutter"],
        "role": "Team Lead"
    }
}





# ============================================================================
# STAGE 2: PLANNER
# ============================================================================

PLANNER_SCHEMA = {
    "plan": [
        {
            "step": 1,
            "tool": "<tool_name>",
            "input": {
                "<param_name>": "<value or $state.key_name or $ner.entities[0].normalized_name>"
            },
            "output_key": "<state_key_name>",
            "depends_on": ["<optional: list of step numbers>"],
            "thought": "<reasoning>",
            "entity_binding": "<which entity from NER this step processes>"
        }
    ],
    "answer_synthesis": {
        "format": "<single_value | comparison | list | narrative>",
        "state_keys_needed": ["<keys from execution state to use>"],
        "template": "<how to format final answer>"
    }
}


class PlannerAgent:
    """Stage 2: Create execution plan based on NER output"""
    
    def __init__(self, tools_group: ToolsGroup):
        self.tools_group = tools_group
    
    def build_planner_prompt(self, question: str, ner_output: Dict[str, Any]) -> str:
        """Build planner prompt with NER context."""
        return f"""
You are a PLANNER AGENT that creates execution plans based on NER output.

You have access to the following tools:
{to_llm_json(self.tools_group.tools_llmm_schema())}

NER OUTPUT (use this to understand the query):
{to_llm_json(ner_output)}

PLANNING RULES:
1. Use NER entities via "$ner.entities[INDEX].normalized_name"
2. For chained queries, use "$state.key_name" to reference previous step outputs
3. Each step should process ONE entity or perform ONE operation
4. Specify dependencies in "depends_on" array
5. Include "entity_binding" to track which entity each step processes
6. Define how to synthesize the final answer in "answer_synthesis"
7. Check that need multiple tools or not to answer the question

QUERY TYPE STRATEGIES:

simple_retrieval:
- Single step to get the requested attribute
- Example: Get experience → return value

multi_entity:
- One step per entity (parallel execution)
- Example: Get A's exp → Get B's exp → return both

chained_query:
- Sequential dependent steps
- Example: Get A's manager → Get manager's experience

comparison:
- Get all values first → Use comparison tool → return result

PLANNER SCHEMA:
{to_llm_json(PLANNER_SCHEMA)}

ORIGINAL QUESTION: "{question}"

Create an execution plan that:
1. Uses the entities identified in NER output
2. Follows the query type strategy
3. Includes proper chaining for dependent queries
4. Specifies how to format the final answer

Respond with ONLY valid JSON following the PLANNER_SCHEMA.
"""
    
    def create_plan(self, question: str,ner_output: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan based on NER output."""
        prompt = self.build_planner_prompt(question,ner_output)
        plan_output = get_data_using_llm(question, prompt, "")
        return plan_output




if __name__ == "__main__":
    ner_output={'entities': [{'name': 'athul', 'type': 'PERSON', 'role': 'subject', 'normalized_name': 'athul'}], 'intent': {'action': 'get', 'target_attribute': 'skills', 'comparison_type': 'none', 'entity_count': 'single'}, 'query_type': 'simple_retrieval'}
    planner_agent = PlannerAgent(cv_specific_tools)
    planner_output = planner_agent.create_plan("What are the skills of athul?",ner_output)
    print(json.dumps(planner_output, indent=2))