import os
import sys  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.functions.query_utils import get_data_using_llm
from common.functions.planner_utils import get_tool_schema, ToolsGroup, to_llm_json
from typing import Dict, List, Any
import json

# ============================================================================
# STAGE 1: NER (Named Entity Recognition)
# ============================================================================



ENTITY_TYPES = {
    "PERSON": "Names of candidates (e.g., 'John', 'Amal')",
    "EMAIL": "Email addresses of candidates (e.g., 'john@gmail.com')",
    "SKILL": "Technical or soft skills (e.g., 'python', 'leadership')",
    "ATTRIBUTE": "Properties being queried (e.g., 'experience', 'skills', 'manager')",
    "ORGANIZATION": "Company names",
    "LOCATION": "Cities, countries"
}

ACTION_TYPES = {
    "get": "Retrieve specific information",
    "compare": "Compare multiple entities",
    "list": "List all entities matching criteria",
    "count": "Count entities",
    "find": "Search for entities"
}

TARGET_ATTRIBUTES = [
    "experience",
    "skills",
    "manager",
    "data",
    "age",
    "city"
]
COMPARISON_TYPES = [
    "more",
    "less",
    "equal",
    "none"
]
ENTITY_COUNTS = [
    "single",
    "multiple"
]

QUERY_TYPES = {
    "simple_retrieval": "Direct lookup (e.g., \"What is Athul's experience?\")",
    "multi_entity": "Multiple entities, independent queries (e.g., \"Get experience of Athul and Amal\")",
    "chained_query": "Dependent queries (e.g., \"What is Athul's manager's experience?\")",
    "comparison": "Compare and rank (e.g., \"Who has more experience?\")"
}


ROLE_TYPES = [
    "subject",
    "object",
    "comparison_target"
]

NER_SCHEMA = {
    "entities": [
        {
            "name": "<entity_value>",
            "type": "<" + " | ".join(ENTITY_TYPES.keys()) + ">",
            "role": "<" + " | ".join(ROLE_TYPES) + ">",
            "normalized_name": "<lowercase_canonical_form>"
        }
    ],
    "intent": {
        "action": "<" + " | ".join(ACTION_TYPES.keys()) + ">",
        "target_attribute": "<" + " | ".join(TARGET_ATTRIBUTES) + ">",
        "comparison_type": "<" + " | ".join(COMPARISON_TYPES) + ">",
        "entity_count": "<" + " | ".join(ENTITY_COUNTS) + ">"
    },
    "query_type": "<" + " | ".join(QUERY_TYPES.keys()) + ">"
}


class NERAgent:
    """Stage 1: Named Entity Recognition and Intent Classification"""
    
        

    @staticmethod
    def build_ner_prompt() -> str:
        """Build NER prompt template."""
        question_1 = "What is the experience of Athul's manager?"
        example1 = {
        "entities": [
            {"name": "Athul", "type": "PERSON", "role": "subject", "normalized_name": "athul"}
        ],
        "intent": {
            "action": "get",
            "target_attribute": "experience",
            "comparison_type": "none",
            "entity_count": "single"
        },
        "query_type": "chained_query"
    }
    
        question_2 = "Who has more experience, Athul or Amal?"
        example2 = {
            "entities": [
                {"name": "Athul", "type": "PERSON", "role": "comparison_target", "normalized_name": "athul"},
                {"name": "Amal", "type": "PERSON", "role": "comparison_target", "normalized_name": "amal"}
            ],
            "intent": {
                "action": "compare",
                "target_attribute": "experience",
                "comparison_type": "more",
                "entity_count": "multiple"
            },
            "query_type": "comparison"
        }

        return f"""
You are a Named Entity Recognition (NER) Agent specializing in HR/Recruitment queries.

Your task is to:
1. Extract all entities (people, skills, attributes) from the question
2. Classify the intent and query type
3. Normalize entity names to lowercase canonical forms

ENTITY TYPES:
{"\n".join([f"- {k}: {v}" for k, v in ENTITY_TYPES.items()])}

INTENT ACTIONS:
{"\n".join([f"- {k}: {v}" for k, v in ACTION_TYPES.items()])}

QUERY TYPES:
{"\n".join([f"- {k}: {v}" for k, v in QUERY_TYPES.items()])}

OUTPUT SCHEMA:
{to_llm_json(NER_SCHEMA)}

EXAMPLES:

Question: {question_1} 
{to_llm_json(example1)}

Question: {question_2}
{to_llm_json(example2)}

Question: "{{question}}"

Respond with ONLY a valid JSON object following the schema above.
"""
    
    


    @staticmethod
    def ner_validation(context,ner_output: Dict[str, Any]) -> bool:
        """Validate NER output using code."""
        # TODO: Implement validation using code.
        return True

    @staticmethod
    def extract_entities(question: str) -> Dict[str, Any]:
        """Extract entities and intent from question using LLM."""
        prompt = NERAgent.build_ner_prompt()
        ner_output = get_data_using_llm(question, prompt, "")
        is_valid = NERAgent.ner_validation(question,ner_output)
        if is_valid:
            return ner_output
        else:
            return None



if __name__ == "__main__":
    ner_agent = NERAgent()
    ner_output = ner_agent.extract_entities("What are the skills of athul?")
    print(json.dumps(ner_output, indent=2))