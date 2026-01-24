import re
import os
from dotenv import load_dotenv
import sys
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain.tools import tool
from common.functions.planner_utils import get_tool_schema, ToolsGroup, to_llm_json
from common.functions.query_utils import get_data_using_llm
import json
from typing import Dict, List, Any


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    """Executes the plan with support for NER bindings and chained references"""
    
    def __init__(self, tools_group: ToolsGroup):
        self.tools_group = tools_group
        self.tool_map = tools_group.get_tool_map()
    
    def resolve_input_value(self, value: Any, state: Dict, ner_output: Dict) -> Any:
        """Resolve input values from state or NER output."""
        if not isinstance(value, str):
            return value
        
        # Handle state references: $state.key_name
        if value.startswith("$state."):
            state_key = value.replace("$state.", "")
            if state_key in state:
                return state[state_key]
            else:
                raise ValueError(f"State key '{state_key}' not found. Available: {list(state.keys())}")
        
        # Handle NER references: $ner.entities[0].normalized_name
        if value.startswith("$ner."):
            try:
                # Parse path like "entities[0].normalized_name"
                path = value.replace("$ner.", "")
                
                # Simple parser for array access
                if "[" in path and "]" in path:
                    array_part, field_part = path.split("].")
                    array_name, index = array_part.split("[")
                    index = int(index)
                    
                    result = ner_output[array_name][index][field_part]
                    return result
                else:
                    # Direct field access
                    return ner_output[path]
            except (KeyError, IndexError, ValueError) as e:
                raise ValueError(f"NER path '{value}' resolution failed: {e}")
        
        return value
    
    def resolve_inputs(self, inputs: Dict, state: Dict, ner_output: Dict) -> Dict:
        """Resolve all input parameters."""
        resolved = {}
        
        for key, value in inputs.items():
            if isinstance(value, dict):
                resolved[key] = self.resolve_inputs(value, state, ner_output)
            elif isinstance(value, list):
                resolved[key] = [self.resolve_input_value(v, state, ner_output) for v in value]
            else:
                resolved[key] = self.resolve_input_value(value, state, ner_output)
        
        return resolved
    
    def execute(self, plan: Dict, ner_output: Dict) -> Dict[str, Any]:
        """Execute the plan and return state."""
        state = {}
        
        if "plan" not in plan:
            raise KeyError("Missing 'plan' key in planner output")
        
        sorted_plan = sorted(plan["plan"], key=lambda x: x.get("step", 0))
        
        print("\n" + "="*80)
        print("EXECUTION TRACE")
        print("="*80)
        
        for step in sorted_plan:
            try:
                tool_name = step.get("tool")
                inputs = step.get("input", {})
                output_key = step.get("output_key", f"step_{step.get('step', 'unknown')}")
                entity_binding = step.get("entity_binding", "N/A")
                
                if tool_name not in self.tool_map:
                    raise ValueError(f"Unknown tool: {tool_name}")
                
                # Resolve inputs from state and NER
                resolved_inputs = self.resolve_inputs(inputs, state, ner_output)
                
                # Execute tool
                result = self.tool_map[tool_name].invoke(resolved_inputs)
                state[output_key] = result
                
                print(f"\n✓ Step {step.get('step')}")
                print(f"  Entity: {entity_binding}")
                print(f"  Thought: {step.get('thought', 'N/A')}")
                print(f"  Tool: {tool_name}")
                print(f"  Inputs: {resolved_inputs}")
                print(f"  Result: {result}")
                
            except Exception as e:
                print(f"\n✗ Step {step.get('step')} FAILED: {str(e)}")
                state[output_key] = f"ERROR: {str(e)}"
        
        return state


if __name__ == "__main__":
    ner_output={'entities': [{'name': 'athul', 'type': 'PERSON', 'role': 'subject', 'normalized_name': 'athul'}], 'intent': {'action': 'get', 'target_attribute': 'skills', 'comparison_type': 'none', 'entity_count': 'single'}, 'query_type': 'simple_retrieval'}
    planner_output = {
        "plan": [
        {
        "step": 1,
        "tool": "get_skills",
        "input": {
            "candidate_name": "$ner.entities[0].normalized_name"
        },
        "output_key": "athul_skills",
        "depends_on": [],
        "thought": "The query asks for the skills of athul. Use get_skills tool to retrieve the skills for athul.",
        "entity_binding": "athul"
        }
    ],
    "answer_synthesis": {
        "format": "list",
        "state_keys_needed": [
        "athul_skills"
        ],
        "template": "The skills of athul are: $state.athul_skills"
        }
    }
    executioner_agent = ExecutionerAgent(cv_specific_tools,ner_output)
    executioner_output = executioner_agent.execute(planner_output)
    print(json.dumps(executioner_output, indent=2))