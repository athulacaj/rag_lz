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
# TOOLS DEFINITION
# ============================================================================
@tool
def get_candidate_data(candidate_name: str) -> str:
    """Returns complete candidate data for a specific candidate.
    
    Args:
        candidate_name (str): The name of the candidate (case-insensitive).
    
    Returns:
        str: JSON string of candidate data or empty dict if not found.
    """
    normalized_name = candidate_name.lower().strip()
    result = CANDIDATE_DB.get(normalized_name, {})
    return json.dumps(result)


@tool
def get_experience(candidate_name: str) -> str:
    """Returns the years of experience for a specific candidate.
    
    Args:
        candidate_name (str): The name of the candidate (case-insensitive).
    
    Returns:
        str: Experience string (e.g., "2 years") or "unknown" if not found.
    """
    normalized_name = candidate_name.lower().strip()
    candidate = CANDIDATE_DB.get(normalized_name)
    
    if not candidate:
        return "unknown"
    
    return candidate.get("experience", "unknown")


@tool
def get_manager(candidate_name: str) -> str:
    """Returns the manager name for a specific candidate.
    
    Args:
        candidate_name (str): The name of the candidate (case-insensitive).
    
    Returns:
        str: Manager's name or "unknown" if not found.
    """
    normalized_name = candidate_name.lower().strip()
    candidate = CANDIDATE_DB.get(normalized_name)
    
    if not candidate:
        return "unknown"
    
    return candidate.get("manager", "unknown")


@tool
def compare_experience(experience_list: str) -> str:
    """Compares experience values and returns the candidate with most experience.
    
    Args:
        experience_list (str): JSON string of dict with candidate names as keys and experience as values.
    
    Returns:
        str: JSON with comparison results.
    """
    try:
        exp_dict = json.loads(experience_list)
        
        def extract_years(exp_str):
            match = re.search(r'(\d+)', exp_str)
            return int(match.group(1)) if match else 0
        
        most_exp = max(exp_dict.items(), key=lambda x: extract_years(x[1]))
        
        return json.dumps({
            "most_experienced": most_exp[0],
            "years": most_exp[1],
            "all_candidates": exp_dict
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_skills(candidate_name: str) -> str:
    """Returns the skills for a specific candidate.
    
    Args:
        candidate_name (str): The name of the candidate (case-insensitive).
    
    Returns:
        str: JSON list of skills or empty list if not found.
    """
    normalized_name = candidate_name.lower().strip()
    candidate = CANDIDATE_DB.get(normalized_name)
    
    if not candidate:
        return json.dumps([])
    
    return json.dumps(candidate.get("skills", []))


@tool
def get_candidates_with_skill(skill: str) -> List[str]:
    """Returns the candidates who have a specific skill.
    
    Args:
        skill (str): The skill to search for (case-insensitive).
    
    Returns:
        List[str]: List of candidate names who have the skill.
    """
    normalized_skill = skill.lower().strip()
    matching_candidates = []
    
    for candidate_name, candidate_data in CANDIDATE_DB.items():
        candidate_skills = candidate_data.get("skills", [])
        if normalized_skill in [s.lower() for s in candidate_skills]:
            matching_candidates.append(candidate_name)
    
    return matching_candidates


@tool
def get_candidates_with_age(age: int, operator: str) -> List[str]:
    """Returns the candidates who have a specific age.
    
    Args:
        age (int): The age to compare.
        operator (str): The operator to use for comparison (>, <, >=, <=, ==).
    
    Returns:
        List[str]: List of candidate names who have the specified age.
    """
    matching_candidates = []
    
    for candidate_name, candidate_data in CANDIDATE_DB.items():
        candidate_age = candidate_data.get("age", 0)
        if operator == ">" and candidate_age > age:
            matching_candidates.append(candidate_name)
        elif operator == "<" and candidate_age < age:
            matching_candidates.append(candidate_name)
        elif operator == ">=" and candidate_age >= age:
            matching_candidates.append(candidate_name)
        elif operator == "<=" and candidate_age <= age:
            matching_candidates.append(candidate_name)
        elif operator == "==" and candidate_age == age:
            matching_candidates.append(candidate_name)
    
    return matching_candidates

@tool
def intersection_operator(input:list[list]) -> List[str]:
    """Returns The intersection of all lists.
    
    Args:
        input (list[list]): List of lists to find intersection of.
    
    Returns:
        list: Intersection of all lists.

    Example:
    >>> intersection_operator([[1, 2, 3], [2, 3, 4], [2]])
    [2]
    """
    return list(set.intersection(*map(set, input)))

# Initialize tools group
cv_tools = ToolsGroup()
cv_tools.add_tool(get_candidate_data)
cv_tools.add_tool(get_experience)
cv_tools.add_tool(get_manager)
cv_tools.add_tool(compare_experience)
cv_tools.add_tool(get_skills)
cv_tools.add_tool(get_candidates_with_skill)
cv_tools.add_tool(get_candidates_with_age)
cv_tools.add_tool(intersection_operator)


# ============================================================================
# STAGE 1: NER (Named Entity Recognition)
# ============================================================================

NER_SCHEMA = {
    "entities": [
        {
            "name": "<entity_value>",
            "type": "<PERSON | SKILL | ATTRIBUTE | ORGANIZATION | LOCATION>",
            "role": "<subject | object | comparison_target>",
            "normalized_name": "<lowercase_canonical_form>"
        }
    ],
    "intent": {
        "action": "<get | compare | list | count | find>",
        "target_attribute": "<experience | skills | manager | data | age | city>",
        "comparison_type": "<more | less | equal | none>",
        "entity_count": "<single | multiple>"
    },
    "query_type": "<simple_retrieval | multi_entity | chained_query | comparison>"
}


class NERAgent:
    """Stage 1: Named Entity Recognition and Intent Classification"""
    
        

    @staticmethod
    def build_ner_prompt() -> str:
        """Build NER prompt template."""

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
- PERSON: Names of candidates, managers, or employees (e.g., "Athul", "Amal")
- SKILL: Technical or soft skills (e.g., "python", "leadership")
- ATTRIBUTE: Properties being queried (e.g., "experience", "skills", "manager")
- ORGANIZATION: Company names
- LOCATION: Cities, countries

INTENT ACTIONS:
- get: Retrieve specific information
- compare: Compare multiple entities
- list: List all entities matching criteria
- count: Count entities
- find: Search for entities

QUERY TYPES:
- simple_retrieval: Direct lookup (e.g., "What is Athul's experience?")
- multi_entity: Multiple entities, independent queries (e.g., "Get experience of Athul and Amal")
- chained_query: Dependent queries (e.g., "What is Athul's manager's experience?")
- comparison: Compare and rank (e.g., "Who has more experience?")

OUTPUT SCHEMA:
{to_llm_json(NER_SCHEMA)}

EXAMPLES:

Question: "What is the experience of Athul's manager?"
{to_llm_json(example1)}

Question: "Who has more experience, Athul or Amal?"
{to_llm_json(example2)}

Question: "{{question}}"

Respond with ONLY a valid JSON object following the schema above.
"""
    
    @staticmethod
    def extract_entities(question: str) -> Dict[str, Any]:
        """Extract entities and intent from question using LLM."""
        prompt = NERAgent.build_ner_prompt()
        ner_output = get_data_using_llm(question, prompt, "")
        return ner_output


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
5. For comparisons, gather all data first, then use compare_experience tool
6. Include "entity_binding" to track which entity each step processes
7. Define how to synthesize the final answer in "answer_synthesis"

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
    
    def create_plan(self, question: str, ner_output: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan based on NER output."""
        prompt = self.build_planner_prompt(question, ner_output)
        plan_output = get_data_using_llm(question, prompt, "")
        return plan_output


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


# ============================================================================
# ANSWER SYNTHESIZER
# ============================================================================

class AnswerSynthesizer:
    """Synthesizes final answer from execution state"""
    
    @staticmethod
    def synthesize(plan: Dict, state: Dict, ner_output: Dict, question: str) -> str:
        """Generate final answer based on execution state."""
        synthesis_config = plan.get("answer_synthesis", {})
        format_type = synthesis_config.get("format", "single_value")
        state_keys = synthesis_config.get("state_keys_needed", [])
        
        # Collect relevant results
        results = {key: state.get(key, "N/A") for key in state_keys}
        
        print("\n" + "="*80)
        print("ANSWER SYNTHESIS")
        print("="*80)
        print(f"Format: {format_type}")
        print(f"Using state keys: {state_keys}")
        print(f"Results: {json.dumps(results, indent=2)}")
        
        # Format based on type
        if format_type == "single_value":
            return f"Answer: {list(results.values())[0] if results else 'No result'}"
        
        elif format_type == "comparison":
            return f"Comparison Result: {json.dumps(results, indent=2)}"
        
        elif format_type == "list":
            items = [f"{k}: {v}" for k, v in results.items()]
            return "Results:\n" + "\n".join(items)
        
        elif format_type == "narrative":
            # Use LLM to create natural language answer
            return AnswerSynthesizer._generate_narrative(question, results, ner_output)
        
        return json.dumps(results, indent=2)
    
    @staticmethod
    def _generate_narrative(question: str, results: Dict, ner_output: Dict) -> str:
        """Generate natural language answer."""
        prompt = f"""
Given the question and execution results, generate a natural language answer.

Question: {question}
Execution Results: {json.dumps(results, indent=2)}
Entities: {json.dumps(ner_output.get('entities', []), indent=2)}

Provide a clear, concise answer in 1-2 sentences.
"""
        # This would call LLM - simplified here
        return f"Based on the data: {json.dumps(results, indent=2)}"


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class NERPlannerOrchestrator:
    """Main orchestrator that coordinates NER → Planner → Execution → Synthesis"""
    
    def __init__(self, tools_group: ToolsGroup):
        self.tools_group = tools_group
        self.ner_agent = NERAgent()
        self.planner_agent = PlannerAgent(tools_group)
        self.execution_engine = ExecutionEngine(tools_group)
        self.answer_synthesizer = AnswerSynthesizer()
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """Process query through full pipeline."""
        print("\n" + "="*80)
        print(f"QUERY: {question}")
        print("="*80)
        
        # Stage 1: NER
        print("\n[STAGE 1: NER]")
        ner_output = self.ner_agent.extract_entities(question)
        print(json.dumps(ner_output, indent=2))
        
        # Stage 2: Planning
        print("\n[STAGE 2: PLANNING]")
        plan = self.planner_agent.create_plan(question, ner_output)
        print(json.dumps(plan, indent=2))
        
        # Stage 3: Execution
        print("\n[STAGE 3: EXECUTION]")
        state = self.execution_engine.execute(plan, ner_output)
        
        # Stage 4: Answer Synthesis
        print("\n[STAGE 4: SYNTHESIS]")
        final_answer = self.answer_synthesizer.synthesize(plan, state, ner_output, question)
        
        return {
            "question": question,
            "ner_output": ner_output,
            "plan": plan,
            "execution_state": state,
            "answer": final_answer
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run examples through the NER + Planner pipeline."""
    
    orchestrator = NERPlannerOrchestrator(cv_tools)
    
    # Example queries
    queries = [
        # "What is the experience of Athul's manager?",
        # "Who has more experience, Athul or Amal?",
        # "What are the skills of Athul, Amal and Rahul?",
        # "Get Athul's manager's skills"
        "who have skills in java and flutter and age greater than 30"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"EXAMPLE {i}")
        print(f"{'#'*80}")
        
        result = orchestrator.process_query(query)
        
        print("\n" + "="*80)
        print("FINAL ANSWER")
        print("="*80)
        print(result["answer"])
        print("\n")


if __name__ == "__main__":
    main()