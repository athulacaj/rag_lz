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
from tools.specific_tools import cv_specific_tools
from ner import NERAgent
from planner import PlannerAgent
from executioner import ExecutionEngine
from synthesizer import AnswerSynthesizer

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
    
    orchestrator = NERPlannerOrchestrator(cv_specific_tools)
    
    # Example queries
    queries = [
        # "What is the experience of Athul's manager?",
        "Who has more experience, Athul or Amal?",
        # "What are the skills of Athul, Amal and Rahul?",
        # "Get Athul's manager's skills"
        # "who have skills in java and flutter and age greater than 30"
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