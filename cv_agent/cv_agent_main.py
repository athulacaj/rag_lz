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
from cv_agent.tools.specific_tools import cv_specific_tools
from cv_agent.ner import NERAgent
from cv_agent.planner import PlannerAgent
from cv_agent.executioner import ExecutionEngine
from cv_agent.synthesizer import AnswerSynthesizer
import logging
logger = logging.getLogger('rag_logger')

from common.functions.query_utils import (
    generate_answer,
    check_need_more_context_needed
)

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
        logger.info("\n" + "="*80)
        logger.info(f"QUERY: {question}")
        logger.info("="*80)
        
        # Stage 1: NER
        logger.info("\n[STAGE 1: NER]")
        ner_output = self.ner_agent.extract_entities(question)
        logger.info(json.dumps(ner_output, indent=2))
        
        # Stage 2: Planning
        logger.info("\n[STAGE 2: PLANNING]")
        plan = self.planner_agent.create_plan(question, ner_output)
        i=1
        for tool in plan["plan"]:
            logger.info(f"tool no: {i}")
            logger.info(f"tool name: {tool['tool']}")
            logger.info(f"thought: {tool['thought']}")
            i+=1
        # logger.info(json.dumps(plan, indent=2))
        
        # Stage 3: Execution
        logger.info("\n[STAGE 3: EXECUTION]")
        state = self.execution_engine.execute(plan, ner_output)
        
        # Stage 4: Answer Synthesis
        logger.info("\n[STAGE 4: SYNTHESIS]")
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

def cv_agent_query(query, model_name=None, embedding_model=None, parser=None, db_name=None):
    """Run examples through the NER + Planner pipeline."""
    
    orchestrator = NERPlannerOrchestrator(cv_specific_tools)
    
    result = orchestrator.process_query(query)
    answer,context_text = generate_answer(query, [],[], model_name=model_name,context=json.dumps(result["answer"]))
    return answer,context_text


if __name__ == "__main__":
    answer,context_text = cv_agent_query("do  nihal have any carrier gap")
    print(answer)
    print(context_text)