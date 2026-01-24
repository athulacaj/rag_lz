
from typing import Dict, List, Any
import json


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



