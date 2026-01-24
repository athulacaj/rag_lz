import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain.tools import tool
from common.functions.planner_utils import get_tool_schema, ToolsGroup, to_llm_json
from typing import Dict, List, Any
import json

# ============================================================================
# TOOLS DEFINITION
# ============================================================================
@tool
def get_candidate_data(candidate_name: str, disambiguation_id: str = None) -> str:
    """
    Get candidate data with disambiguation support
    """
    normalized_name = candidate_name.lower().strip()
    
    # Find matches
    matches = []
    for key, data in CANDIDATE_DB.items():
        if normalized_name in key.lower():
            matches.append(data)
    
    # No matches
    if len(matches) == 0:
        return json.dumps({"error": "Candidate not found"})
    
    # Single match - return it
    if len(matches) == 1:
        return json.dumps(matches[0])
    
    # Multiple matches - check if we have disambiguation
    if disambiguation_id:
        for candidate in matches:
            if (disambiguation_id.lower() in str(candidate.get('id', '')).lower() or
                disambiguation_id.lower() in candidate.get('department', '').lower() or
                disambiguation_id.lower() in candidate.get('location', '').lower()):
                return json.dumps(candidate)
    
    # Still ambiguous - return all candidates for clarification
    return json.dumps({
        "error": "Multiple candidates found",
        "candidates": [
            {
                
                "name": c.get('name'),
                "email": c.get('email')
            }
            for c in matches
        ]
    })


# Initialize tools group
cv_specific_tools = ToolsGroup()
cv_specific_tools.add_tool(get_candidate_data)
