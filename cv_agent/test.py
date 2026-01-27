from typing import Dict, Any, List, Tuple
import re
from tools.specific_tools import (
    get_candidate_data_by_name,
    get_candidate_data_by_email,
    search_candidates_by_skills,
    count_candidates_by_criteria,
    get_all_candidates_summary,
    generic_database_query,
)

class CandidateQueryRouter:
    """Routes natural language queries to appropriate tools."""
    
    def __init__(self):
        self.tool_mappings = {
            'single_candidate_by_name': get_candidate_data_by_name,
            'single_candidate_by_email': get_candidate_data_by_email,
            'search_by_skills': search_candidates_by_skills,
            'count_candidates': count_candidates_by_criteria,
            'all_candidates': get_all_candidates_summary,
            'generic_query': generic_database_query,
        }
    
    def route_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Routes a query to the appropriate tool and extracts parameters.
        
        Returns:
            Tuple[str, Dict[str, Any]]: (tool_name, parameters)
        """
        query_lower = query.lower().strip()
        
        # Pattern 1: Specific candidate by name (not email-like)
        if self._is_single_candidate_query(query_lower) and '@' not in query:
            candidate_name = self._extract_candidate_name(query)
            return ('single_candidate_by_name', {'candidate_name': candidate_name})
        
        # Pattern 2: Specific candidate by email
        if '@' in query:
            email = self._extract_email(query)
            if email:
                return ('single_candidate_by_email', {'candidate_email': email})
        
        # Pattern 3: Count queries ("how many")
        if any(phrase in query_lower for phrase in ['how many', 'count', 'number of']):
            skills = self._extract_skills(query)
            return ('count_candidates', {'skills': skills if skills else None})
        
        # Pattern 4: Search/filter by skills
        if self._is_skills_search_query(query_lower):
            skills = self._extract_skills(query)
            match_all = 'and' in query_lower or 'both' in query_lower
            return ('search_by_skills', {
                'skills': skills,
                'match_all': match_all
            })
        
        # Pattern 5: List all / show all
        if any(phrase in query_lower for phrase in ['all candidates', 'list all', 'show all']):
            return ('all_candidates', {})
        
        # Pattern 6: Generic fallback
        return ('generic_query', {'natural_language_query': query})
    
    def _is_single_candidate_query(self, query: str) -> bool:
        """Check if query is about a single specific candidate."""
        single_indicators = [
            'does', 'do', 'is', 'has', 'have',
            'tell me about', 'show me', 'find', 'get',
            'what are', 'what is'
        ]
        # Check if query starts with these indicators
        for indicator in single_indicators:
            if query.startswith(indicator):
                return True
        return False
    
    def _is_skills_search_query(self, query: str) -> bool:
        """Check if query is searching for candidates by skills."""
        search_indicators = [
            'who has', 'who have', 'candidates with',
            'people with', 'users with', 'find candidates',
            'search for', 'with skills', 'having skills'
        ]
        return any(indicator in query for indicator in search_indicators)
    
    def _extract_candidate_name(self, query: str) -> str:
        """Extract candidate name from query."""
        # Remove common question words
        remove_words = [
            'does', 'do', 'is', 'has', 'have', 'tell me about',
            'show me', 'find', 'get', 'what are', 'what is',
            'the candidate', 'candidate', 'skills', 'skill',
            'experience', 'in', 'for', '?'
        ]
        
        cleaned = query.lower()
        for word in remove_words:
            cleaned = cleaned.replace(word, ' ')
        
        # Extract first meaningful word (likely the name)
        words = [w.strip() for w in cleaned.split() if w.strip()]
        if words:
            return words[0].capitalize()
        return ""
    
    def _extract_email(self, query: str) -> str:
        """Extract email from query."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, query)
        return match.group(0) if match else ""
    
    def _extract_skills(self, query: str) -> List[str]:
        """Extract skills from query."""
        # Common skill keywords
        skill_keywords = {
            'python': ['python', 'py'],
            'java': ['java'],
            'javascript': ['javascript', 'js'],
            'typescript': ['typescript', 'ts'],
            'c++': ['c++', 'cpp'],
            'mobile development': ['mobile development', 'mobile dev', 'mobile'],
            'flutter': ['flutter'],
            'react': ['react', 'reactjs'],
            'angular': ['angular'],
            'node.js': ['node', 'nodejs', 'node.js'],
            'web development': ['web development', 'web dev'],
            'backend': ['backend', 'back-end'],
            'frontend': ['frontend', 'front-end'],
        }
        
        found_skills = []
        query_lower = query.lower()
        
        for skill, keywords in skill_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    found_skills.append(skill)
                    break
        
        return list(set(found_skills))  # Remove duplicates
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Routes and executes a query, returning the results.
        
        Args:
            query (str): Natural language query
            
        Returns:
            Dict[str, Any]: Query results
        """
        tool_name, params = self.route_query(query)
        tool_function = self.tool_mappings.get(tool_name)
        
        if not tool_function:
            return {
                "error": f"Unknown tool: {tool_name}",
                "query": query
            }
        
        try:
            result = tool_function(**params)
            return {
                "tool_used": tool_name,
                "parameters": params,
                "result": result,
                "success": True
            }
        except Exception as e:
            return {
                "tool_used": tool_name,
                "parameters": params,
                "error": str(e),
                "success": False
            }


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    router = CandidateQueryRouter()
    
    # Test queries
    test_queries = [
        "Does Athul have skills in mobile development",
        "How many users have skills in mobile development",
        "Find candidates with Python and Java",
        "athul9040@gmail.com",
        "Who has experience in Flutter?",
        "Show me all candidates",
        "List candidates from Google",
        "Count candidates with backend skills",
    ]
    
    print("=" * 80)
    print("QUERY ROUTING EXAMPLES")
    print("=" * 80)
    
    for query in test_queries:
        tool_name, params = router.route_query(query)
        print(f"\nQuery: {query}")
        print(f"  → Tool: {tool_name}")
        print(f"  → Params: {params}")
    
    print("\n" + "=" * 80)
    print("EXECUTION EXAMPLES")
    print("=" * 80)
    
    # Example execution
    result = router.execute_query("How many users have skills in Python")
    print(f"\nQuery: 'How many users have skills in Python'")
    print(f"Result: {result}")


# ==================== INTEGRATION WITH LangGraph/LangChain ====================

class CandidateQueryAgent:
    """
    Agent that can be integrated with LangGraph or LangChain.
    """
    
    def __init__(self):
        self.router = CandidateQueryRouter()
        self.conversation_history = []
    
    def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Process a query and return a natural language response.
        
        Args:
            query (str): User's query
            context (Dict[str, Any]): Optional context from conversation
            
        Returns:
            str: Natural language response
        """
        # Execute the query
        result = self.router.execute_query(query)
        
        # Store in history
        self.conversation_history.append({
            "query": query,
            "result": result,
            "timestamp": datetime.now()
        })
        
        # Format response
        if not result.get("success", False):
            return f"I encountered an error: {result.get('error', 'Unknown error')}"
        
        # Generate natural language response based on result
        return self._format_response(result)
    
    def _format_response(self, result: Dict[str, Any]) -> str:
        """Format the result into a natural language response."""
        tool_used = result.get("tool_used")
        data = result.get("result", {})
        
        if tool_used == "single_candidate_by_name":
            if "error" in data:
                return f"I couldn't find that candidate: {data['error']}"
            return f"Found {data.get('name')}! Email: {data.get('email')}, Skills: {', '.join(data.get('skills', []))}"
        
        elif tool_used == "count_candidates":
            count = data.get("count", 0)
            criteria = data.get("criteria", {})
            skills = criteria.get("skills", [])
            if skills:
                return f"I found {count} candidate(s) with {', '.join(skills)} skills."
            return f"Total candidates in database: {count}"
        
        elif tool_used == "search_by_skills":
            count = data.get("count", 0)
            candidates = data.get("candidates", [])
            if count == 0:
                return "No candidates found matching those skills."
            
            response = f"Found {count} candidate(s):\n"
            for candidate in candidates[:5]:  # Show first 5
                response += f"  • {candidate.get('name')} ({candidate.get('email')})\n"
            
            if count > 5:
                response += f"  ... and {count - 5} more"
            return response
        
        elif tool_used == "generic_query":
            if "error" in data:
                return f"{data.get('error')}. {data.get('suggestion', '')}"
            return str(data.get("results", "Query executed successfully"))
        
        return "Query executed successfully!"