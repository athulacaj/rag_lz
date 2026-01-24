<!-- https://www.youtube.com/watch?v=yS_hwnJusDk -->

# Planner

There are several robust Python libraries designed specifically to bridge the gap between Python functions and LLM "planner" prompts.
    1. LangChain (Most Popular)
    2. Microsoft Semantic Kernel (Best for "Planners")
    3. Instructor (Best for Pydantic/Raw Control)


from langchain.agents import tool

@tool
def get_weather(city: str) -> str:
    """Returns the weather for a specific city."""
    return f"The weather in {city} is sunny."

# This automatically generates the JSON schema for the planner
print(get_weather.name)
print(get_weather.args)



🏗 Recommended CV-RAG Architecture (Final)


User Question
   ↓
NER + Intent Extraction
   ↓
Entity Validation (DB / Index check)
   ↓
Planner (JSON plan)
   ↓
Executor
   ↓
Retrievers / Tools (Vector DB, SQL, APIs)
   ↓
State
   ↓
Answer Assembler


🔑 Golden Rule (Remember This)

Planner answers “HOW do I get the answer?”
Retriever answers “WHERE is the data?”

Never swap them.

🎯 Practical Tip for You

Since you’re building:

CV RAG

Multiple candidates

Deterministic answers

👉 Always run planner first.
Vector DB should be treated as just another tool.