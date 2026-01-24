# https://docs.langchain.com/oss/python/langchain/agents

import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("GEMINI_KEY")
if not api_key:
    raise ValueError("GEMINI_KEY not found in environment variables.")


from langchain.agents import create_agent
from langchain.tools import tool

@tool("web_search")  # Custom name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name) 

@tool
def get_weather(city: str) -> str:
    """Returns the weather for a specific city."""
    return f"The weather in {city} is sunny."

# This automatically generates the JSON schema for the planner
print(get_weather.name)
print(get_weather.args)

# 1. Extract the schema using LangChain's built-in conversion
# LangChain tools convert easily to OpenAI-format function definitions
tool_schema = [
    {
        "name": get_weather.name,
        "description": get_weather.description,
        "parameters": get_weather.args  # Pydantic schema of inputs
    }
]

# 2. Inject this schema into your System Prompt
planner_system_prompt = f"""
You are a Planner Agent.
...
🔹 Available Tools
{json.dumps(tool_schema, indent=2)}
...
"""

print(planner_system_prompt)
