import json


def get_tool_schema(tool):
    tool_schema = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args  # Pydantic schema of inputs
    }
    return tool_schema


def to_llm_json(dict_obj):
    return json.dumps(dict_obj).replace("{", "{{").replace("}", "}}")

class ToolsGroup:
    def __init__(self,tools=None):
        if tools is None:
            tools = []
        self.tools = tools

    def add_tool(self, tool):
        self.tools.append(tool)

    def get_tool_schema(self):
        return [get_tool_schema(tool) for tool in self.tools]

    def get_tool_map(self):
        return {tool.name: tool for tool in self.tools}

    def tools_llmm_schema(self):
        return ([json.dumps(get_tool_schema(tool)) for tool in self.tools])


if __name__ == "__main__":
    from langchain.tools import tool
    tools = ToolsGroup()
    @tool
    def get_sum(a: int, b: int) -> int:
        """Returns the sum of two numbers.
        Example:
        sum(1, 2)
        
        Returns:
        3
        """
        return a + b
    tools.add_tool(get_sum)
    print(tools.tools_llmm_schema())
