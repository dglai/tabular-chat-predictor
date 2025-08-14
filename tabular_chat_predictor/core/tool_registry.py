"""
Tool registry system with mixin support for automatic tool registration.
"""

import json
from typing import Dict, List, Tuple, Any, Optional, Callable
from .tool import Tool, make_tool


class ToolRegistry:
    """Registry for managing and executing tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def add_tool(self):
        """Decorator to add a function as a tool."""
        def decorator(func):
            tool_name = func.__name__
            tool = make_tool(func)
            self.tools[tool_name] = tool
            return func
        return decorator
    
    def register_tool(self, name: str, tool: Tool):
        """Register a tool instance."""
        self.tools[name] = tool
    
    def get_tool(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        return self.tools[name]
    
    def parse_tool_call(self, tool_call) -> Tuple[Tool, Callable, dict]:
        """Parse LiteLLM tool call and return tool, wrapped function, and parsed kwargs."""
        tool_name = tool_call.function.name
        tool = self.get_tool(tool_name)
        
        # Parse JSON arguments
        json_args = json.loads(tool_call.function.arguments or "{}")
        
        # Parse arguments using the tool's parse method to get properly typed kwargs
        parsed_kwargs = tool.parse(json_args)
        
        return tool, tool.func, parsed_kwargs
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    @property
    def schemas(self) -> List[dict]:
        """Get all tool schemas for LLM."""
        return [tool.schema for tool in self.tools.values()]


class ToolRegistryMixin:
    """Mixin providing tool registry functionality with decorator support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_registry = ToolRegistry()
        self._register_tools()
    
    def _register_tools(self):
        """Automatically register methods decorated with @tool."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '_is_tool'):
                tool_name = attr_name  # Use method name as tool name
                self.tool_registry.register_tool(tool_name, make_tool(attr))
    
    @staticmethod
    def tool(func):
        """Decorator to mark methods as tools."""
        func._is_tool = True
        return func
    
    def parse_tool_call(self, tool_call) -> Tuple[Tool, Callable, dict]:
        """Parse LiteLLM tool call and return tool, wrapped function, and parsed kwargs."""
        return self.tool_registry.parse_tool_call(tool_call)
    
    @property
    def tools(self):
        """Access to tools dictionary."""
        return self.tool_registry.tools
    
    @property
    def tool_schemas(self):
        """Get all tool schemas for LLM."""
        return self.tool_registry.schemas