# Using the Tool System

This document explains how to use the Tool class, ToolRegistry class, and ToolRegistryMixin to create LLM-compatible tools with automatic JSON schema generation.

NOTE: LiteLLM has a `litellm.utils.function_to_dict` function, but it has [multiple](https://github.com/BerriAI/litellm/issues/4250) [issues](https://github.com/BerriAI/litellm/issues/9323). So I'm creating my own.

## Overview

The tool system converts Python functions into tools that can be called by Large Language Models through function calling. It automatically:

- Generates JSON schemas from Python function signatures
- Validates and parses JSON arguments into Python types
- Handles complex type annotations including Pydantic models, enums, and optional types

## Tool Class

The `Tool` class wraps a Python function and provides JSON schema generation and argument parsing.

### Creating a Tool

```python
from tabular_chat_predictor.core.tool import make_tool

def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

# Create a tool
tool = make_tool(calculate_sum)

# Access the JSON schema
print(tool.schema)

# Call with JSON arguments
result = tool({"a": 5, "b": 3})  # Returns 8
```

## ToolRegistry Class

The `ToolRegistry` class manages collections of tools and provides integration with LiteLLM.

```python
from tabular_chat_predictor.core.tool_registry import ToolRegistry

# Create registry
registry = ToolRegistry()

# Add tools using decorator
@registry.add_tool()
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

# Get all schemas for LLM
schemas = registry.schemas

# Parse LiteLLM tool call
tool, func, kwargs = registry.parse_tool_call(tool_call)
result = func(**kwargs)
```

## ToolRegistryMixin

The `ToolRegistryMixin` provides decorator-based tool registration for classes. This is the recommended approach.

```python
from tabular_chat_predictor.core.tool_registry import ToolRegistryMixin
from typing import Annotated
from pydantic import Field

class Calculator(ToolRegistryMixin):
    def __init__(self):
        super().__init__()  # Important: call super().__init__()
    
    @ToolRegistryMixin.tool
    def add(
        self,
        a: Annotated[int, Field(description="First number")],
        b: Annotated[int, Field(description="Second number")]
    ) -> int:
        """Add two numbers."""
        return a + b
    
    @ToolRegistryMixin.tool
    def divide(
        self,
        dividend: Annotated[float, Field(description="Number to divide")],
        divisor: Annotated[float, Field(description="Number to divide by")]
    ) -> float:
        """Divide two numbers."""
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return dividend / divisor

# Usage
calc = Calculator()
schemas = calc.tool_schemas  # Get schemas for LLM
```

## Function Annotations

Functions must have proper type annotations. The system supports enhanced annotations using Pydantic `Field`.

### Basic Annotations

```python
def basic_function(name: str, age: int, active: bool) -> str:
    """Function with basic type annotations."""
    return f"{name} is {age} years old"
```

### Enhanced Annotations with Field

```python
from typing import Annotated, Optional, List
from pydantic import Field

def enhanced_function(
    name: Annotated[str, Field(description="Person's full name")],
    age: Annotated[int, Field(description="Age in years")],
    tags: Annotated[List[str], Field(description="List of tags")] = [],
    notes: Optional[Annotated[str, Field(description="Additional notes")]] = None
) -> dict:
    """Function with enhanced Field annotations."""
    return {"name": name, "age": age, "tags": tags, "notes": notes}
```

## Supported Type Annotations

### Basic Types
- `int` → `{"type": "integer"}`
- `str` → `{"type": "string"}`
- `float` → `{"type": "number"}`
- `bool` → `{"type": "boolean"}`

### Lists
```python
def list_example(items: List[str]) -> str:
    pass
# Generates: {"type": "array", "items": {"type": "string"}}
```

### Optional Types
```python
def optional_example(required: str, optional: Optional[str] = None) -> str:
    pass
# Optional parameters are excluded from "required" list
```

### Enums
```python
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"

def enum_example(priority: Priority) -> str:
    pass
# Generates: {"type": "string", "enum": ["low", "medium", "high"]}
```

### Literal Types
```python
from typing import Literal

def literal_example(mode: Literal["read", "write"]) -> str:
    pass
# Generates: {"type": "string", "enum": ["read", "write"]}
```

### Pydantic Models
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str = Field(description="User name")
    age: int = Field(description="User age")
    email: Optional[str] = None

def model_example(user: User) -> str:
    pass
# Generates full object schema with properties and required fields
```

## Integration with LiteLLM

```python
import litellm
import json

class MyService(ToolRegistryMixin):
    def __init__(self):
        super().__init__()
    
    @ToolRegistryMixin.tool
    def search(
        self,
        query: Annotated[str, Field(description="Search query")],
        limit: Annotated[int, Field(description="Max results")] = 10
    ) -> dict:
        """Search for items."""
        return {"results": [], "count": 0}

# Create service
service = MyService()

# Use with LiteLLM
messages = [{"role": "user", "content": "Search for products"}]

response = litellm.completion(
    model="gpt-4",
    messages=messages,
    tools=service.tool_schemas,  # Pass tool schemas
    tool_choice="auto"
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        # Parse and execute tool call
        tool, func, kwargs = service.parse_tool_call(tool_call)
        result = func(**kwargs)
        
        # Add result to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(result)
        })
```

## Complete Example

```python
from tabular_chat_predictor.core.tool_registry import ToolRegistryMixin
from typing import Annotated, Optional, List
from pydantic import Field, BaseModel
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"

class Task(BaseModel):
    title: str = Field(description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Task status")

class TaskManager(ToolRegistryMixin):
    def __init__(self):
        self.tasks = []
        super().__init__()
    
    @ToolRegistryMixin.tool
    def create_task(
        self,
        title: Annotated[str, Field(description="Task title")],
        description: Annotated[Optional[str], Field(description="Task description")] = None
    ) -> dict:
        """Create a new task."""
        task = Task(title=title, description=description)
        self.tasks.append(task)
        return {"success": True, "task": task.dict()}
    
    @ToolRegistryMixin.tool
    def list_tasks(
        self,
        status: Annotated[Optional[TaskStatus], Field(description="Filter by status")] = None
    ) -> dict:
        """List all tasks, optionally filtered by status."""
        filtered_tasks = self.tasks
        if status:
            filtered_tasks = [t for t in self.tasks if t.status == status]
        
        return {
            "tasks": [task.dict() for task in filtered_tasks],
            "count": len(filtered_tasks)
        }

# Usage
manager = TaskManager()
schemas = manager.tool_schemas  # Ready for LiteLLM
```

This covers the essential functionality for using the Tool class, ToolRegistry, and ToolRegistryMixin with proper annotations and LiteLLM integration.