"""
Refactored coding assistant that uses ComputeEngine and BaseAgent.

This module adapts the ask_with_repl logic to work with the new architecture
using ComputeEngine for computational operations and BaseAgent for common functionality.
"""

import json
from typing import Dict, Any, Annotated
from jinja2 import Template
from pydantic import Field

from .base_agent import BaseAgent
from ..compute.compute_engine import ComputeEngine
from ..core.protocols import ChatUserInterface, ChatLogger
from ..core.tool import make_tool
from ..core.tool_registry import ToolRegistryMixin


class CodingAssistant(BaseAgent, ToolRegistryMixin):
    """Coding assistant that uses ComputeEngine and protocol interfaces."""
    
    def __init__(
        self,
        compute: ComputeEngine,
        ui: ChatUserInterface,
        logger: ChatLogger,
        template: Template,
        model: str = "openrouter/anthropic/claude-sonnet-4",
        temperature: float = 0.1
    ):
        """
        Initialize the coding assistant.
        
        Args:
            compute: ComputeEngine instance for computational operations
            ui: User interface for displaying results
            logger: Logger for recording operations
            template: Jinja2 template for generating system prompts
            model: LLM model to use
            temperature: Temperature for LLM responses
        """
        # Initialize BaseAgent
        BaseAgent.__init__(self, compute, ui, logger, model, temperature)
        
        # Initialize ToolRegistryMixin
        ToolRegistryMixin.__init__(self)
        
        self.template = template
    
    @ToolRegistryMixin.tool
    def run_python(
        self,
        code: Annotated[str, Field(description="Python code to execute. Use print() to show output. Variables persist across executions.")]
    ) -> Dict[str, Any]:
        """Execute Python code in the persistent IPython environment. Variables and results persist across calls."""
        # Display the code that will be executed
        self.ui.display_code_execution(code)
        self.logger.log_info(f"Executing Python code: {code[:100]}...")
        
        # Execute code using ComputeEngine
        result = self.compute.run_python_code(code)
        
        # Display the execution result
        if result["success"]:
            if result["output"]:
                self.ui.display_code_output(result["output"])

            # Display any figures that were generated
            if result.get("figures"):
                for fig_info in result["figures"]:
                    if fig_info["type"] == "matplotlib":
                        self.ui.display_chart(
                            fig_info["figure_object"],
                            fig_info.get("title", "Generated Chart")
                        )
                        del fig_info["figure_object"]  # Remove to allow JSON serialization
            
            self.logger.log_info("Python code executed successfully")
        else:
            self.ui.display_error(f"Code execution failed: {result['error']}")
            self.logger.log_error(f"Python code execution failed: {result['error']}")
        
        return result
    
    def ask_with_repl(
        self,
        prompt: str,
        max_rounds: int = 20,
        template_variables: Dict[str, Any] = None
    ) -> str:
        """
        Coding assistant agent with persistent IPython REPL using ComputeEngine.
        
        Args:
            prompt: User's question/request
            max_rounds: Maximum conversation rounds
            template_variables: Variables to be rendered in the prompt template
            
        Returns:
            Final assistant response after tool execution
        """
        self.logger.log_info(f"Starting coding assistant with prompt: {prompt[:100]}...")
        
        # Display initialization message
        self.ui.update_tool_status("Starting coding assistant with ComputeEngine...")
        
        # Prepare system message using template
        render_vars = template_variables or {}
        render_vars['user_prompt'] = prompt
        user_message = self.template.render(**render_vars)
        
        # Set up conversation
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        tools = self.tool_schemas
        
        self.logger.log_info(f"Starting coding assistant conversation with {self.model}")
        
        # Main conversation loop
        for round_idx in range(1, max_rounds + 1):
            # Display current round
            self.ui.update_tool_status(f"Coding assistant: Round {round_idx} (max {max_rounds})")
            self.logger.log_debug(f"Coding assistant round {round_idx}")
            
            # Get assistant response with streaming
            response = self.llm_client.get_response(messages, tools)
            
            # Check if assistant wants to call tools
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                # No more tool calls, conversation complete
                final_response = response.choices[0].message.content
                self.logger.log_info("Coding assistant conversation complete")
                return final_response
            
            # Add the assistant message that requested tools
            messages.append(response.choices[0].message)
            
            # Execute each tool call
            self.logger.log_info(f"Executing {len(tool_calls)} tool call(s)")
            
            for tool_call in tool_calls:
                tool, func, kwargs = self.parse_tool_call(tool_call)
                result = func(**kwargs)
                
                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(result, indent=2)
                })
            if round_idx == max_rounds - 1:
                messages.append({
                    "role": "system",
                    "content": "You have ran 20 rounds without completing the task. Please summarize what you have done so far."
                })
                tools = []
        
        # Max rounds reached
        warning_msg = f"Conversation reached maximum {max_rounds} rounds without completion."
        self.logger.log_info(warning_msg)
        return warning_msg 