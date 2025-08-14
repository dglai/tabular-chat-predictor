"""
Chat orchestrator for managing conversation flow using the new agent architecture.

This module contains the main orchestration logic that coordinates between
LLM interactions and agent tools using ComputeEngine for computational operations.
"""

import json
import time
from typing import List, Dict, Any, Annotated
from pathlib import Path
from jinja2 import Template
from pydantic import Field

from .base_agent import BaseAgent
from .coding_assistant import CodingAssistant
from .predicting_assistant import PredictingAssistant
from ..compute.compute_engine import ComputeEngine
from ..core.protocols import ChatUserInterface, ChatLogger
from ..core.tool_registry import ToolRegistryMixin


class Orchestrator(BaseAgent, ToolRegistryMixin):
    """Main conversation orchestrator with agent tools using ComputeEngine."""
    
    def __init__(
        self,
        compute: ComputeEngine,
        ui: ChatUserInterface,
        logger: ChatLogger,
        model: str = "openrouter/anthropic/claude-sonnet-4",
        temperature: float = 0.1
    ):
        """
        Initialize the orchestrator.
        
        Args:
            compute: ComputeEngine instance for computational operations
            ui: User interface implementation
            logger: Logger implementation
            model: LLM model to use
            temperature: Temperature for LLM responses
        """
        # Initialize BaseAgent
        BaseAgent.__init__(self, compute, ui, logger, model, temperature)
        
        # Initialize ToolRegistryMixin
        ToolRegistryMixin.__init__(self)
        
        # Load templates
        self.orchestrator_template = self._load_orchestrator_template()
        predictive_template = self._load_predictive_template()
        coding_template = self._load_coding_assistant_template()
        
        # Initialize assistant agents
        self.coding_assistant = CodingAssistant(
            compute=compute,
            ui=ui,
            logger=logger,
            template=coding_template,
            model=model,
            temperature=temperature
        )
        
        self.predicting_assistant = PredictingAssistant(
            compute=compute,
            ui=ui,
            logger=logger,
            template=predictive_template,
            model=model,
            temperature=temperature
        )
        
        # Conversation state
        self.is_running = False
        self.conversation = []
        
        self.logger.log_info("Orchestrator initialized")
    
    @ToolRegistryMixin.tool
    def coding_assistant_tool(
        self,
        instruction: Annotated[str, Field(description="Instruction for data exploration, analysis, or custom code execution")]
    ) -> Dict[str, Any]:
        """Interactive coding assistant for data analysis and exploration"""
        self.logger.log_function_call("coding_assistant", {"instruction": instruction})
        self.ui.begin_function_execution("coding_assistant", {"instruction": instruction})
        start_time = time.time()
        
        # Get metadata for template variables
        metadata = self.compute.get_metadata()
        
        result = self.coding_assistant.ask_with_repl(
            prompt=instruction,
            template_variables={
                'schema_yaml': metadata["schema_yaml"],
                'table_info': metadata["table_info"]
            }
        )
        result = {"success": True, "result": result}
        end_time = time.time()
        
        self.logger.log_function_result("coding_assistant", True, end_time - start_time, result)
        self.ui.end_function_execution(result)
        return result
    
    @ToolRegistryMixin.tool
    def predicting_assistant_tool(
        self,
        query: Annotated[str, Field(description="Predictive query requiring ML model training and predictions")]
    ) -> Dict[str, Any]:
        """Execute complete predictive workflow for ML predictions and explanations"""
        self.logger.log_function_call("predicting_assistant", {"query": query})
        self.ui.begin_function_execution("predicting_assistant", {"query": query})
        start_time = time.time()
        
        # The PredictingAssistant handles its own UI/logging internally
        result = self.predicting_assistant.handle_predictive_query(query)
        end_time = time.time()
        
        self.logger.log_function_result("predicting_assistant", result["success"], end_time - start_time, result)
        self.ui.end_function_execution(result)
        
        return result
    
    def start_conversation(self):
        """Start the main conversation loop."""
        self.is_running = True
        self.logger.log_info("Starting chat conversation")
        
        # Display initial message
        self.ui.display_message(self._get_welcome_message())

        self.conversation = []
        
        # Main conversation loop
        while self.is_running:
            user_input = self.ui.get_user_input()
            
            if user_input is None or user_input.strip().lower() in ['quit', 'exit', 'bye']:
                self.stop_conversation()
                break
            
            if not user_input.strip():
                continue
            
            self.process_user_message(user_input)
    
    def process_user_message(self, user_input: str):
        """LLM-powered orchestration with intelligent tool selection"""
        self.logger.log_info(f"Processing user message: {user_input}...")
        
        # Get metadata for prompt rendering
        metadata = self.compute.get_metadata()
        
        # Create orchestrator prompt
        orchestrator_prompt = self.orchestrator_template.render(
            query=user_input,
            schema_yaml=metadata["schema_yaml"],
            test_timestamp=metadata["test_timestamp"].strftime("%Y-%m-%d")
        )
        
        # Continue existing conversation
        self.conversation.append({"role": "user", "content": orchestrator_prompt})
        
        # Display user message
        self.ui.begin_user_message()
        self.ui.display_message(user_input)
        self.ui.end_user_message()
        
        # Multi-turn conversation with LLM tool selection
        self._execute_llm_orchestration()
    
    def _execute_llm_orchestration(self):
        """Execute LLM-driven tool orchestration"""
        self.ui.begin_assistant_message()
        
        for round_idx in range(1, 10):  # Max rounds to prevent infinite loops
            self.logger.log_debug(f"Orchestration round {round_idx}")
            
            # Get LLM response with all available tools (includes streaming)
            response = self.llm_client.get_response(self.conversation, self.tool_schemas)
            response_message = response.choices[0].message
            
            # Add assistant message to conversation
            self.conversation.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": response_message.tool_calls if response_message.tool_calls else None
            })
            
            if not response_message.tool_calls:
                # No more tool calls, orchestration complete
                self.logger.log_info("Orchestration complete - no more tool calls")
                break
            
            # Execute tool calls (high-level assistant tools)
            self.logger.log_info(f"Executing {len(response_message.tool_calls)} assistant tool call(s)")
            tool_results = self._execute_assistant_tool_calls(response_message.tool_calls)
            self.conversation.extend(tool_results)
        
        self.ui.end_assistant_message()
    
    def _execute_assistant_tool_calls(self, tool_calls):
        """Execute high-level assistant tool calls with proper UI/logging"""
        results = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            start_time = time.time()
            
            tool, func, kwargs = self.parse_tool_call(tool_call)
            result = func(**kwargs)
            
            duration = time.time() - start_time
            self.logger.log_function_result(function_name, result.get("success", True), duration, result)
            
            # Add tool result to conversation
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(result, indent=2)
            })
        
        return results
    
    def stop_conversation(self):
        """Stop the conversation loop."""
        self.is_running = False
        self.logger.log_info("Stopping chat conversation")
        self.ui.display_message("Goodbye! Feel free to return anytime to explore your data.")
    
    def _load_orchestrator_template(self) -> Template:
        """Load the orchestrator prompt Jinja template."""
        template_path = Path(__file__).parent.parent / "templates" / "orchestrator_prompt.jinja"
        with open(template_path, 'r') as f:
            template_content = f.read()
        return Template(template_content)
    
    def _load_predictive_template(self) -> Template:
        """Load the predictive query prompt Jinja template."""
        template_path = Path(__file__).parent.parent / "templates" / "predictive_query_prompt.jinja"
        with open(template_path, 'r') as f:
            template_content = f.read()
        return Template(template_content)
    
    def _load_coding_assistant_template(self) -> Template:
        """Load the coding assistant prompt Jinja template."""
        template_path = Path(__file__).parent.parent / "coding_assistant_prompt.jinja"
        with open(template_path, 'r') as f:
            template_content = f.read()
        return Template(template_content)
    
    def _get_welcome_message(self) -> str:
        """Get the welcome message for users."""
        return """Welcome! I'm your data science assistant. I can help you:

🔮 **Make Predictions**: Train ML models and predict outcomes
📊 **Explain Results**: Use SHAP to understand why predictions were made  
💻 **Explore Data**: Write custom code to analyze your database
🎯 **Answer Questions**: Help you understand your data and results

What would you like to explore in your database today?"""
    
    def get_conversation_state(self) -> Dict[str, Any]:
        """Get current conversation state for debugging/monitoring."""
        metadata = self.compute.get_metadata()
        return {
            "is_running": self.is_running,
            "has_schema": bool(metadata["schema_yaml"]),
            "model": self.model,
            "temperature": self.temperature,
            "num_assistant_tools": len(self.tool_schemas)
        }