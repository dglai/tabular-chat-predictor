"""
LLM client for handling communication with language models.

This module provides a frontend-agnostic way to interact with LLMs
with streaming support and proper logging.
"""

import time
from typing import List, Dict, Any, Generator
import litellm

from ..core.protocols import ChatUserInterface, ChatLogger


class LLMClient:
    """Frontend-agnostic LLM client with streaming support."""
    
    def __init__(
        self,
        model: str,
        temperature: float,
        ui: ChatUserInterface,
        logger: ChatLogger
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: LLM model to use
            temperature: Temperature for LLM responses
            ui: User interface for displaying responses
            logger: Logger for recording LLM interactions
        """
        self.model = model
        self.temperature = temperature
        self.ui = ui
        self.logger = logger
    
    def get_response(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None
    ) -> Any:
        """
        Get streaming response from LLM with function calling support.
        
        Args:
            messages: Conversation messages
            tools: Available tools/functions for function calling (optional)
            
        Returns:
            LLM response object
        """
        start_time = time.time()
        
        # Log the request
        self.logger.log_llm_request(self.model, messages, tools)
        
        chunks = []
        
        def response_generator() -> Generator[str, None, None]:
            nonlocal chunks
            
            # Prepare completion parameters
            completion_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "stream": True,
            }
            
            # Only add tool-related parameters if tools are provided
            if tools:
                completion_params["tools"] = tools
                completion_params["tool_choice"] = "auto"
            
            for chunk in litellm.completion(**completion_params):
                chunks.append(chunk)
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        
        # Stream through UI interface
        complete_content = self.ui.display_streaming_message(response_generator())
        
        # Build complete response
        response = litellm.stream_chunk_builder(chunks, messages=messages)
        duration = time.time() - start_time
        
        # Log the response
        tool_calls = response.choices[0].message.tool_calls if response.choices else None
        self.logger.log_llm_response(complete_content, tool_calls, duration)
        
        return response