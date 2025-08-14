#!/usr/bin/env python3
"""
BaseAgent - Common functionality for all agents in the new architecture.
Provides shared initialization pattern for consistent agent architecture.
"""

import logging

from ..compute.compute_engine import ComputeEngine
from ..core.protocols import ChatUserInterface, ChatLogger
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Common functionality for all agents.
    
    Provides shared initialization with ComputeEngine, UI, and Logger protocols.
    All agents inherit from this base class to ensure consistent architecture.
    """
    
    def __init__(
        self,
        compute: ComputeEngine,
        ui: ChatUserInterface,
        logger: ChatLogger,
        model: str = "openrouter/anthropic/claude-sonnet-4",
        temperature: float = 0.1
    ):
        """
        Initialize the base agent.
        
        Args:
            compute: ComputeEngine instance for computational operations
            ui: ChatUserInterface for consistent UI display
            logger: ChatLogger for protocol-based logging
            model: LLM model to use (default: claude-sonnet-4)
            temperature: Temperature for LLM responses (default: 0.1)
        """
        self.compute = compute
        self.ui = ui
        self.logger = logger
        self.model = model
        self.temperature = temperature
        
        # Initialize LLM client with shared configuration
        self.llm_client = LLMClient(
            model=model,
            temperature=temperature,
            ui=ui,
            logger=logger
        )
        
        logger.log_info(f"Initialized {self.__class__.__name__} with model: {model}")