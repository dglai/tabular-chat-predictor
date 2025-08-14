#!/usr/bin/env python3
"""
Agents package - LLM-dependent components for the webdemo.

This package contains all agent classes that depend on LLM interactions,
separated from the pure computational components in the compute package.
"""

from .base_agent import BaseAgent
from .llm_client import LLMClient

__all__ = [
    'BaseAgent',
    'LLMClient',
]