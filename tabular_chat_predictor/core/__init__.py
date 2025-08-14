"""
Core abstractions for the chat predictor system.

This module provides the protocol interfaces and core orchestration logic
that enables frontend-agnostic chat functionality.
"""

from .protocols import ChatUserInterface, ChatLogger

__all__ = [
    'ChatUserInterface',
    'ChatLogger', 
]