"""
Compute package - Non-LLM computational components for the webdemo.

This package contains all components that perform computational operations
without LLM dependencies:
- ComputeEngine: Main computational interface
- TabPFNManager: Manages TabPFN predictors
- REPLManager: Manages IPython shell and code execution
- Dataset loaders: Functions for loading DFS preprocessed datasets
"""

from .compute_engine import ComputeEngine
from .tabpfn_manager import TabPFNManager
from .repl_manager import REPLManager

__all__ = [
    'ComputeEngine',
    'TabPFNManager',
    'REPLManager',
]