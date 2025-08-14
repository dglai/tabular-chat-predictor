#!/usr/bin/env python3
"""
REPL Manager for managing IPython shells and code execution.
Provides computational Python execution capabilities for the compute engine.
"""

import io
import sys
import traceback
import base64
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
import duckdb
from datetime import datetime
from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pyplot as plt
import seaborn as sns


class REPLManager:
    """
    Manages IPython shells for code execution with persistent state.
    Handles environment setup, code execution, variable tracking, and figure capture.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the REPL manager with provided variables.
        
        Args:
            **kwargs: Variables to add to the user namespace (e.g., tables, functions, etc.)
        """
        # Create a new InteractiveShell instance (not singleton)
        self.shell = InteractiveShell()
        self.user_kwargs = kwargs
        self._setup_environment()
    
    def _setup_environment(self):
        """Set up the execution environment with required libraries and user variables."""
        # Add standard libraries
        libs_to_add = {
            'numpy': np,
            'np': np,
            'pandas': pd,
            'pd': pd,
            'duckdb': duckdb,
            'datetime': datetime,
            'matplotlib': plt,
            'plt': plt,
            'seaborn': sns,
            'sns': sns
        }
        
        # Add user-provided kwargs to the namespace
        libs_to_add.update(self.user_kwargs)
        
        self.shell.user_ns.update(libs_to_add)
        
        # Configure matplotlib for better display
        setup_code = """
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")
"""
        
        self.shell.run_cell(setup_code)
        
        # Create DuckDB connection with tables registered (if tables are provided)
        tables = self.user_kwargs.get('tables', {})
        if tables:
            conn = duckdb.connect(':memory:')
            for table_name, df in tables.items():
                conn.register(table_name, df)
            self.shell.user_ns['conn'] = conn

        self.shell.run_cell("pd.set_option('display.max_columns', None)")
        
        table_count = len(tables) if tables else 0
        libs_msg = f"✓ Environment initialized with {table_count} tables"
        print(libs_msg)
    
    def run_python(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in the persistent environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary with execution results including output, errors, and success status
        """
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'variables_added': [],
            'figures': []
        }
        
        # Store variables before execution to track what's new
        vars_before = set(self.shell.user_ns.keys())
        
        # Clear any existing matplotlib figures
        plt.close('all')
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the code using IPython's run_cell
            exec_result = self.shell.run_cell(code, store_history=True)
        
        # Check for execution errors using IPython's error handling
        if exec_result.error_before_exec:
            result['error'] = ''.join(traceback.format_exception(exec_result.error_before_exec))
            result['success'] = False
            return result
        
        if exec_result.error_in_exec:
            result['error'] = ''.join(traceback.format_exception(exec_result.error_in_exec))
            result['success'] = False
            return result
        
        # Execution was successful
        result['success'] = True
        
        # Capture output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        result['output'] = stdout_output
        if stderr_output:
            result['error'] = stderr_output
        
        # Track new variables
        vars_after = set(self.shell.user_ns.keys())
        result['variables_added'] = list(vars_after - vars_before)
        
        # Capture matplotlib figures
        result['figures'] = self._capture_matplotlib_figures()
        
        return result
    
    def _capture_matplotlib_figures(self) -> List[Dict[str, Any]]:
        """
        Capture all matplotlib figures that were created during execution.
        
        Returns:
            List of figure information dictionaries
        """
        figures = []
        
        # Check if any figures were created
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                
                # Get figure title if available
                title = None
                if fig._suptitle:
                    title = fig._suptitle.get_text()
                
                figures.append({
                    'type': 'matplotlib',
                    'title': title,
                    'figure_num': fig_num,
                    'figure_object': fig  # Keep reference for direct display
                })
        
        return figures
    
    def clear_figures(self):
        """Clear all matplotlib figures."""
        plt.close('all')
    
    def get_matplotlib_figures(self) -> List[Dict[str, Any]]:
        """Get all current matplotlib figures."""
        return self._capture_matplotlib_figures()
    
    def get_repl_variables(self) -> Dict[str, Any]:
        """Get all variables from the REPL namespace."""
        variables = {}
        for name, value in self.shell.user_ns.items():
            if not name.startswith('_') and name not in ['In', 'Out', 'exit', 'quit', 'get_ipython']:
                variables[name] = value
        return variables
    
    def get_variable(self, name: str) -> Any:
        """Get a specific variable from the namespace."""
        return self.shell.user_ns.get(name)
    
    def list_variables(self) -> Dict[str, str]:
        """List all user-defined variables and their types."""
        variables = {}
        for name, value in self.shell.user_ns.items():
            if not name.startswith('_') and name not in ['In', 'Out', 'exit', 'quit', 'get_ipython']:
                variables[name] = type(value).__name__
        return variables
    
    def reset_repl(self):
        """Reset the environment while keeping tables."""
        self.shell.reset(new_session=False)
        self._setup_environment()
    
    def close(self):
        """Clean up resources."""
        if 'conn' in self.shell.user_ns:
            self.shell.user_ns['conn'].close()
        plt.close('all')
    