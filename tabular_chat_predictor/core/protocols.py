"""
Protocol interfaces for the chat predictor system.

These protocols define the contracts that frontend implementations must follow
to provide user interface and logging functionality.
"""

from typing import Protocol, Optional, Dict, Any, List, Generator


class ChatUserInterface(Protocol):
    """Protocol defining the interface for user interactions."""
    
    def get_user_input(self, prompt: str = "You: ") -> Optional[str]:
        """
        Get input from user. Returns None if user wants to quit.
        
        Args:
            prompt: The prompt to display to the user
            
        Returns:
            User input string, or None if user wants to quit
        """
        ...

    def begin_user_message(self):
        """
        Begin a new user message context.
        
        This method should be called before displaying the user's message.
        """
        ...

    def end_user_message(self):
        """
        End the current user message context.
        
        This method should be called after displaying the user's message.
        """
        ...

    def begin_assistant_message(self):
        """
        Begin a new assistant message context.
        
        This method should be called before displaying the assistant's message.
        """
        ...

    def end_assistant_message(self):
        """
        End the current assistant message context.
        
        This method should be called after displaying the assistant's message.
        """
        ...
    
    def display_message(self, content: str) -> None:
        """
        Display a message.
        
        Args:
            content: The message content to display
        """
        ...
    
    def display_streaming_message(self, message_generator: Generator[str, None, None]) -> str:
        """
        Display a streaming message and return the complete content.
        
        Args:
            message_generator: Generator that yields message chunks
            
        Returns:
            Complete message content
        """
        ...
    
    def begin_function_execution(self, function_name: str, arguments: Dict[str, Any]) -> None:
        """
        Show that a function is being executed.
        
        Args:
            function_name: Name of the function being executed
            arguments: Arguments passed to the function
        """
        ...
    
    def end_function_execution(self, result: Dict[str, Any]) -> None:
        """
        Display the result of a function execution.
        
        Args:
            result: Function execution result containing success status and data
        """
        ...
    
    def display_error(self, error: str) -> None:
        """
        Display an error message.
        
        Args:
            error: Error message to display
        """
        ...
    
    def display_info(self, info: str) -> None:
        """
        Display informational message.
        
        Args:
            info: Information message to display
        """
        ...
    
    def display_warning(self, warning: str) -> None:
        """
        Display warning message.
        
        Args:
            warning: Warning message to display
        """
        ...

    def update_tool_status(self, status: str) -> None:
        """
        Update the status of a tool execution.
        
        Args:
            status: Status message to display
        """
        ...
    
    def display_code_execution(self, code: str, language: str = "python") -> None:
        """
        Display code that's about to be executed.
        
        Args:
            code: Code content to display
            language: Programming language for syntax highlighting
        """
        ...
    
    def display_code_output(self, output: str) -> None:
        """
        Display the result of code execution.
        
        Args:
            output: Output from code execution
        """
        ...
    
    def display_repl_status(self, message: str, status_type: str = "info") -> None:
        """
        Display REPL status messages.
        
        Args:
            message: Status message to display
            status_type: Type of status (info, round, success, error)
        """
        ...
    
    def display_mermaid_diagram(self, mermaid_content: str, title: str = "Database Schema") -> None:
        """
        Display a Mermaid diagram.
        
        Args:
            mermaid_content: Mermaid diagram content
            title: Title for the diagram section
        """
        ...
    
    def display_chart(self, figure, title: str = None) -> None:
        """
        Display a matplotlib chart/figure.
        
        Args:
            figure: Matplotlib figure object to display
            title: Optional title for the chart
        """
        ...
    
    def initialize_session(self) -> None:
        """Initialize the user interface session."""
        ...
    
    def cleanup_session(self) -> None:
        """Clean up the user interface session."""
        ...


class ChatLogger(Protocol):
    """Protocol defining the interface for logging."""
    
    def log_debug(self, message: str) -> None:
        """
        Log debug message.
        
        Args:
            message: Debug message to log
        """
        ...
    
    def log_info(self, message: str) -> None:
        """
        Log info message.
        
        Args:
            message: Info message to log
        """
        ...
    
    def log_warning(self, message: str) -> None:
        """
        Log warning message.
        
        Args:
            message: Warning message to log
        """
        ...
    
    def log_error(self, message: str, exc_info: bool = False) -> None:
        """
        Log error message.
        
        Args:
            message: Error message to log
            exc_info: Whether to include exception information
        """
        ...
    
    def log_function_call(self, function_name: str, arguments: Dict[str, Any]) -> None:
        """
        Log function call details.
        
        Args:
            function_name: Name of the function being called
            arguments: Arguments passed to the function
        """
        ...
    
    def log_function_result(self, function_name: str, success: bool, duration: float, result: Dict[str, Any] = None) -> None:
        """
        Log function execution results.
        
        Args:
            function_name: Name of the function that was executed
            success: Whether execution was successful
            duration: Execution duration in seconds
            result: Function execution result (optional)
        """
        ...
    
    def log_llm_request(self, model: str, messages: List[Dict], tools: List[Dict] = None) -> None:
        """
        Log LLM request details.
        
        Args:
            model: LLM model being used
            messages: Messages sent to the LLM
            tools: Available tools/functions (optional)
        """
        ...
    
    def log_llm_response(self, response_content: str, tool_calls: List = None, duration: float = None) -> None:
        """
        Log LLM response details.
        
        Args:
            response_content: Content of the LLM response
            tool_calls: Tool calls requested by the LLM (optional)
            duration: Response duration in seconds (optional)
        """
        ...
    
    def log_code_execution(self, code: str, round_idx: int) -> None:
        """
        Log code execution details.
        
        Args:
            code: Code being executed
            round_idx: Round index in the conversation
        """
        ...
    
    def log_repl_status(self, message: str, round_idx: int = None) -> None:
        """
        Log REPL status and round information.
        
        Args:
            message: Status message
            round_idx: Round index (optional)
        """
        ...
    
    def log_execution_result(self, success: bool, output: str, error: str = None) -> None:
        """
        Log execution results.
        
        Args:
            success: Whether execution was successful
            output: Execution output
            error: Error message if execution failed (optional)
        """
        ...