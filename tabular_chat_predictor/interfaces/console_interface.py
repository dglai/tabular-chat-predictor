"""
Console implementation of ChatUserInterface and ChatLogger protocols.

This module provides Rich-based console implementations that can be used
for command-line interfaces.
"""

import time
from uuid import uuid4
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Generator, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.live import Live
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import FormattedText

from ..core.protocols import ChatUserInterface, ChatLogger


class ConsoleUserInterface(ChatUserInterface):
    """Rich console implementation of ChatUserInterface."""
    
    def __init__(self, console: Console = None):
        """
        Initialize console interface.
        
        Args:
            console: Rich Console instance (creates new one if None)
        """
        self.console = console or Console()
        self.history = InMemoryHistory()
        self.prompt_style = Style.from_dict({
            'username': '#00aaff bold',  # Blue color for "You: "
        })
        self.current_role = None

    def begin_user_message(self):
        """Begin a new user message context."""
        self.console.print("[bold blue]👤 You:[/bold blue]", end=" ")
        self.current_role = "user"

    def end_user_message(self):
        """End the current user message context."""
        self.console.print()
        self.current_role = None

    def begin_assistant_message(self):
        """Begin a new assistant message context."""
        self.console.print("[bold green]🤖 Assistant:[/bold green]", end=" ")
        self.current_role = "assistant"

    def end_assistant_message(self):
        """End the current assistant message context."""
        self.console.print()
        self.current_role = None
    
    def display_message(self, content: str):
        """Display a message with role-based formatting."""
        self.console.print(content)
    
    def display_streaming_message(self, content_generator: Generator[str, None, None]) -> str:
        """Display streaming message content with live updates."""
        complete_content = ""
        current_text = Text("")
        current_text.append("🤖 Assistant: ", style="bold green")
        
        with Live(current_text, refresh_per_second=10, console=self.console) as live:
            for chunk in content_generator:
                complete_content += chunk
                current_text.append(chunk, style="green")
                live.update(current_text)
        
        return complete_content
    
    def get_user_input(self) -> str:
        """Get user input with enhanced editing capabilities."""
        try:
            user_input = prompt(
                FormattedText([('class:username', 'You: ')]),
                history=self.history,
                enable_history_search=True,
                style=self.prompt_style,
                complete_style='column'
            )
            return user_input
        except (KeyboardInterrupt, EOFError):
            return None
    
    def begin_function_execution(self, function_name: str, arguments: Dict[str, Any]):
        """Display function execution start."""
        self.console.print(f"[bold cyan]🔧 Executing {function_name}() with: {arguments}[/bold cyan]")
    
    def end_function_execution(self, result: Dict[str, Any]):
        """Display function execution result."""
        if result.get("success"):
            self.console.print(f"[green]✅ {result.get('message', 'Function completed successfully')}[/green]")
            
            # Display specific result types
            if "prediction" in result:
                self.console.print(Panel(
                    f"Prediction: {result['prediction']}",
                    title="Prediction Result",
                    border_style="green"
                ))
            
            if "shap_result" in result:
                import json
                self.console.print(Panel(
                    Syntax(json.dumps(result["shap_result"], indent=2), "json", theme="monokai"),
                    title="SHAP Explanation",
                    border_style="green"
                ))
            
            if "output_file" in result:
                display_content = f"📁 Output File: {result['output_file']}\n"
                if "num_entities" in result:
                    display_content += f"📊 Entities: {result['num_entities']}\n"
                if "sample_rows" in result:
                    display_content += f"\n📋 Sample rows:\n{result['sample_rows']}"
                
                self.console.print(Panel(
                    display_content,
                    title="Batch Prediction Results",
                    border_style="green"
                ))
        else:
            self.console.print(f"[red]❌ Function failed: {result.get('error', 'Unknown error')}[/red]")
    
    def display_code_execution(self, code: str, language: str = "python"):
        """Display code that will be executed."""
        self.console.print(f"[bold yellow]🐍 Executing {language.title()} code:[/bold yellow]")
        self.console.print(Panel(
            Syntax(code, language, theme="monokai", line_numbers=True),
            title=f"{language.title()} Code",
            border_style="yellow"
        ))
    
    def display_code_output(self, output: str):
        """Display code execution output."""
        self.console.print("[bold blue]📤 Output:[/bold blue]")
        self.console.print(Panel(
            output,
            title="Execution Output",
            border_style="blue"
        ))
    
    def display_error(self, error_message: str):
        """Display error message."""
        self.console.print(f"[bold red]❌ Error: {error_message}[/bold red]")
    
    def display_info(self, info: str):
        """Display informational message."""
        self.console.print(f"[blue]ℹ️  {info}[/blue]")
    
    def display_warning(self, warning: str):
        """Display warning message."""
        self.console.print(f"[yellow]⚠️  WARNING: {warning}[/yellow]")

    def update_tool_status(self, status: str):
        """Update the status of a tool execution."""
        self.console.print(f"[dim]🔧 Tool status: {status}[/dim]")
    
    def display_repl_status(self, message: str, status_type: str = "info"):
        """Display REPL status messages."""
        icons = {"info": "ℹ️", "round": "🔄", "success": "✅", "error": "❌"}
        icon = icons.get(status_type, "ℹ️")
        self.console.print(f"[cyan]{icon} {message}[/cyan]")

    def display_chart(self, figure, title: str = None):
        """Display a chart or figure in the console."""
        # Instead of displaying the chart in console, save it to a file "outputs/figure_{uuid4-last-8-digit}.png" and display the path.
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        figure_path = output_dir / f"figure_{'' if not title else title + '_'}{uuid4().hex[-8:]}.png"
        figure.savefig(figure_path)
        self.console.print(f"[cyan]📊 Chart saved to: {figure_path} [/cyan]")
    
    def initialize_session(self):
        """Initialize the user interface session."""
        pass  # Console interface doesn't need special initialization
    
    def cleanup_session(self):
        """Clean up the user interface session."""
        pass  # Console interface doesn't need special cleanup


class ConsoleLogger(ChatLogger):
    """Console implementation of ChatLogger."""
    
    def __init__(self, console: Console = None, verbose: bool = True):
        """
        Initialize console logger.
        
        Args:
            console: Rich Console instance (creates new one if None)
            verbose: Whether to display debug messages
        """
        self.console = console or Console()
        self.verbose = verbose
    
    def log_debug(self, message: str):
        """Log debug message."""
        if self.verbose:
            self.console.print(f"[dim]🔍 DEBUG: {message}[/dim]")
    
    def log_info(self, message: str):
        """Log info message."""
        if self.verbose:
            self.console.print(f"[blue]ℹ️  INFO: {message}[/blue]")
    
    def log_error(self, message: str, exc_info: bool = False):
        """Log error message."""
        self.console.print(f"[bold red]❌ ERROR: {message}[/bold red]")
        # Note: exc_info parameter available but not used in console implementation
    
    def log_warning(self, message: str):
        """Log warning message."""
        if self.verbose:
            self.console.print(f"[yellow]⚠️  WARNING: {message}[/yellow]")
    
    def log_function_call(self, function_name: str, arguments: Dict[str, Any]):
        """Log function call."""
        if self.verbose:
            self.console.print(f"[cyan]📞 CALL: {function_name}({arguments})[/cyan]")
    
    def log_function_result(self, function_name: str, success: bool, duration: float, result: Dict[str, Any]):
        """Log function result."""
        if self.verbose:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            self.console.print(f"[cyan]📋 RESULT: {function_name} - {status} ({duration:.2f}s)[/cyan]")
    
    def log_llm_request(self, model: str, messages: Any, tools: Any = None):
        """Log LLM request."""
        if self.verbose:
            num_messages = len(messages) if hasattr(messages, '__len__') else 'unknown'
            num_tools = len(tools) if tools and hasattr(tools, '__len__') else 0
            self.console.print(f"[magenta]🧠 LLM REQUEST: {model} - {num_messages} messages, {num_tools} tools[/magenta]")
    
    def log_llm_response(self, content: str, tool_calls: Any, duration: float):
        """Log LLM response."""
        if self.verbose:
            content_preview = content[:50] + "..." if content and len(content) > 50 else content or ""
            has_tools = bool(tool_calls)
            self.console.print(f"[magenta]🧠 LLM RESPONSE: '{content_preview}' - tools: {has_tools} ({duration:.2f}s)[/magenta]")
    
    def log_code_execution(self, code: str, round_idx: int):
        """Log code execution details."""
        if self.verbose:
            self.console.print(f"[cyan]🐍 CODE EXECUTION: Round {round_idx} - {len(code)} chars[/cyan]")
    
    def log_repl_status(self, message: str, round_idx: int = None):
        """Log REPL status and round information."""
        if self.verbose:
            round_info = f" (Round {round_idx})" if round_idx is not None else ""
            self.console.print(f"[cyan]🔄 REPL: {message}{round_info}[/cyan]")
    
    def log_execution_result(self, success: bool, output: str, error: str = None):
        """Log execution results."""
        if self.verbose:
            status = "SUCCESS" if success else "FAILED"
            output_len = len(output) if output else 0
            self.console.print(f"[cyan]📋 EXECUTION: {status} - output: {output_len} chars[/cyan]")
            if error:
                self.console.print(f"[red]💥 EXECUTION ERROR: {error}[/red]")