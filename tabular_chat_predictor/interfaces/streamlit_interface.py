"""
Streamlit implementation of ChatUserInterface and ChatLogger protocols.

This module provides Streamlit-based implementations that can be used
for web-based interfaces.
"""

import time
from typing import Generator, Dict, Any
import streamlit as st
import json
from streamlit_mermaid import st_mermaid

from ..core.protocols import ChatUserInterface, ChatLogger


class StreamlitUserInterface(ChatUserInterface):
    """Streamlit implementation of ChatUserInterface."""
    
    def __init__(self):
        """Initialize Streamlit interface."""
        # Initialize session state for conversation history if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_streaming" not in st.session_state:
            st.session_state.current_streaming = None
        if "history" not in st.session_state:
            st.session_state.history = []
        
        # Initialize session state for nested status containers
        if "status_containers" not in st.session_state:
            st.session_state.status_containers = []
        if "current_status_container" not in st.session_state:
            st.session_state.current_status_container = None
            
        # Initialize session state for message block contexts
        if "message_contexts" not in st.session_state:
            st.session_state.message_contexts = []
        if "message_roles" not in st.session_state:
            st.session_state.message_roles = [None]
        if "current_message_role" not in st.session_state:
            st.session_state.current_message_role = None
            
        self.wrapped_display_funcs = {}
        self.overwritten_display_funcs = set()
        
        # Clean up any orphaned contexts from previous sessions
        self._cleanup_orphaned_contexts()

    def _save_to_history(func):
        """Decorator to additionally cache the arguments and returns of each display element into a history"""
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            # Save the arguments to history
            st.session_state.history.append({
                "function": func.__name__,
                "args": args,
                "kwargs": kwargs,
                "returns": result
            })
            # Also cache the wrapped function for later use
            if func.__name__ not in self.wrapped_display_funcs:
                self.wrapped_display_funcs[func.__name__] = func
            return result
        return wrapper

    @_save_to_history
    def display_message(self, content: str):
        """Display a message in Streamlit chat format."""
        # Add to session state for persistence
        st.session_state.messages.append({"role": st.session_state.current_message_role, "content": content})
        st.markdown(content)
    
    @_save_to_history
    def display_streaming_message(self, content_generator: Generator[str, None, None]) -> str:
        """Display streaming message content with live updates."""
        complete_content = ""
        
        # Create a placeholder for streaming content
        message_placeholder = st.empty()
        
        # Stream the content
        for chunk in content_generator:
            complete_content += chunk
            message_placeholder.markdown(complete_content + "▌")  # Add cursor
        
        # Remove cursor and show final content
        message_placeholder.markdown(complete_content)
        
        # Add to session state
        st.session_state.messages.append({"role": "assistant", "content": complete_content})
        
        return complete_content
    
    @_save_to_history
    def begin_user_message(self):
        """Begin a user message block."""
        context = st.chat_message("user")
        context.__enter__()
        st.session_state.message_contexts.append(context)
        st.session_state.message_roles.append("user")
        st.session_state.current_message_role = "user"
    
    @_save_to_history
    def end_user_message(self):
        """End the current user message block."""
        if st.session_state.message_contexts:
            context = st.session_state.message_contexts.pop()
            try:
                context.__exit__(None, None, None)
            except Exception as e:
                print(f"Warning: Error closing user message context: {e}")
        if st.session_state.message_roles:
            st.session_state.message_roles.pop()
            st.session_state.current_message_role = st.session_state.message_roles[-1]
    
    @_save_to_history
    def begin_assistant_message(self):
        """Begin an assistant message block."""
        context = st.chat_message("assistant")
        context.__enter__()
        st.session_state.message_contexts.append(context)
        st.session_state.message_roles.append("assistant")
        st.session_state.current_message_role = "assistant"
    
    @_save_to_history
    def end_assistant_message(self):
        """End the current assistant message block."""
        if st.session_state.message_contexts:
            context = st.session_state.message_contexts.pop()
            try:
                context.__exit__(None, None, None)
            except Exception as e:
                print(f"Warning: Error closing assistant message context: {e}")
        if st.session_state.message_roles:
            st.session_state.message_roles.pop()
            st.session_state.current_message_role = st.session_state.message_roles[-1]
    
    def get_user_input(self) -> str:
        """Get user input from Streamlit chat input."""
        # This is handled by the Streamlit frontend, not directly here
        # The frontend will call process_user_message directly
        return None
    
    @_save_to_history
    def begin_function_execution(self, function_name: str, arguments: Dict[str, Any]):
        """Display function execution start - creates new status container."""
        # Create status container for this tool call
        status_container = st.status(f"🔧 Executing {function_name}()", expanded=False)
        
        # Manually enter the context
        status_container.__enter__()
        
        # Store container reference and enter tool context
        st.session_state.status_containers.append(status_container)
        st.session_state.current_status_container = status_container
        
        # Display arguments inside the container
        st.write(f"**Arguments:** {arguments}")
    
    @_save_to_history
    def end_function_execution(self, result: Dict[str, Any]):
        """Display function execution result - updates container label and exits context."""
        # Display result (we should be inside the container at this point)
        if result.get("success"):
            st.success(f"✅ {result.get('message', 'Function completed successfully')}")
            
            # Display specific result types
            if "prediction" in result:
                st.subheader("Prediction Result")
                st.write(f"**Prediction:** {result['prediction']}")
            
            if "shap_result" in result:
                st.subheader("SHAP Explanation")
                st.json(result["shap_result"])
            
            if "output_file" in result:
                st.subheader("Batch Prediction Results")
                st.write(f"📁 **Output File:** {result['output_file']}")
                if "num_entities" in result:
                    st.write(f"📊 **Entities:** {result['num_entities']}")
                if "sample_rows" in result:
                    st.write("📋 **Sample rows:**")
                    st.text(result['sample_rows'])
            
            if "response" in result:
                st.subheader("Coding Assistant Response")
                st.write(result["response"])
                
        else:
            st.error(f"❌ Function failed: {result.get('error', 'Unknown error')}")
        
        # Update container label with final result
        final_message = result.get('message', 'Function completed successfully')
        icon = "✅" if result.get("success") else "❌"
        
        st.session_state.current_status_container.update(
            label=f"{icon} {final_message}",
            state="complete" if result.get("success") else "error"
        )
        
        # Manually exit context
        st.session_state.current_status_container.__exit__(None, None, None)
        
        # Clean up state
        st.session_state.status_containers.pop()
        st.session_state.current_status_container = st.session_state.status_containers[-1] if st.session_state.status_containers else None

    @_save_to_history
    def display_code_execution(self, code: str, language: str = "python"):
        """Display code that will be executed - nests in container if in tool context."""
        # Display inside the current tool container
        st.subheader(f"🐍 Executing {language.title()} Code")
        st.code(code, language=language)

    @_save_to_history
    def display_code_output(self, output: str):
        """Display code execution output - nests in container if in tool context."""
        # Display inside the current tool container
        st.subheader("📤 Code Output")
        st.code(output)
    
    @_save_to_history
    def display_error(self, error_message: str):
        """Display error message - updates container to error state if in tool context."""
        # Display error inside container and update state
        st.error(f"❌ Error: {error_message}")
    
    @_save_to_history
    def display_info(self, info: str):
        """Display informational message - nests in container if in tool context."""
        # Display inside the current tool container
        st.info(f"ℹ️  {info}")
    
    @_save_to_history
    def display_warning(self, warning: str):
        """Display warning message - nests in container if in tool context."""
        # Display inside the current tool container
        st.warning(f"⚠️  WARNING: {warning}")

    @_save_to_history
    def update_tool_status(self, status: str):
        """Update the status of a tool execution - nests in container if in tool context."""
        # Display inside the current tool container
        if st.session_state.get('current_status_container', None):
            st.session_state.current_status_container.update(label=f"🔧 {status}")
        else:
            st.info(f"🔧 {status}")
    
    @_save_to_history
    def display_repl_status(self, message: str, status_type: str = "info"):
        """Display REPL status messages - nests in container if in tool context."""
        icons = {"info": "ℹ️", "round": "🔄", "success": "✅", "error": "❌"}
        icon = icons.get(status_type, "ℹ️")
        # Display inside the current tool container
        st.info(f"{icon} {message}")
    
    def display_mermaid_diagram(self, mermaid_content: str, title: str = "Database Schema"):
        """Display a Mermaid diagram."""
        st.subheader(f"📊 {title}")
        st_mermaid(mermaid_content)
        st.markdown("---")  # Add separator
    
    @_save_to_history
    def display_chart(self, figure, title: str = None):
        """Display a matplotlib chart - nests in container if in tool context."""
        if title:
            st.subheader(f"📈 {title}")
        else:
            st.subheader("📈 Generated Chart")
        
        # Display the matplotlib figure
        st.pyplot(figure)
    
    def initialize_session(self):
        """Initialize the user interface session."""
        # Streamlit session state is automatically managed
        pass
    
    def cleanup_session(self):
        """Clean up the user interface session."""
        # Streamlit automatically handles cleanup
        pass

    def display_conversation_history(self):
        # Clean up any orphaned contexts first
        self._cleanup_orphaned_contexts()
        
        for history_entry in st.session_state.history:
            if history_entry["function"] == "display_streaming_message":
                # Display streaming message content
                st.markdown(history_entry["returns"])
            else:
                func = self.wrapped_display_funcs.get(history_entry["function"])
                # Call the wrapped function with the stored arguments
                func(self, *history_entry["args"], **history_entry["kwargs"])

    def _cleanup_orphaned_contexts(self):
        """Clean up any orphaned manual contexts from previous sessions."""
        # Clean up tool contexts
        if st.session_state.get('tool_call_context', False):
            try:
                while st.session_state.status_containers:
                    status_container = st.session_state.status_containers.pop()
                    status_container.__exit__(None, None, None)
                    print("Cleaning up orphaned tool context")
            except:
                pass  # Context might already be closed
            finally:
                st.session_state.status_containers = []
                st.session_state.current_status_container = None
        
        # Clean up message contexts
        if st.session_state.get('message_contexts'):
            try:
                while st.session_state.message_contexts:
                    context = st.session_state.message_contexts.pop()
                    context.__exit__(None, None, None)
                    print("Cleaning up orphaned message context")
            except:
                pass  # Context might already be closed
            finally:
                st.session_state.message_contexts = []
                st.session_state.message_roles = [None]
                st.session_state.current_message_role = None


class StreamlitLogger(ChatLogger):
    """Streamlit implementation of ChatLogger."""
    
    def __init__(self, show_debug: bool = False):
        """
        Initialize Streamlit logger.
        
        Args:
            show_debug: Whether to show debug messages in the UI
        """
        self.show_debug = show_debug
        
        # Initialize session state for logs if not exists
        if "logs" not in st.session_state:
            st.session_state.logs = []
    
    def _add_log(self, level: str, message: str):
        """Add log entry to session state."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message
        }
        st.session_state.logs.append(log_entry)
        
        # Keep only last 100 logs to prevent memory issues
        if len(st.session_state.logs) > 100:
            st.session_state.logs = st.session_state.logs[-100:]
    
    def log_debug(self, message: str):
        """Log debug message."""
        self._add_log("DEBUG", message)
        if self.show_debug:
            st.sidebar.text(f"🔍 DEBUG: {message}")
    
    def log_info(self, message: str):
        """Log info message."""
        self._add_log("INFO", message)
        if self.show_debug:
            st.sidebar.info(f"ℹ️ {message}")
    
    def log_error(self, message: str, exc_info: bool = False):
        """Log error message."""
        self._add_log("ERROR", message)
        st.sidebar.error(f"❌ {message}")
        # Note: exc_info parameter available but not used in streamlit implementation
    
    def log_warning(self, message: str):
        """Log warning message."""
        self._add_log("WARNING", message)
        if self.show_debug:
            st.sidebar.warning(f"⚠️  {message}")
    
    def log_function_call(self, function_name: str, arguments: Dict[str, Any]):
        """Log function call."""
        message = f"CALL: {function_name}({arguments})"
        self._add_log("FUNCTION_CALL", message)
        if self.show_debug:
            st.sidebar.text(f"📞 {message}")
    
    def log_function_result(self, function_name: str, success: bool, duration: float, result: Dict[str, Any]):
        """Log function result."""
        status = "SUCCESS" if success else "FAILED"
        message = f"RESULT: {function_name} - {status} ({duration:.2f}s)"
        self._add_log("FUNCTION_RESULT", message)
        if self.show_debug:
            st.sidebar.text(f"📋 {message}")
    
    def log_llm_request(self, model: str, messages: Any, tools: Any = None):
        """Log LLM request."""
        num_messages = len(messages) if hasattr(messages, '__len__') else 'unknown'
        num_tools = len(tools) if tools and hasattr(tools, '__len__') else 0
        message = f"LLM REQUEST: {model} - {num_messages} messages, {num_tools} tools"
        self._add_log("LLM_REQUEST", message)
        if self.show_debug:
            st.sidebar.text(f"🧠 {message}")
    
    def log_llm_response(self, content: str, tool_calls: Any, duration: float):
        """Log LLM response."""
        content_preview = content[:50] + "..." if content and len(content) > 50 else content or ""
        has_tools = bool(tool_calls)
        message = f"LLM RESPONSE: '{content_preview}' - tools: {has_tools} ({duration:.2f}s)"
        self._add_log("LLM_RESPONSE", message)
        if self.show_debug:
            st.sidebar.text(f"🧠 {message}")
    
    def log_code_execution(self, code: str, round_idx: int):
        """Log code execution details."""
        message = f"CODE EXECUTION: Round {round_idx} - {len(code)} chars"
        self._add_log("CODE_EXECUTION", message)
        if self.show_debug:
            st.sidebar.text(f"🐍 {message}")
    
    def log_repl_status(self, message: str, round_idx: int = None):
        """Log REPL status and round information."""
        round_info = f" (Round {round_idx})" if round_idx is not None else ""
        log_message = f"REPL: {message}{round_info}"
        self._add_log("REPL_STATUS", log_message)
        if self.show_debug:
            st.sidebar.text(f"🔄 {log_message}")
    
    def log_execution_result(self, success: bool, output: str, error: str = None):
        """Log execution results."""
        status = "SUCCESS" if success else "FAILED"
        output_len = len(output) if output else 0
        log_message = f"EXECUTION: {status} - output: {output_len} chars"
        self._add_log("EXECUTION_RESULT", log_message)
        if self.show_debug:
            st.sidebar.text(f"📋 {log_message}")
            if error:
                st.sidebar.error(f"💥 EXECUTION ERROR: {error}")
    
    def display_logs_sidebar(self):
        """Display logs in Streamlit sidebar."""
        if st.sidebar.button("Show Logs"):
            with st.sidebar.expander("System Logs", expanded=True):
                for log in reversed(st.session_state.logs[-20:]):  # Show last 20 logs
                    timestamp = time.strftime("%H:%M:%S", time.localtime(log["timestamp"]))
                    st.text(f"[{timestamp}] {log['level']}: {log['message']}")