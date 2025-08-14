"""
Streamlit frontend using the refactored core components.

This provides a web-based interface using the new protocol-based architecture.
"""

import os
import traceback
import streamlit as st
from datetime import datetime
from pathlib import Path

from ..agents.orchestrator import Orchestrator
from ..compute.compute_engine import ComputeEngine
from ..interfaces.streamlit_interface import StreamlitUserInterface, StreamlitLogger
from ..schema_to_mermaid import schema_yaml_to_mermaid
from streamlit_mermaid import st_mermaid


def initialize_compute_engine(dataset_path: str, test_timestamp: datetime) -> ComputeEngine:
    """Initialize the ComputeEngine with given parameters."""
    with st.spinner("Loading dataset and initializing compute engine..."):
        compute = ComputeEngine(
            dataset_path=dataset_path,
            test_timestamp=test_timestamp
        )
    
    st.success("✅ Compute engine initialized successfully!")
    return compute


def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.title("🔧 Configuration")
    
    # Dataset selection dropdown
    st.sidebar.subheader("Dataset Selection")
    dataset_choice = st.sidebar.selectbox(
        "Choose Dataset",
        ["Amazon", "StackExchange"],
        help="Select a pre-configured dataset"
    )
    
    # Pre-configured dataset settings
    if dataset_choice == "Amazon":
        dataset_path = "datasets/demo/rel-amazon-input"
        test_timestamp_default = "2016-01-01"
    else:  # StackExchange
        dataset_path = "datasets/demo/rel-stack-input"
        test_timestamp_default = "2021-01-01"
    
    # Dataset configuration (now read-only display)
    st.sidebar.subheader("Dataset Configuration")
    st.sidebar.text_input(
        "Dataset Path",
        value=dataset_path,
        disabled=True,
        help="Path to the dataset directory (auto-configured)"
    )
    
    
    test_timestamp = st.sidebar.date_input(
        "Today's Date",
        value=test_timestamp_default,
        help="Test timestamp for predictions",
        disabled=True
    )
    test_timestamp = datetime.combine(test_timestamp, datetime.min.time())  # Combine date with time
    
    # Model configuration
    model = os.environ.get("LITELLM_MODEL", "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
    temperature = 0
    
    # Debug options
    st.sidebar.subheader("Debug")
    show_debug = st.sidebar.checkbox("Show Debug Logs", value=False)
    
    return {
        "dataset_choice": dataset_choice,
        "dataset_path": dataset_path,
        "test_timestamp": test_timestamp,
        "model": model,
        "temperature": temperature,
        "show_debug": show_debug
    }


def get_example_queries(dataset_choice: str) -> list[str]:
    """Get example queries for the selected dataset."""
    if dataset_choice == "Amazon":
        return [
            "Would user ID 1592898 buy or review any product in the next quarter (3 months)?",
            "How much will user ID 1593600 spend in the next quarter (3 months)?",
            "Would product ID 420601 be bought or reviewed by any user in the next quarter (3 months)?",
            "How much sales will product ID 432527 generate in the next quarter (3 months)?"
        ]
    else:  # StackExchange
        return [
            "Would user ID 2666 make any votes, posts or comments in the next quarter (3 months)?",
            "Would user ID 76999 receive any new badge in the next quarter (3 months)?",
            "How many votes would post ID 115603 receive in the next quarter (3 months)?"
        ]


def display_example_queries(dataset_choice: str):
    """Display example queries for the selected dataset."""
    st.subheader("📝 Example Queries")
    st.markdown(f"Here are some example queries you can try with the **{dataset_choice}** dataset:")
    
    example_queries = get_example_queries(dataset_choice)
    
    for i, query in enumerate(example_queries, 1):
        with st.expander(f"Example {i}: {query[:50]}..."):
            st.markdown(f"**Query:** {query}")
            if st.button(f"Use this query", key=f"example_{i}"):
                # Add the query to the chat input
                st.session_state.example_query = query
                st.rerun()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Tabular Chat Predictor",
        page_icon="🔮",
        layout="wide"
    )
    
    st.title("🔮 Tabular Chat Predictor")
    st.markdown("Interactive LLM-driven predictive analytics for tabular data")
    
    # Setup sidebar configuration
    config = setup_sidebar()
    
    # Initialize session state
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "predictor_initialized" not in st.session_state:
        st.session_state.predictor_initialized = False
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None
    
    # Auto-initialize predictor when dataset changes or on first load
    dataset_changed = st.session_state.current_dataset != config["dataset_choice"]
    
    if dataset_changed or not st.session_state.predictor_initialized:
        st.session_state.current_dataset = config["dataset_choice"]
        
        # Initialize interfaces
        ui = StreamlitUserInterface()
        logger = StreamlitLogger(show_debug=config["show_debug"])
        
        # Initialize compute engine
        compute = initialize_compute_engine(
            config["dataset_path"],
            config["test_timestamp"]
        )
        
        # Generate and display mermaid diagram from dataset metadata
        try:
            schema_yaml_path = f"{config['dataset_path']}/metadata.yaml"
            mermaid_content = schema_yaml_to_mermaid(schema_yaml_path)
            st.session_state.mermaid_diagram = {
                "content": mermaid_content,
                "title": f"{config['dataset_choice']} Database Schema"
            }
        except Exception as e:
            logger.log_warning(f"Could not generate mermaid diagram: {e}")
        
        # Create orchestrator
        st.session_state.orchestrator = Orchestrator(
            compute=compute,
            ui=ui,
            logger=logger,
            model=config["model"],
            temperature=config["temperature"]
        )
        
        st.session_state.predictor_initialized = True
        
        # Display initialization info
        st.info(f"""
        **Auto-initialized with:**
        - Dataset: {config['dataset_choice']} ({config['dataset_path']})
        - Test timestamp: {config['test_timestamp']}
        - Model: {config['model']}
        """)

    # Main chat interface
    if st.session_state.predictor_initialized and st.session_state.orchestrator:
        orchestrator = st.session_state.orchestrator
        ui = orchestrator.ui
        logger = orchestrator.logger
        
        # Display mermaid diagram at the beginning of chat interface
        if "mermaid_diagram" in st.session_state:
            diagram_info = st.session_state.mermaid_diagram
            ui.display_mermaid_diagram(diagram_info['content'], diagram_info['title'])

        ui.display_conversation_history()
        
        # Handle example query selection
        if "example_query" in st.session_state:
            prompt = st.session_state.example_query
            del st.session_state.example_query
            # Process the message
            try:
                orchestrator.process_user_message(prompt)
            except Exception as e:
                st.error(f"❌ Error processing message: {''.join(traceback.format_exception(e))}")
        
        # Chat input
        elif prompt := st.chat_input("Ask me about your data..."):
            # Process the message
            try:
                orchestrator.process_user_message(prompt)
            except Exception as e:
                st.error(f"❌ Error processing message: {''.join(traceback.format_exception(e))}")
        
        # Display logs in sidebar if debug is enabled
        if config["show_debug"]:
            logger.display_logs_sidebar()
        
        # Conversation state info
        if st.sidebar.button("📊 Show Conversation State"):
            state = orchestrator.get_conversation_state()
            st.sidebar.json(state)
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome! 👋
        
        This is the Tabular Chat Predictor - an interactive LLM-driven predictive analytics tool for tabular data.
        
        ### Features:
        - 🔮 **Make Predictions**: Train ML models and predict outcomes
        - 📊 **Explain Results**: Use SHAP to understand predictions
        - 💻 **Explore Data**: Write custom code to analyze your database
        - 🎯 **Answer Questions**: Get insights about your data
        
        ### Getting Started:
        1. Select your dataset (Amazon or StackExchange) in the sidebar
        2. Configure your model settings in the sidebar
        3. The predictor will automatically initialize when you select a dataset
        4. Try one of the example queries below or ask your own questions!
        
        ### Architecture:
        This application uses a clean protocol-based architecture:
        - **Protocol-based interfaces** for clean separation of concerns
        - **Frontend-agnostic orchestrator** that works with any UI
        - **Streaming LLM responses** with real-time updates
        - **Unified function execution** with consistent error handling
        """)
        
        # Display example queries for the selected dataset
        display_example_queries(config["dataset_choice"])


if __name__ == "__main__":
    main()