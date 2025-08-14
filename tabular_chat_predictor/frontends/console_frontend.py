"""
Console frontend using the refactored core components.

This is a drop-in replacement for chat_predictor.py that uses the new
protocol-based architecture.
"""

import argparse
from datetime import datetime
from pathlib import Path
from rich.console import Console

from ..agents.orchestrator import Orchestrator
from ..compute.compute_engine import ComputeEngine
from ..interfaces.console_interface import ConsoleUserInterface, ConsoleLogger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive LLM-driven predictive analytics chat interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tabular_chat_predictor.frontends.console_frontend datasets/demo/rel-amazon-input "2016-01-01"
  python -m tabular_chat_predictor.frontends.console_frontend datasets/demo/rel-stack-input "2014-09-01"
        """
    )
    
    parser.add_argument(
        "dataset_path",
        help="Path to the dataset directory (e.g., 'datasets/demo/rel-amazon-input')"
    )
    
    
    parser.add_argument(
        "test_timestamp",
        help="Test timestamp in YYYY-MM-DD format (e.g., '2016-01-01')"
    )
    
    parser.add_argument(
        "--model",
        default="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        help="LLM model to use (default: bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM responses (default: 0.1)"
    )
    
    parser.add_argument(
        "--dfs-depth",
        type=int,
        default=3,
        help="DFS depth for feature engineering (default: 3)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def initialize_compute_engine(dataset_path: str, test_timestamp: datetime) -> ComputeEngine:
    """Initialize the ComputeEngine with given parameters."""
    console = Console()
    
    with console.status("[bold green]Loading dataset and initializing compute engine..."):
        compute = ComputeEngine(
            dataset_path=dataset_path,
            test_timestamp=test_timestamp
        )
    
    console.print("[bold green]✅ Compute engine initialized successfully![/bold green]")
    console.print(f"[blue]📊 Dataset: {dataset_path}[/blue]")
    console.print(f"[blue]📅 Test timestamp: {test_timestamp}[/blue]")
    console.print()
    
    return compute


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Parse test timestamp
    test_timestamp = datetime.strptime(args.test_timestamp, "%Y-%m-%d")
    
    # Initialize console and logger
    console = Console()
    ui = ConsoleUserInterface(console)
    logger = ConsoleLogger(console, verbose=args.verbose)
    
    # Initialize compute engine
    compute = initialize_compute_engine(
        args.dataset_path,
        test_timestamp
    )
    
    # Create orchestrator
    orchestrator = Orchestrator(
        compute=compute,
        ui=ui,
        logger=logger,
        model=args.model,
        temperature=args.temperature
    )
    
    # Start conversation
    console.print("[bold cyan]💬 Chat started! Ask me questions about your data.[/bold cyan]")
    console.print("[dim]Type 'quit', 'exit', or 'bye' to end the conversation.[/dim]")
    console.print()
    
    orchestrator.start_conversation()


if __name__ == "__main__":
    main()