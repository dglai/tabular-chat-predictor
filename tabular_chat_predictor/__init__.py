"""
Web demo module for online TabPFN prediction.

This module provides functionality for real-time entity classification and regression
using TabPFN with DFS-preprocessed features. It supports temporal feature extraction
and point-in-time correctness for online prediction scenarios.

Key Components:
- ComputeEngine: Main computational interface for TabPFN predictions
- PredictingAssistant: Assistant for handling predictive workflows
- TabPFNManager: Manages TabPFN predictors with simplified dataset loading
- Dataset loaders: Utility functions for loading rel-*-input format datasets

Example Usage:
    ```python
    from tabular_chat_predictor.compute.compute_engine import ComputeEngine
    from tabular_chat_predictor.agents.predicting_assistant import PredictingAssistant
    from datetime import datetime
    
    # Initialize compute engine with simplified dataset path
    compute = ComputeEngine(
        dataset_path='datasets/demo/rel-amazon-input',
        test_timestamp=datetime(2023, 1, 1)
    )
    
    # Create predicting assistant
    assistant = PredictingAssistant(
        compute=compute,
        ui=ui,
        logger=logger,
        template=template
    )
    
    # Handle predictive queries
    result = assistant.handle_predictive_query(
        "Will customer 123 make a purchase in the next 30 days?"
    )
    ```
"""

# Simplified imports - no longer exposing internal predictor classes
# Users should interact through ComputeEngine and agents

__all__ = [
    # Main interfaces will be added here as needed
]

__version__ = '0.2.0'