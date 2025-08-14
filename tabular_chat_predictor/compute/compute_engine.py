#!/usr/bin/env python3
"""
ComputeEngine - Main computational interface that coordinates TabPFNManager and REPLManager.
Provides pure computational capabilities separate from LLM-dependent operations.
"""

import logging
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Union
from pathlib import Path
import pandas as pd
import numpy as np

from .dataset_loader import load_dataset, get_primary_key_column
from .tabpfn_manager import TabPFNManager
from .repl_manager import REPLManager

logger = logging.getLogger(__name__)


class ComputeEngine:
    """
    Main computational interface that coordinates TabPFNManager and REPLManager.
    Handles all non-LLM computational operations including data loading, model fitting,
    predictions, and code execution.
    """
    
    def __init__(
        self,
        dataset_path: str,
        test_timestamp: datetime
    ):
        """
        Initialize the compute engine.
        
        Args:
            dataset_path: Path to the dataset (e.g., 'datasets/demo/rel-amazon-input')
            test_timestamp: Test timestamp for all predictions
        """
        
        self.dataset_path = dataset_path
        self.test_timestamp = test_timestamp
        
        logger.info(f"Initializing ComputeEngine for dataset: {dataset_path}")
        
        # Initialize TabPFNManager
        self.manager = TabPFNManager(dataset_path, test_timestamp)
        
        # Load schema YAML
        self.schema_yaml = self._load_schema_yaml()
        
        # Initialize REPLManager with tables and timestamps
        self.repl_manager = self._initialize_repl_manager()
        
        logger.info(f"ComputeEngine initialized with {len(list(self.manager.dfs_features.keys()))} target tables")
    
    def get_available_target_tables(self) -> List[str]:
        """Get list of available target tables."""
        return list(self.manager.dfs_features.keys())
    
    def get_cached_timestamps(self, target_table_name: str) -> List[datetime]:
        """Get cached timestamps for a target table."""
        return self.manager.get_cached_timestamps(target_table_name)
    
    def get_schema_yaml(self) -> str:
        """Get the schema YAML content."""
        return self.schema_yaml
    
    def get_table_info(self) -> str:
        """Get formatted table information."""
        tables = self._extract_tables_from_dataset()
        return self._get_table_info(tables)
    
    def get_tables_for_repl(self) -> Dict[str, pd.DataFrame]:
        """Get tables formatted for REPL usage."""
        return self._extract_tables_from_dataset()
    
    def _initialize_repl_manager(self) -> REPLManager:
        """Initialize the REPL manager with tables and timestamps."""
        # Add tables to the REPL environment
        tables = self._extract_tables_from_dataset()
        timestamps_dict = {
            f"{table}_timestamps": self.get_cached_timestamps(table)
            for table in self.get_available_target_tables()
        }
        
        repl_kwargs = {
            'tables': tables,
            **timestamps_dict
        }
        
        logger.info("REPL manager initialized")
        return REPLManager(**repl_kwargs)
    
    def run_python_code(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in the REPL environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary with execution results
        """
        return self.repl_manager.run_python(code)
    
    def get_repl_variables(self) -> Dict[str, Any]:
        """Get all variables from the REPL namespace."""
        return self.repl_manager.get_repl_variables()
    
    def reset_repl(self) -> None:
        """Reset the REPL environment."""
        self.repl_manager.reset_repl()
    
    def validate_training_labels(self, training_labels: pd.DataFrame) -> None:
        """Validate the format of training labels."""
        required_columns = ['__id', '__timestamp', '__label']
        missing_columns = [col for col in required_columns if col not in training_labels.columns]
        if missing_columns:
            raise ValueError(f"training_labels missing required columns: {missing_columns}")
        
        training_labels['__timestamp'] = pd.to_datetime(training_labels['__timestamp'])
        logger.info(f"Training labels validation passed: {len(training_labels)} samples")
    
    def fit(
        self,
        target_table_name: str,
        task_type: str,
        training_labels_csv: str
    ) -> None:
        """
        Fit the TabPFN model using pre-generated training labels from CSV.
        
        Args:
            target_table_name: Name of the target table
            task_type: Type of task ('classification' or 'regression')
            training_labels_csv: Path to CSV file containing training labels
        """
        logger.info(f"Fitting model for {target_table_name} with task type: {task_type}")
        logger.info(f"Training labels CSV: {training_labels_csv}")
        
        # Load training labels from CSV
        logger.info("Loading training labels from CSV...")
        training_labels = pd.read_csv(training_labels_csv)
        
        # Validate training labels format
        logger.info("Validating training labels format...")
        self.validate_training_labels(training_labels)
        
        # Delegate to TabPFNManager
        logger.info(f"Fitting model for {target_table_name}...")
        self.manager.fit(target_table_name, task_type, training_labels)
        
        logger.info(f"Model fitted successfully for {target_table_name}")
    
    def predict(
        self,
        target_table_name: str,
        test_primary_key_value: Union[str, int, float, List]
    ) -> Union[float, int, np.ndarray]:
        """
        Make a prediction for the given entity or entities.
        
        Args:
            target_table_name: Name of the target table
            test_primary_key_value: Primary key value(s) for the entity/entities to predict
            
        Returns:
            Prediction result(s)
        """
        logger.info(f"Making prediction for {target_table_name}, entity: {test_primary_key_value}")
        
        # Convert to list if single value
        if not isinstance(test_primary_key_value, list):
            entity_ids = [test_primary_key_value]
        else:
            entity_ids = test_primary_key_value
        
        # Delegate to TabPFNManager
        prediction = self.manager.predict(target_table_name, entity_ids)
        
        logger.info(f"Prediction completed for {target_table_name}")
        return prediction
    
    def predict_batch(
        self,
        target_table_name: str,
        test_primary_keys_file: str,
        batch_size: int = 10000
    ) -> str:
        """
        Make batch predictions for entities listed in a file.
        
        Args:
            target_table_name: Name of the target table
            test_primary_keys_file: Path to CSV file containing primary key values
            batch_size: Maximum number of samples to process at once
            
        Returns:
            Path to the generated CSV file with predictions
        """
        logger.info(f"Starting batch prediction for {target_table_name}")
        logger.info(f"Batch size: {batch_size}")
        
        # Read primary keys from CSV file
        primary_keys_df = pd.read_csv(test_primary_keys_file)
        
        # Extract the Series (should be a single column)
        if len(primary_keys_df.columns) != 1:
            raise ValueError(f"CSV file should contain exactly one column with primary keys, found {len(primary_keys_df.columns)} columns")
        
        primary_keys = primary_keys_df.iloc[:, 0].tolist()
        
        # Make batch prediction
        predictions = self.manager.predict(target_table_name, primary_keys)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            '__id': primary_keys,
            '__timestamp': self.test_timestamp,
            '__pred': predictions
        })
        
        # Generate output filename
        output_filename = f"prediction_{uuid.uuid4().hex[:8]}.csv"
        output_path = Path("outputs") / output_filename
        
        # Create outputs directory if it doesn't exist
        output_path.parent.mkdir(exist_ok=True)
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Batch prediction completed. Results saved to: {output_path}")
        return str(output_path)
    
    def shap(
        self,
        target_table_name: str,
        test_primary_key_value: Union[str, int, float],
        shap_algorithm: str = "permutation",
        top_features: int = 10
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for the given entity.
        
        Args:
            target_table_name: Name of the target table
            test_primary_key_value: Primary key value for the entity to explain
            shap_algorithm: SHAP algorithm to use
            top_features: Number of top features to report
            
        Returns:
            Dictionary containing SHAP explanations
        """
        logger.info(f"Generating SHAP explanation for {target_table_name}, entity: {test_primary_key_value}")
        
        # Delegate to TabPFNManager
        shap_result = self.manager.shap(
            target_table_name, str(test_primary_key_value), shap_algorithm, top_features
        )
        
        logger.info(f"SHAP explanation completed for {target_table_name}")
        return shap_result
    
    def get_predictor_status(self) -> Dict[str, Any]:
        """Get status information about all predictors."""
        status = {}
        for target_table in self.manager.dfs_features.keys():
            status[target_table] = {
                'is_fitted': target_table in self.manager.predictors and self.manager.predictors[target_table].predictor is not None,
                'cached_timestamps_count': len(self.manager.cached_timestamps.get(target_table, [])),
                'data_loaded': True  # Always true since we load all data at init
            }
        return status
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get all metadata needed by agents for prompt rendering.
        
        Returns:
            Dictionary containing schema, table info, variables, timestamps, etc.
        """
        logger.info("Gathering metadata for agents")
        
        # Get REPL variables and available packages
        repl_variables = self.repl_manager.get_repl_variables()
        
        # Get predictor status
        predictor_status = self.get_predictor_status()
        
        # Get available target tables
        target_tables = self.get_available_target_tables()
        
        # Get cached timestamps for each target table
        cached_timestamps = {
            target_table: f"{target_table}_timestamps"
            for target_table in target_tables
        }
        
        # Extract table information from dataset
        table_info = self._extract_table_info()
        
        metadata = {
            "schema_yaml": self.schema_yaml,
            "table_info": table_info,
            "available_packages": self._get_available_packages(),
            "local_variables": repl_variables,
            "cached_timestamps": cached_timestamps,
            "predictor_status": predictor_status,
            "test_timestamp": self.test_timestamp,
            "target_tables": target_tables,
        }
        
        logger.info(f"Metadata gathered: {len(target_tables)} target tables, {len(repl_variables)} variables")
        return metadata
    
    def _extract_tables_from_dataset(self) -> Dict[str, pd.DataFrame]:
        """Extract tables from the dataset."""
        # For the simplified version, we'll create mock tables from DFS features
        # This is a placeholder - in practice, you might want to load original tables
        tables = {}
        
        for target_table, dfs_data in self.manager.dfs_features.items():
            # Create a sample table from DFS data (first few rows)
            sample_data = dfs_data.head(100)  # Limit to 100 rows for REPL
            tables[target_table] = sample_data
            logger.debug(f"Extracted table '{target_table}' with shape {sample_data.shape}")
        
        logger.info(f"Extracted {len(tables)} tables from dataset")
        return tables
    
    def _extract_table_info(self) -> str:
        """Extract and format table information."""
        tables = self._extract_tables_from_dataset()
        return self._get_table_info(tables)
    
    def _get_table_info(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Get structured information about tables (moved from utils.py)."""
        table_info = {}
        for table_name, df in tables.items():
            sample_df = df.head(3)
            sample_str = sample_df.to_string(index=False, max_colwidth=30)
            table_info[table_name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'sample_str': sample_str
            }
        return table_info
    
    def _get_available_packages(self) -> List[str]:
        """Get list of available Python packages in the REPL."""
        # Return a standard list of packages that are typically available
        return [
            "pandas", "numpy", "matplotlib", "seaborn", "sklearn",
            "scipy", "datetime", "json", "os", "sys", "pathlib"
        ]
    
    def _load_schema_yaml(self) -> str:
        """Load the schema YAML file as a string."""
        schema_yaml_path = Path(self.dataset_path) / "metadata.yaml"
        with open(schema_yaml_path, 'r') as f:
            schema_content = f.read()
        logger.info(f"Loaded schema from: {schema_yaml_path}")
        return schema_content
    
    def close(self):
        """Clean up resources."""
        if self.repl_manager is not None:
            self.repl_manager.close()
        # TabPFNManager doesn't need explicit cleanup currently