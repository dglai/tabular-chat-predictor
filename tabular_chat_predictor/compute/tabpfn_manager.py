from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .dataset_loader import load_dataset, get_primary_key_column
from .tabpfn_predictor import TabPFNPredictor

class TabPFNManager:
    def __init__(self, dataset_path: str, test_timestamp: datetime):
        self.dataset_path = dataset_path
        self.dfs_features = load_dataset(dataset_path)
        self.test_timestamp = test_timestamp
        self.predictors = {}
        self.cached_timestamps = {}
        
        # Extract cached timestamps for each target table
        for target_table in self.dfs_features.keys():
            self.cached_timestamps[target_table] = self._extract_cached_timestamps(target_table)
    
    def fit(self, target_table: str, task_type: str, training_labels: pd.DataFrame):
        """Fit using external training labels (preserves original signature)."""
        # Get all DFS data for target table (single DataFrame with all timestamps)
        dfs_data = self.dfs_features[target_table]
        
        # Parse primary key column name from metadata
        id_col = get_primary_key_column(self.dataset_path, target_table)
        merged_data = dfs_data.merge(
            training_labels.rename(columns={'__id': id_col, '__timestamp': '__timestamp__'}),
            on=[id_col, '__timestamp__'],
            how='inner'
        )
        
        # Fit predictor using merged data with generic column exclusion
        predictor = TabPFNPredictor(self.dataset_path, target_table)
        predictor.fit(merged_data, task_type)
        self.predictors[target_table] = predictor
    
    def predict(self, target_table: str, entity_ids: list):
        """Extract latest DFS features and align timestamps for prediction."""
        # Get all DFS data for target table
        dfs_data = self.dfs_features[target_table]
        
        # Parse primary key column name from metadata
        id_col = get_primary_key_column(self.dataset_path, target_table)
        entity_mask = dfs_data[id_col].isin(entity_ids)
        entity_data = dfs_data[entity_mask]
        
        # Filter for timestamps before test_timestamp
        time_mask = pd.to_datetime(entity_data['__timestamp__']) < self.test_timestamp
        valid_data = entity_data[time_mask]
        
        # Get latest record for each entity
        latest_indices = valid_data.groupby(id_col)['__timestamp__'].idxmax()
        test_features = valid_data.loc[latest_indices]
        
        # CRITICAL: Align timestamp features with test_timestamp
        test_features_aligned = self._align_timestamp_features(test_features, self.test_timestamp)
        
        return self.predictors[target_table].predict(test_features_aligned)
    
    def shap(self, target_table: str, entity_id: str, shap_algorithm: str = "permutation", top_features: int = 10) -> Dict[str, Any]:
        """Generate SHAP explanation for a single entity."""
        if target_table not in self.predictors:
            raise ValueError(f"Target table '{target_table}' not found. Available: {list(self.predictors.keys())}")
        
        # Check if model is fitted
        if self.predictors[target_table].predictor is None:
            raise ValueError(f"Model for target table '{target_table}' is not fitted. Call fit() first.")
        
        # Get all DFS data for target table
        dfs_data = self.dfs_features[target_table]
        
        # Parse primary key column name from metadata
        id_col = get_primary_key_column(self.dataset_path, target_table)
        entity_mask = dfs_data[id_col] == entity_id
        entity_data = dfs_data[entity_mask]
        
        # Filter for timestamps before test_timestamp
        time_mask = pd.to_datetime(entity_data['__timestamp__']) < self.test_timestamp
        valid_data = entity_data[time_mask]
        
        if valid_data.empty:
            raise ValueError(f"No historical data found for entity {entity_id} before {self.test_timestamp}")
        
        # Get latest record for the entity
        latest_idx = valid_data['__timestamp__'].idxmax()
        test_features = valid_data.loc[[latest_idx]]
        
        # CRITICAL: Align timestamp features with test_timestamp
        test_features_aligned = self._align_timestamp_features(test_features, self.test_timestamp)
        
        # Generate SHAP explanation
        shap_result = self.predictors[target_table].generate_shap_explanation(
            test_features_aligned, shap_algorithm, top_features
        )
        
        # Add test info to match original signature
        shap_result['test_info'] = {
            'entity_id': entity_id,
            'timestamp': self.test_timestamp,
        }
        
        return shap_result
    
    def _align_timestamp_features(self, test_data: pd.DataFrame, test_timestamp: datetime) -> pd.DataFrame:
        """Replace timestamp-derived features with ones computed from test_timestamp."""
        test_aligned = test_data.copy()
        test_ts = pd.to_datetime(test_timestamp)
        
        # Recompute derived timestamp features from test_timestamp
        if 'YEAR(__timestamp__)' in test_aligned.columns:
            test_aligned['YEAR(__timestamp__)'] = test_ts.year
        if 'MONTH(__timestamp__)' in test_aligned.columns:
            test_aligned['MONTH(__timestamp__)'] = test_ts.month
        if 'DAY(__timestamp__)' in test_aligned.columns:
            test_aligned['DAY(__timestamp__)'] = test_ts.day
        if 'DAYOFWEEK(__timestamp__)' in test_aligned.columns:
            test_aligned['DAYOFWEEK(__timestamp__)'] = test_ts.dayofweek
        if 'TIMESTAMP___timestamp__' in test_aligned.columns:
            test_aligned['TIMESTAMP___timestamp__'] = test_ts.timestamp() * 1e9  # nanoseconds
        if '__timestamp__' in test_aligned.columns:
            test_aligned['__timestamp__'] = test_ts.timestamp() * 1e9  # nanoseconds
        
        return test_aligned
    
    def get_cached_timestamps(self, target_table: str) -> List[datetime]:
        """Get cached timestamps for PredictingAssistant."""
        return self.cached_timestamps[target_table]
    
    def _extract_cached_timestamps(self, target_table: str) -> List[datetime]:
        """Extract timestamps from DFS data (single file contains all timestamps)."""
        dfs_data = self.dfs_features[target_table]
        timestamps = pd.to_datetime(dfs_data['__timestamp__']).unique()
        timestamps = timestamps[timestamps < self.test_timestamp]
        return sorted(timestamps)