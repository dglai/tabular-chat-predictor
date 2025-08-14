import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor
from tabpfn_extensions.interpretability import shap as tabpfn_shap
from .dataset_loader import get_exclusion_columns, get_primary_key_column

class TabPFNPredictor:
    def __init__(self, dataset_path: str, target_table: str, use_auto=False):
        self.dataset_path = dataset_path
        self.target_table = target_table
        self.use_auto = use_auto
        self.predictor = None
        self.feature_cols = None
        self.task_type = None
        
    def fit(self, dfs_data: pd.DataFrame, task_type: str):
        # Extract features using generic exclusion function
        exclude = get_exclusion_columns(self.dataset_path, self.target_table)
        self.feature_cols = [col for col in dfs_data.columns if col not in exclude]
        self.task_type = task_type
        
        X = dfs_data[self.feature_cols]
        y = dfs_data['__label__']
        
        # Initialize TabPFN
        if task_type == 'classification':
            self.predictor = AutoTabPFNClassifier(max_time=120, device="cuda") if self.use_auto else TabPFNClassifier()
        else:
            self.predictor = AutoTabPFNRegressor(max_time=120, device="cuda") if self.use_auto else TabPFNRegressor()
            
        self.predictor.fit(X, y)
    
    def predict(self, test_data: pd.DataFrame):
        return self.predictor.predict(test_data[self.feature_cols])
    
    def predict_proba(self, test_data: pd.DataFrame):
        return self.predictor.predict_proba(test_data[self.feature_cols])
    
    def generate_shap_explanation(self, test_data: pd.DataFrame, shap_algorithm: str = "permutation", top_features: int = 10) -> Dict[str, Any]:
        """Generate SHAP explanation for test instance(s)."""
        if self.predictor is None:
            raise ValueError("Model not fitted. Call fit() before generate_shap_explanation().")
        
        # Extract test features
        X_test = test_data[self.feature_cols]
        
        # Calculate SHAP values
        shap_start_time = datetime.now()
        shap_values = tabpfn_shap.get_shap_values(
            estimator=self.predictor,
            test_x=X_test,
            attribute_names=self.feature_cols,
            algorithm=shap_algorithm
        )
        
        # Get feature importances (absolute SHAP values)
        shap_values_array = shap_values.values
        
        # Handle classification vs regression
        if self.task_type == 'classification' and len(shap_values_array.shape) == 3:
            # For binary classification, compute an average
            feature_importances = np.abs(shap_values_array[0, :, :]).mean(1)
        else:
            # For regression or when SHAP values are 2D
            feature_importances = np.abs(shap_values_array[0, :])
        
        # Get top contributing features
        top_features_idx = np.argsort(feature_importances)[-top_features:][::-1]
        top_feature_names = [self.feature_cols[i] for i in top_features_idx]
        top_feature_importances = [feature_importances[i] for i in top_features_idx]
        
        return {
            'shap_analysis': {
                'success': True,
                'feature_importances': feature_importances.tolist(),
                'top_features': top_feature_names,
                'top_feature_importances': top_feature_importances,
                'algorithm': shap_algorithm,
                'top_features_count': top_features,
            }
        }