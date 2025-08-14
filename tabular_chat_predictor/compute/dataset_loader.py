import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Set

def load_dataset(dataset_path: str):
    """Load rel-*-input format with single DFS files per target table."""
    base_path = Path(dataset_path)
    
    # Load DFS features from __dfs__/*.npz (single file per target table)
    dfs_features = {}
    dfs_dir = base_path / "__dfs__"
    for dfs_file in dfs_dir.glob("*.npz"):
        target_table = dfs_file.stem  # customer.npz -> customer
        data = np.load(dfs_file, allow_pickle=True)
        # Single DataFrame containing all temporal snapshots
        dfs_features[target_table] = pd.DataFrame({k: data[k] for k in data.keys()})
    
    return dfs_features

def get_primary_key_column(dataset_path: str, target_table: str) -> str:
    """Get the primary key column name for a specific target table from metadata."""
    metadata_path = Path(dataset_path) / "metadata.yaml"
    
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    # Find the target table in metadata
    for table in metadata['tables']:
        if table['name'] == target_table:
            # Find the primary key column
            for col in table['columns']:
                if col['dtype'] == 'primary_key':
                    return col['name']
    
    # If we reach here, something is wrong with the metadata
    raise ValueError(f"No primary key found for target table '{target_table}' in metadata")

def get_exclusion_columns(dataset_path: str, target_table: str) -> Set[str]:
    """Get columns to exclude from features for a specific target table."""
    metadata_path = Path(dataset_path) / "metadata.yaml"
    
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    exclude_cols = set()
    
    # Find the target table in metadata
    target_table_meta = None
    for table in metadata['tables']:
        if table['name'] == target_table:
            target_table_meta = table
            break
    
    if target_table_meta:
        # Extract primary keys and foreign keys from target table
        for col in target_table_meta['columns']:
            col_name = col['name']
            col_dtype = col['dtype']
            
            if col_dtype == 'primary_key':
                exclude_cols.add(col_name)
            elif col_dtype == 'foreign_key':
                exclude_cols.add(col_name)
    
    # Add reserved columns added by convert_dataset.py
    # __timestamp__ is the DFS computation timestamp, not original table timestamps
    exclude_cols.update(['__label__', '__timestamp__'])
    
    return exclude_cols