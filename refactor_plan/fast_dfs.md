# FastDFS Task Removal Design Document

## Executive Summary

This document outlines the design for refactoring FastDFS to remove the concept of "Tasks" and simplify Deep Feature Synthesis (DFS) to focus on augmenting tables (dataframes) with features generated from Relational Database (RDB) structures. The goal is to make DFS more intuitive by treating it as a process of feature engineering on target tables using information from related tables in an RDB.

## Current Architecture Analysis

### Current Task-Centric Design

The current FastDFS architecture is built around "Tasks" which represent machine learning problems:

```yaml
# Current metadata.yaml structure
tables:                    # Base data tables
- name: user
  columns: [user_id, user_feature_0]
- name: item  
  columns: [item_id, item_feature_0]
- name: interaction
  columns: [user_id, item_id, timestamp]

tasks:                     # ML tasks with train/val/test splits
- name: linkpred
  columns: [user_id, item_id, timestamp, label]
  shared_schema: interaction.*
  target_column: label
  task_type: classification
```

**Current DFS Process:**
1. Load RDB tables + task tables (with train/val/test splits)
2. Create featuretools EntitySet from combined data
3. Generate features for task tables using aggregations from RDB
4. Output augmented task datasets

**Issues with Current Design:**
- **Conceptual complexity**: DFS is inherently about feature engineering, not ML tasks
- **Data leakage concerns**: Mixing train/val/test data in the same pipeline
- **Limited flexibility**: Tied to specific task formats and train/val/test splits
- **Unnecessary abstraction**: Tasks add complexity when users just want features on tables

## New Architecture: Table-Centric DFS

### Core Principle

> **DFS is a feature augmentation process**: Given a target table and an RDB containing related tables, generate new features for the target table by aggregating information across relationships.

### New Conceptual Model

```python
# Instead of: "Run DFS on a task"
dataset = load_rdb_dataset(path)
dfs_processor.run(dataset, output_path)  # Processes all tasks

# New approach: "Augment a table with DFS features"
rdb = load_rdb(path)                     # Load just the relational database
augmented_table = dfs_processor.compute_features(
    rdb=rdb,
    target_table="interaction", 
    target_key_columns=["user_id", "item_id"],
    cutoff_time_column="timestamp",
    max_depth=2
)
```

### Key Changes

1. **Remove Task concept entirely** from dataset definitions
2. **Separate RDB loading** from target table specification  
3. **Runtime target specification** rather than pre-defined tasks
4. **Direct table augmentation** rather than dataset transformation
5. **Flexible relationship specification** at computation time

## New Dataset Definition

### Simplified RDB Metadata

```yaml
# New simplified metadata.yaml
dataset_name: ecommerce_rdb
tables:
- name: user
  source: data/user.npz
  columns:
  - name: user_id
    dtype: primary_key
  - name: user_feature_0
    dtype: float
  - name: registration_date
    dtype: datetime
    
- name: item
  source: data/item.npz 
  columns:
  - name: item_id
    dtype: primary_key
  - name: item_feature_0
    dtype: float
  - name: category
    dtype: category
    
- name: interaction
  source: data/interaction.npz
  columns:
  - name: user_id
    dtype: foreign_key
    link_to: user.user_id
  - name: item_id 
    dtype: foreign_key
    link_to: item.item_id
  - name: timestamp
    dtype: datetime
  - name: rating
    dtype: float
  time_column: timestamp

# Relationships are inferred from foreign keys
# No tasks section - tasks are defined at runtime
```

### Benefits of New Format

- **Cleaner separation**: RDB structure vs. ML problem definition
- **Reusability**: Same RDB can be used for multiple different feature engineering tasks
- **Simplicity**: No complex task metadata or shared schemas
- **Standard format**: Follows typical relational database conventions

## New Interface Design

### 1. RDB Loading Interface

```python
from fastdfs import RDBDataset

class RDBDataset:
    """Represents a relational database for feature engineering."""
    
    def __init__(self, path: Path):
        self.path = path
        self.metadata = self._load_metadata()
        self.tables = self._load_tables()
    
    @property 
    def table_names(self) -> List[str]:
        """Get list of table names."""
        
    def get_table(self, name: str) -> pd.DataFrame:
        """Get a table as a pandas DataFrame."""
        
    def get_table_metadata(self, name: str) -> RDBTableSchema:
        """Get metadata for a specific table."""
        
    def get_relationships(self) -> List[Tuple[str, str, str, str]]:
        """Get relationships as (child_table, child_col, parent_table, parent_col)."""
        
    def create_new_with_tables(self, new_tables: Dict[str, pd.DataFrame]) -> 'RDBDataset':
        """Create new RDBDataset with updated tables (for transforms)."""
        
    @property
    def sqlalchemy_metadata(self) -> sqlalchemy.MetaData:
        """Get SQLAlchemy metadata for the RDB."""
```

### 2. DFS Engine Interface

```python
from fastdfs import DFSEngine
import pydantic
from typing import List, Optional, Dict, Any

class DFSConfig(pydantic.BaseModel):
    """
    Configuration model for Deep Feature Synthesis parameters.
    
    This class defines all the configurable parameters for the DFS process,
    including aggregation primitives, depth limits, and engine selection.
    
    Attributes:
        agg_primitives: List of aggregation primitive names to use
        max_depth: Maximum depth for feature generation  
        use_cutoff_time: Whether to use temporal cutoff times
        engine: Name of the DFS engine to use for computation
        engine_path: Optional path for engine-specific configuration
    """
    agg_primitives: List[str] = [
        "max",
        "min",
        "mean",
        "count",
        "mode",
        "std",
    ]
    max_depth: int = 2
    use_cutoff_time: bool = True
    engine: str = "featuretools"
    engine_path: Optional[str] = "/tmp/duck.db"

class DFSEngine:
    """Base class for DFS computation engines."""
    
    def __init__(self, config: DFSConfig):
        self.config = config
    
    def compute_features(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute DFS features for a target dataframe using RDB context.
        
        Args:
            rdb: The relational database providing context for feature generation
            target_dataframe: DataFrame to augment with features (doesn't need to exist in RDB)
            key_mappings: Map from target_dataframe columns to RDB primary keys
                         e.g., {"user_id": "user.user_id", "item_id": "item.item_id"}
            cutoff_time_column: Column name in target_dataframe for temporal cutoff (optional)
            config_overrides: Dictionary of config parameters to override for this computation
            
        Returns:
            DataFrame with original target_dataframe data plus generated features
        """
        # Merge config overrides
        effective_config = self.config.copy()
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(effective_config, key):
                    setattr(effective_config, key, value)
        
        return self._compute_features_impl(
            rdb, target_dataframe, key_mappings, cutoff_time_column, effective_config
        )
    
    @abc.abstractmethod
    def _compute_features_impl(
        self,
        rdb: RDBDataset,
        target_dataframe: pd.DataFrame,
        key_mappings: Dict[str, str],
        cutoff_time_column: Optional[str],
        config: DFSConfig
    ) -> pd.DataFrame:
        """Implementation-specific feature computation."""
        pass
```

### 3. Transform Interface

```python
from fastdfs import RDBTransform

class RDBTransform:
    """Base class for RDB transformations - simplified composable operations."""
    
    def __call__(self, rdb: RDBDataset) -> RDBDataset:
        """
        Apply transformation to RDB and return new RDB.
        
        Simplified interface - no separate fit/transform phases.
        Each transform is a pure function: RDB -> RDB
        """
        pass

class TableTransform:
    """Transform that operates on individual tables within an RDB."""
    
    def __call__(self, table: pd.DataFrame, table_metadata: RDBTableSchema) -> pd.DataFrame:
        """Apply transformation to a single table."""
        pass

class ColumnTransform:
    """Transform that operates on specific columns matching criteria."""
    
    def applies_to(self, column_metadata: DBBColumnSchema) -> bool:
        """Check if this transform should be applied to a column."""
        pass
    
    def __call__(self, column: pd.Series, column_metadata: DBBColumnSchema) -> pd.DataFrame:
        """
        Transform a column, potentially outputting multiple new columns.
        Returns DataFrame with new columns to add/replace.
        """
        pass

class RDBTransformPipeline:
    """Pipeline of RDB transformations."""
    
    def __init__(self, transforms: List[RDBTransform]):
        self.transforms = transforms
    
    def __call__(self, rdb: RDBDataset) -> RDBDataset:
        """Apply all transforms in sequence."""
        result = rdb
        for transform in self.transforms:
            result = transform(result)
        return result

# Wrapper to apply table/column transforms to RDB
class RDBTransformWrapper(RDBTransform):
    """Wrapper to apply TableTransform or ColumnTransform to entire RDB."""
    
    def __init__(self, inner_transform: Union[TableTransform, ColumnTransform]):
        self.inner_transform = inner_transform
    
    def __call__(self, rdb: RDBDataset) -> RDBDataset:
        """Apply inner transform to all applicable tables/columns in RDB."""
        new_tables = {}
        
        for table_name, table_df in rdb.tables.items():
            table_metadata = rdb.get_table_metadata(table_name)
            
            if isinstance(self.inner_transform, TableTransform):
                # Apply to entire table
                new_tables[table_name] = self.inner_transform(table_df, table_metadata)
                
            elif isinstance(self.inner_transform, ColumnTransform):
                # Apply to applicable columns
                new_table = table_df.copy()
                
                for col_schema in table_metadata.columns:
                    if self.inner_transform.applies_to(col_schema):
                        col_name = col_schema.name
                        new_cols_df = self.inner_transform(table_df[col_name], col_schema)
                        
                        # Replace/add columns from transform result
                        for new_col_name, new_col_data in new_cols_df.items():
                            new_table[new_col_name] = new_col_data
                        
                        # Optionally remove original column if transform replaces it
                        if hasattr(self.inner_transform, 'replaces_original') and self.inner_transform.replaces_original:
                            new_table = new_table.drop(columns=[col_name])
                
                new_tables[table_name] = new_table
        
        return rdb.create_new_with_tables(new_tables)
```

### 4. High-Level API

```python
# Simple feature augmentation with external target dataframe
import fastdfs
import pandas as pd

# Load RDB (no tasks)
rdb = fastdfs.load_rdb("path/to/rdb")

# Load or create target dataframe (doesn't need to exist in RDB)
target_df = pd.read_parquet("user_item_pairs.parquet")
# target_df has columns: user_id, item_id, timestamp

# Augment target dataframe with DFS features from RDB
augmented_df = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings={
        "user_id": "user.user_id",     # Map target_df.user_id to RDB user table
        "item_id": "item.item_id"      # Map target_df.item_id to RDB item table  
    },
    cutoff_time_column="timestamp",
    config_overrides={
        "max_depth": 2,
        "engine": "dfs2sql",
        "agg_primitives": ["count", "mean", "max", "min"]
    }
)

# Method 2: Using custom DFS configuration
dfs_config = fastdfs.DFSConfig(
    max_depth=3,
    engine="featuretools",
    agg_primitives=["count", "sum", "mean", "std"],
    use_cutoff_time=True,
)

augmented_df = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=target_df,
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="timestamp",
    config=dfs_config
)

# Apply transforms to RDB (simplified - no fit/transform)
transform_pipeline = fastdfs.RDBTransformPipeline([
    fastdfs.transforms.CanonicalizeNumeric(),
    fastdfs.transforms.FeaturizeDatetime(methods=["year", "month", "hour"]),
    fastdfs.transforms.NormalizeNumeric()
])
transformed_rdb = transform_pipeline(rdb)

# Combined transform + DFS workflow
augmented_df = fastdfs.compute_dfs_features(
    rdb=transformed_rdb,
    target_dataframe=target_df,
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    config_overrides={"max_depth": 2}
)

# Example: Generate features for ML training data
train_df = pd.read_parquet("train_interactions.parquet")
train_with_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=train_df,
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="interaction_time",
    config_overrides={"max_depth": 2, "use_cutoff_time": True}
)

# Example: Generate features for new prediction data
new_predictions_df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "item_id": [100, 200, 300],
    "prediction_time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
})
prediction_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=new_predictions_df,
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="prediction_time",
    config=dfs_config
)
```


## Running Examples

### Example 1: E-commerce Recommendation Features

```python
import fastdfs
import pandas as pd

# Load e-commerce RDB
rdb = fastdfs.load_rdb("ecommerce_rdb/")
# RDB contains: users, items, interactions, purchases, reviews

# Create target dataframe for user-item pairs to score
candidate_pairs = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3],
    "item_id": [100, 101, 100, 102, 103],
    "scoring_time": pd.to_datetime([
        "2024-01-15", "2024-01-15", "2024-01-16", 
        "2024-01-16", "2024-01-17"
    ])
})

# Generate features for these specific user-item pairs
interaction_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=candidate_pairs,
    key_mappings={
        "user_id": "user.user_id",
        "item_id": "item.item_id"
    },
    cutoff_time_column="scoring_time",
    config_overrides={
        "max_depth": 2,
        "agg_primitives": ["count", "mean", "max", "min"],
        "engine": "dfs2sql"
    }
)

# Results include:
# - user_id, item_id, scoring_time (original columns)
# - COUNT(purchases WHERE purchases.user_id = target.user_id AND purchases.timestamp < target.scoring_time)  
# - MEAN(reviews.rating WHERE reviews.item_id = target.item_id AND reviews.timestamp < target.scoring_time)
# - MAX(interactions.timestamp WHERE interactions.user_id = target.user_id AND interactions.timestamp < target.scoring_time)
# - etc.

print(f"Generated {len(interaction_features.columns)} features for {len(candidate_pairs)} user-item pairs")
interaction_features.to_parquet("recommendation_features.parquet")
```

### Example 2: Financial Transaction Analysis

```python
import fastdfs
import pandas as pd

# Load financial RDB
rdb = fastdfs.load_rdb("financial_rdb/")
# RDB contains: accounts, customers, transactions, merchants

# Apply preprocessing transforms (simplified - no fit/transform)
preprocessor = fastdfs.RDBTransformPipeline([
    fastdfs.transforms.CanonicalizeDatetime(),
    fastdfs.transforms.FeaturizeDatetime(methods=["hour", "day_of_week", "month"]),
    fastdfs.transforms.NormalizeAmounts()
])
clean_rdb = preprocessor(rdb)

# Load specific transactions to analyze (e.g., suspicious transactions)
suspicious_transactions = pd.read_parquet("flagged_transactions.parquet")
# Contains: transaction_id, analysis_timestamp

# Generate features for these specific transactions
transaction_features = fastdfs.compute_dfs_features(
    rdb=clean_rdb,
    target_dataframe=suspicious_transactions,
    key_mappings={
        "transaction_id": "transactions.transaction_id"
    },
    cutoff_time_column="analysis_timestamp",
    config_overrides={
        "max_depth": 3,
        "include_entities": ["accounts", "customers"],  # Focus on these relationships
        "ignore_entities": ["merchants"],  # Skip merchant aggregations for this analysis
        "agg_primitives": ["count", "sum", "mean", "std", "time_since_last"]
    }
)

# Results include features like:
# - Account balance history features
# - Customer spending patterns  
# - Transaction frequency features
# - Time-based features
```

### Example 3: Custom DFS Configuration

```python
import fastdfs
import pandas as pd

rdb = fastdfs.load_rdb("retail_rdb/")

# Create custom DFS engine with specific configuration
dfs_engine = fastdfs.DFSEngine.create("dfs2sql", config={
    "max_depth": 3,
    "agg_primitives": ["count", "sum", "mean", "mode"],
    "engine_specific": {
        "use_approximate_aggregations": True,
        "parallel_workers": 4
    }
})

# Load order data to analyze (could be from any source)
target_orders = pd.read_csv("orders_to_analyze.csv")
# Contains: order_id, customer_id, analysis_date

# Compute features with custom mappings
features = dfs_engine.compute_features(
    rdb=rdb,
    target_dataframe=target_orders,
    key_mappings={
        "order_id": "orders.order_id",
        "customer_id": "customers.customer_id"
    },
    cutoff_time_column="analysis_date",
)

# Example 4: ML Training Pipeline Integration
import sklearn.model_selection
import sklearn.ensemble

# Load training data (separate from RDB)
training_data = pd.read_parquet("training_interactions.parquet")
labels = training_data["converted"].values
train_df, test_df = sklearn.model_selection.train_test_split(training_data, test_size=0.2)

# Generate features for training set
train_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=train_df[["user_id", "item_id", "interaction_time"]],
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="interaction_time",
    max_depth=2
)

# Generate features for test set (using same RDB, different target data)
test_features = fastdfs.compute_dfs_features(
    rdb=rdb,
    target_dataframe=test_df[["user_id", "item_id", "interaction_time"]],
    key_mappings={"user_id": "user.user_id", "item_id": "item.item_id"},
    cutoff_time_column="interaction_time",
    max_depth=2
)

# Train model
model = sklearn.ensemble.RandomForestClassifier()
model.fit(train_features.drop(columns=["user_id", "item_id", "interaction_time"]), 
          train_df["converted"])

# Predict on test set
predictions = model.predict(test_features.drop(columns=["user_id", "item_id", "interaction_time"]))
```

## Benefits of New Architecture

### 1. Conceptual Simplicity
- **Intuitive**: DFS = "add features to a table using related data"
- **No ML coupling**: Separates feature engineering from ML task definition
- **Flexible**: Same RDB can serve multiple feature engineering needs

### 2. Technical Advantages  
- **No data leakage**: No mixing of train/val/test in feature computation
- **Better performance**: Can optimize for single-table output
- **Composability**: Easy to chain transforms and feature computations
- **Extensibility**: Simple to add new engines and primitives

### 3. User Experience
- **Cleaner API**: Direct mapping from intent to code
- **Less configuration**: No complex task metadata files
- **Better tooling**: Can build table/column browsers, feature explorers
- **Interoperability**: Easy integration with pandas/SQL workflows

### 4. Development Benefits
- **Simpler testing**: Test feature computation independently
- **Clearer interfaces**: Well-defined inputs and outputs
- **Reduced complexity**: Fewer abstractions and edge cases
- **Better maintainability**: Focused codebase with clear responsibilities

## Implementation Risks and Mitigations

### Risk 1: Breaking Changes for Existing Users
**Mitigation**: 
- Implement migration tools for automatic dataset conversion
- Provide compatibility layer for old API during transition period
- Clear migration guide with before/after examples

### Risk 2: Loss of Train/Val/Test Split Handling
**Mitigation**:
- Provide utilities for applying features to pre-split data
- Document best practices for temporal splitting with cutoff times
- Consider adding split-aware feature computation as optional feature

### Risk 3: Performance Regressions
**Mitigation**:
- Benchmark new implementation against current system
- Optimize for single-table output (current system optimizes for task datasets)
- Leverage better caching since we're not duplicating data across tasks

### Risk 4: Feature Engineering Workflow Disruption  
**Mitigation**:
- Extensive testing with realistic datasets
- Gradual rollout with both systems available during transition
- Community feedback and iteration before removing old system