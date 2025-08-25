# Stage 2: Task-Scoped DFS Feature Engineering Workflow

## Overview

The `generate_task_features` function is the second stage in the MCP server's prediction pipeline. Its primary responsibility is to take the `TaskTables` produced by Stage 1 and, using the server's persistent `RDBDataset` and DFS engine, generate rich feature sets for both training and test data. These feature sets form the foundation for the subsequent model training and prediction stage.

This process leverages the FastDFS library to perform automated feature engineering through Deep Feature Synthesis (DFS). It transforms the basic entity features from Stage 1 into a comprehensive set of derived features by aggregating information across relationships in the relational database.

## API Signature

As defined in the interface design document:

```python
@dataclass
class FeatureArtifacts:
    """Feature artifacts - clean data only."""
    train_features_table: pd.DataFrame  # Features for training rows
    test_features_table: pd.DataFrame   # Features for test rows

async def generate_task_features(
    task_tables: TaskTables,
    server: 'TabularPredictionMCPServer'
) -> FeatureArtifacts:
    """
    Generate DFS features using server's persistent RDB and current query.
    
    Args:
        task_tables: Train/test tables from Stage 1
        server: MCP server instance (accesses persistent rdb_dataset and current_query_spec)
        
    Returns:
        FeatureArtifacts with DFS features for train/test rows
    """
```

## Detailed Workflow Steps

The feature generation process can be broken down into the following steps:

### 1. Extract Inputs from Server Context

The function begins by accessing the key pieces of state from the server instance:
- **`rdb_dataset`**: The persistent, in-memory representation of the entire relational database, loaded at server startup.
- **`current_query_spec`**: The `QuerySpec` object for the current client request, containing parameters like `dfs_depth` that control feature generation.
- **`dfs_config`**: The server's DFS configuration, which may be overridden by query-specific parameters.

### 2. Prepare Target DataFrames for Feature Generation

The function utilizes the train and test tables from Stage 1 directly for feature generation. The `__id` and `__timestamp` columns within these tables are used to map to the RDB and apply temporal cutoffs, respectively.

### 3. Apply Transform Pipeline to RDB

Before feature generation, the RDB dataset may need preprocessing:
- **Handle Dummy Tables**: Remove or transform tables that don't contribute to feature generation.
- **Featurize Datetime Columns**: Extract useful components (year, month, day, etc.) from datetime columns.
- **Filter Redundant Columns**: Remove columns that would create redundant or noisy features.

### 4. Generate Features Using FastDFS

This is the core step where the DFS algorithm is applied:
- **Configure DFS Engine**: Set up the DFS engine (Featuretools or DFS2SQL) with the appropriate parameters.
- **Map Entity IDs to RDB Tables**: Create mappings between the task table entity IDs and the corresponding primary keys in the RDB.
- **Apply Cutoff Time Constraints**: Ensure temporal consistency by only using data available up to each entity's timestamp.
- **Execute Feature Generation**: Run the DFS algorithm to generate features for both train and test sets.

### 5. Align Feature Sets

Ensure consistency between train and test feature sets:
- **Identify Common Features**: Determine the set of features that can be generated for both train and test data.
- **Handle Missing Values**: Apply consistent strategies for handling missing values across both sets.
- **Normalize Feature Names**: Ensure feature names are consistent and interpretable.

### 6. Construct and Return `FeatureArtifacts`

The generated feature DataFrames are encapsulated in a `FeatureArtifacts` dataclass and returned, completing Stage 2 of the pipeline.

## Sub-API Pseudocode and Examples

Here is a breakdown of the internal functions that `generate_task_features` would call, complete with clear signatures, pseudocode, and examples.

---

### Main Function: `generate_task_features`

This is the main entry point that orchestrates the sub-API calls.

**Pseudocode:**
```python
async def generate_task_features(task_tables: TaskTables, server: 'TabularPredictionMCPServer') -> FeatureArtifacts:
    # 1. Get context from the server
    query = server.current_query_spec
    rdb = server.rdb_dataset
    dfs_config = _merge_dfs_configs(server.dfs_config, query)
    
    # 2. Apply transform pipeline to RDB
    transform_pipeline = _create_transform_pipeline()
    transformed_rdb = transform_pipeline(rdb)
    
    # 3. Generate key mappings based on target table
    key_mappings = _generate_key_mappings(
        query.target_table,
        rdb.get_table_metadata(query.target_table)
    )
    
    # 4. Generate features for train set
    train_features = _generate_features(
        transformed_rdb=transformed_rdb,
        target_df=task_tables.train_table,
        key_mappings=key_mappings,
        cutoff_time_column="__timestamp",
        dfs_config=dfs_config
    )
    
    # 5. Generate features for test set with same feature set
    test_features = _generate_features(
        transformed_rdb=transformed_rdb,
        target_df=task_tables.test_table,
        key_mappings=key_mappings,
        cutoff_time_column="__timestamp",
        dfs_config=dfs_config,
        feature_names=train_features.columns
    )
    
    # 7. Return the final FeatureArtifacts object
    return FeatureArtifacts(
        train_features_table=train_features,
        test_features_table=test_features
    )
```

---

### Sub-API 1: `_merge_dfs_configs`

This function merges the server's default DFS configuration with any query-specific overrides.

**Signature:**
```python
def _merge_dfs_configs(
    server_config: Dict[str, Any],
    query: QuerySpec
) -> Dict[str, Any]:
    """
    Merge server DFS config with query-specific overrides.
    
    Args:
        server_config: Server's default DFS configuration
        query: Current query specification with potential overrides
        
    Returns:
        Merged configuration dictionary
    """
```

**Pseudocode:**
```python
def _merge_dfs_configs(server_config, query):
    # Start with server config
    merged_config = server_config.copy()
    
    # Override with query-specific parameters
    if hasattr(query, 'dfs_depth'):
        merged_config['max_depth'] = query.dfs_depth
    
    # Add other query-specific overrides as needed
    
    return merged_config
```

**Input Example:**
```python
server_config = {
    'max_depth': 2,
    'engine': 'dfs2sql',
    'agg_primitives': ['count', 'mean', 'max', 'min', 'std']
}

query = QuerySpec(
    target_table="users",
    id_column="Id",
    dfs_depth=3  # Override default depth
)
```

**Output Example:**
```python
{
    'max_depth': 3,  # Overridden from query
    'engine': 'dfs2sql',
    'agg_primitives': ['count', 'mean', 'max', 'min', 'std']
}
```

---

### Sub-API 2: `_create_transform_pipeline`

This function creates a transform pipeline for preprocessing the RDB dataset before feature generation.

**Signature:**
```python
def _create_transform_pipeline() -> RDBTransformPipeline:
    """
    Create a transform pipeline for RDB preprocessing.
    
    Returns:
        RDBTransformPipeline with appropriate transforms
    """
```

**Pseudocode:**
```python
def _create_transform_pipeline():
    # Create a pipeline with standard transforms
    return RDBTransformPipeline([
        # Handle dummy tables (tables with no useful features)
        HandleDummyTable(),
        
        # Extract useful components from datetime columns
        RDBTransformWrapper(FeaturizeDatetime(
            features=["year", "month", "day", "hour", "dayofweek"]
        )),
        
        # Filter out redundant columns
        RDBTransformWrapper(FilterColumn(drop_redundant=True))
    ])
```

**Output Example:**
```python
RDBTransformPipeline([
    HandleDummyTable(),
    RDBTransformWrapper(FeaturizeDatetime(features=["year", "month", "day", "hour", "dayofweek"])),
    RDBTransformWrapper(FilterColumn(drop_redundant=True))
])
```

---

### Sub-API 3: `_generate_key_mappings`

This function generates mappings between entity IDs in the task tables and their corresponding primary keys in the RDB.

**Signature:**
```python
def _generate_key_mappings(
    target_table: str,
    target_table_schema: RDBTableSchema
) -> Dict[str, str]:
    """
    Generate key mappings for FastDFS.
    
    Args:
        target_table: Name of the target table
        target_table_schema: Schema of the target table
        
    Returns:
        Dictionary mapping task table columns to RDB primary keys
    """
```

**Pseudocode:**
```python
def _generate_key_mappings(target_table, target_table_schema):
    # Find primary key column
    primary_key = None
    for col in target_table_schema.columns:
        if col.dtype == "primary_key":
            primary_key = col.name
            break
    
    if not primary_key:
        raise ValueError(f"No primary key found in table {target_table}")
    
    # Create mapping from __id to target_table.primary_key
    return {"__id": f"{target_table}.{primary_key}"}
```

**Input Example:**
```python
target_table = "users"
target_table_schema = RDBTableSchema(
    name="users",
    source="data/users.npz",
    columns=[
        RDBColumnSchema(name="Id", dtype="primary_key"),
        RDBColumnSchema(name="DisplayName", dtype="string"),
        RDBColumnSchema(name="Reputation", dtype="int")
    ]
)
```

**Output Example:**
```python
{"__id": "users.Id"}
```

---

### Sub-API 4: `_generate_features`

This function performs the actual feature generation using FastDFS.

**Signature:**
```python
def _generate_features(
    transformed_rdb: RDBDataset,
    target_df: pd.DataFrame,
    key_mappings: Dict[str, str],
    cutoff_time_column: str,
    dfs_config: Dict[str, Any],
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate features using FastDFS.
    
    Args:
        transformed_rdb: Preprocessed RDB dataset
        target_df: Target DataFrame with entity IDs and timestamps
        key_mappings: Mappings from target columns to RDB primary keys
        cutoff_time_column: Name of the column to use for cutoff times
        dfs_config: Configuration for DFS algorithm
        feature_names: Optional list of feature names to generate (for test set)
        
    Returns:
        DataFrame with generated features
    """
```

**Pseudocode:**
```python
def _generate_features(transformed_rdb, target_df, key_mappings, cutoff_time_column, dfs_config, feature_names=None):
    # Configure feature generation parameters
    config_overrides = {
        "max_depth": dfs_config.get("max_depth", 2),
        "engine": dfs_config.get("engine", "dfs2sql"),
        "agg_primitives": dfs_config.get("agg_primitives", ["count", "mean", "max", "min", "std"]),
        "trans_primitives": dfs_config.get("trans_primitives", [])
    }
    
    # If feature_names is provided (for test set), add it to config
    if feature_names is not None:
        config_overrides["features"] = feature_names
    
    # Generate features using FastDFS
    features_df = fastdfs.compute_dfs_features(
        rdb=transformed_rdb,
        target_dataframe=target_df,
        key_mappings=key_mappings,
        cutoff_time_column=cutoff_time_column,
        config_overrides=config_overrides
    )
    
    return features_df
```

**Input Example:**
```python
transformed_rdb = RDBDataset(...)  # Preprocessed RDB
target_df = pd.DataFrame({
    '__id': [1, 2, 3],
    '__timestamp': [datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
    'DisplayName': ['Alice', 'Bob', 'Charlie'],
    'Reputation': [100, 200, 50]
})
key_mappings = {"__id": "users.Id"}
dfs_config = {
    'max_depth': 3,
    'engine': 'dfs2sql',
    'agg_primitives': ['count', 'mean', 'max', 'min', 'std']
}
```

**Output Example:**
```python
pd.DataFrame({
    '__id': [1, 2, 3],
    '__timestamp': [datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
    'DisplayName': ['Alice', 'Bob', 'Charlie'],
    'users.Reputation': [100, 200, 50],
    'COUNT(posts WHERE posts.OwnerUserId = users.Id)': [5, 2, 10],
    'MEAN(posts.Score WHERE posts.OwnerUserId = users.Id)': [3.2, 4.5, 2.1],
    'COUNT(comments WHERE comments.UserId = users.Id)': [12, 5, 8],
    # ... many more generated features ...
})
```

---

## Complete Workflow Example

Here's a comprehensive example showing how all sub-APIs work together:

### Input: User Engagement Prediction Query

```python
# Server state
server.rdb_dataset = RDBDataset("stackexchange_data")  # Loaded at startup
server.dfs_config = {
    'max_depth': 2,
    'engine': 'dfs2sql',
    'agg_primitives': ['count', 'mean', 'max', 'min', 'std']
}
server.current_query_spec = QuerySpec(
    target_table="users",
    entity_ids=[2666],
    id_column="Id",
    ts_current=datetime(2021, 1, 1),
    task_type="classification",
    dfs_depth=3,  # Override default depth
    query_id="user_engagement_classification_2666"
)

# Input from Stage 1
task_tables = TaskTables(
    train_table=pd.DataFrame({
        '__id': [1, 5, 8, 12, ...],                    # ~15,000 training examples
        '__timestamp': [datetime(2021, 1, 1), ...],    # Various historical timestamps
        '__label': [1, 0, 1, 0, ...],                  # Engagement labels
        'DisplayName': ['Alice', 'Bob', ...],          # User features
        'Reputation': [100, 200, 50, ...]              # User features
    }),
    
    test_table=pd.DataFrame({
        '__id': [2666],                                # Target user
        '__timestamp': [datetime(2021, 1, 1)],         # Current timestamp
        '__label': [None],                             # To be predicted
        'DisplayName': ['TargetUser'],                 # User features
        'Reputation': [150]                            # User features
    })
)
```

### Step-by-Step Execution:

**Step 1: Merge DFS Configurations**
```python
dfs_config = _merge_dfs_configs(
    server_config=server.dfs_config,
    query=server.current_query_spec
)
# Result: {'max_depth': 3, 'engine': 'dfs2sql', 'agg_primitives': ['count', 'mean', 'max', 'min', 'std']}
```

**Step 2: Create and Apply Transform Pipeline**
```python
transform_pipeline = _create_transform_pipeline()
transformed_rdb = transform_pipeline(server.rdb_dataset)
# Result: Preprocessed RDB with datetime features and filtered columns
```

**Step 3: Generate Key Mappings**
```python
key_mappings = _generate_key_mappings(
    target_table=server.current_query_spec.target_table,
    target_table_schema=server.rdb_dataset.get_table_metadata("users")
)
# Result: {"__id": "users.Id"}
```

**Step 4: Generate Features for Train Set**
```python
train_features = _generate_features(
    transformed_rdb=transformed_rdb,
    target_df=task_tables.train_table,
    key_mappings=key_mappings,
    cutoff_time_column="__timestamp",
    dfs_config=dfs_config
)
# Result: DataFrame with ~15,000 rows and 100+ feature columns
```

**Step 5: Generate Features for Test Set**
```python
test_features = _generate_features(
    transformed_rdb=transformed_rdb,
    target_df=task_tables.test_table,
    key_mappings=key_mappings,
    cutoff_time_column="__timestamp",
    dfs_config=dfs_config,
    feature_names=train_features.columns
)
# Result: DataFrame with 1 row (user 2666) and the same feature columns as train_features
```

### Final Output: FeatureArtifacts

```python
FeatureArtifacts(
    train_features_table=pd.DataFrame({
        '__id': [1, 5, 8, 12, ...],                    # ~15,000 training examples
        '__timestamp': [datetime(2021, 1, 1), ...],    # Various historical timestamps
        '__label': [1, 0, 1, 0, ...],
        'DisplayName': ['Alice', 'Bob', ...],
        'users.Reputation': [100, 200, 50, ...],       # Original features
        # Generated features (100+ columns)
        'COUNT(posts WHERE posts.OwnerUserId = users.Id)': [5, 2, 10, ...],
        'MEAN(posts.Score WHERE posts.OwnerUserId = users.Id)': [3.2, 4.5, 2.1, ...],
        'COUNT(comments WHERE comments.UserId = users.Id)': [12, 5, 8, ...],
        'MEAN(comments.Score WHERE comments.UserId = users.Id)': [2.1, 1.5, 3.2, ...],
        'COUNT(votes WHERE votes.UserId = users.Id)': [20, 15, 5, ...],
        # ... many more features ...
    }),
    
    test_features_table=pd.DataFrame({
        '__id': [2666],                                # Target user
        '__timestamp': [datetime(2021, 1, 1)],         # Current timestamp
        '__label': [None],
        'DisplayName': ['TargetUser'],
        'users.Reputation': [150],                     # Original features
        # Same generated features as train set
        'COUNT(posts WHERE posts.OwnerUserId = users.Id)': [7],
        'MEAN(posts.Score WHERE posts.OwnerUserId = users.Id)': [4.1],
        'COUNT(comments WHERE comments.UserId = users.Id)': [9],
        'MEAN(comments.Score WHERE comments.UserId = users.Id)': [2.8],
        'COUNT(votes WHERE votes.UserId = users.Id)': [18],
        # ... many more features ...
    })
)
```

## FastDFS Optimizations in MCP Server Context

The `generate_task_features` function leverages several key optimizations provided by the FastDFS library and the MCP server architecture:

### 1. **Persistent RDB and Transform Pipeline**
- **Before**: Traditional DFS implementations reload data and recreate entity sets for each feature generation run.
- **After**: The MCP server maintains the RDB in memory, and the transform pipeline can be applied once and reused across multiple queries.

### 2. **Engine Selection Based on Data Size**
- **Before**: Fixed feature generation engine regardless of dataset size.
- **After**: Can dynamically select between Featuretools (for small datasets) and DFS2SQL (for large datasets) based on query characteristics.

### 3. **Temporal Consistency with Cutoff Times**
- **Before**: Feature generation often suffers from data leakage by using future data.
- **After**: FastDFS's cutoff time mechanism ensures that only data available up to each entity's timestamp is used for feature generation.

### 4. **Feature Reuse Across Queries**
- **Before**: Features are regenerated from scratch for each prediction task.
- **After**: The MCP server can potentially cache feature definitions and reuse them across similar queries.

### 5. **Consistent Feature Sets**
- **Before**: Train and test sets might have different feature sets due to data availability.
- **After**: The implementation ensures that test sets have exactly the same feature columns as training sets.

### 6. **Query-Specific Configuration**
- **Before**: Fixed feature generation parameters for all tasks.
- **After**: Each query can specify custom parameters like `dfs_depth` to tailor feature generation to the specific prediction task.

### 7. **Parallel Processing**
- **Before**: Sequential feature generation for train and test sets.
- **After**: The async implementation allows for potential parallel processing of feature generation tasks.

These optimizations enable the system to generate rich, meaningful features efficiently, even for large datasets and complex relational structures, while maintaining the flexibility to adapt to different prediction tasks.