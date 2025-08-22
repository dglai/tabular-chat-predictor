# Stage 1: Task Table Materialization Workflow

## Overview

The `materialize_task_tables` function is the first stage in the MCP server's prediction pipeline. Its primary responsibility is to take a client's `QuerySpec` and, using the server's persistent `RDBDataset`, generate `train` and `test` tables. These tables form the foundation for the subsequent feature engineering and model training stages.

This process is analogous to the logic in the `convert_dataset.py` script but is adapted for a dynamic, server-based environment. It transforms a high-level prediction request into concrete datasets ready for machine learning.

## API Signature

As defined in the interface design document:

```python
@dataclass
class TaskTables:
    """Task tables using RDBDataset structure."""
    train_table: pd.DataFrame          # Columns: [__id, __timestamp, __label, ...entity_features]
    test_table: pd.DataFrame           # Columns: [__id, __timestamp, __label=NULL, ...entity_features]

async def materialize_task_tables(
    server: 'TabularPredictionMCPServer'
) -> TaskTables:
    """
    Materializes train and test tables based on the current query and persistent RDB state.

    Args:
        server: MCP server instance (accesses persistent rdb_dataset and current_query_spec)
        
    Returns:
        TaskTables with train/test splits and entity features.
    """
```

## Detailed Workflow Steps

The materialization process can be broken down into the following steps:

### 1. Extract Inputs from Server Context

The function begins by accessing the two key pieces of state from the server instance:
- **`rdb_dataset`**: The persistent, in-memory representation of the entire relational database, loaded at server startup. This avoids all file I/O for data and metadata loading during a query.
- **`current_query_spec`**: The `QuerySpec` object for the current client request, containing all parameters for the desired prediction task.

### 2. Generate Timestamps for Training Data

To create a training set, we need to generate historical data points. This is achieved by creating a series of timestamps leading up to the `ts_current` specified in the query.
- **Strategy**: The exponential backoff algorithm from `convert_dataset.py` is used. It generates timestamps with decreasing frequency as they go further into the past (e.g., daily for the last week, then weekly, then monthly).
- **Bounds**: The generation starts from `query_spec.ts_current` and goes back to the earliest relevant date found in the temporal columns of the `rdb_dataset`.

### 3. Determine Entity Sampling Strategy

The function determines how to sample entities for the training set based on the schema of the `target_table`. This logic is adapted from `_determine_strategies` in `convert_dataset.py`.
- **`creation_based`**: Used if the `target_table` itself has a temporal column. Training samples are created from entities that existed at each historical timestamp.
- **`activity_based`**: Used if the `target_table` has no temporal column but is referenced by other tables that do. This is common for entity tables like `users`. Training samples are selected based on the activity in related tables (e.g., users who made a post or comment).
- **`static`**: Used if the `target_table` and its direct relationships have no temporal aspect. All entities are considered for sampling.

### 4. Materialize the Training Table (`train_table`)

This is the core step where the training data is constructed.
1.  **Select & Sample Entities**: Based on the determined strategy, the function selects a pool of candidate entities at each historical timestamp. It then samples from this pool using the `sampling_rate` from the `QuerySpec`. For the `activity_based` strategy, a bias is applied to favor more active entities.
2.  **Generate Labels via `label_spec`**: The `query_spec.label_spec` contains Python code that defines the prediction target. This code is executed, passing in the server's RDB tables and the generated timestamps. It returns a DataFrame containing the `__id`, `__timestamp`, and the ground-truth `__label` for each training instance.
3.  **Join Entity Features**: The columns from the `target_table` (excluding keys and time columns) are considered static entity features. These features are joined onto the labeled data, creating the complete training table.

The final `train_table` contains the columns `[__id, __timestamp, __label, ...entity_features]`.

### 5. Materialize the Test Table (`test_table`)

The test table represents the prediction task.
1.  **Select Test Entities**: The entities are specified by `query_spec.entity_ids`. If this is `None`, all entities in the `target_table` are used.
2.  **Set Timestamp**: The `__timestamp` for all rows in the test table is set to `query_spec.ts_current`.
3.  **Join Entity Features**: The same static entity features from the `target_table` are joined.
4.  **Add Null Label Column**: A `__label` column is added and filled with `NULL` values, as this is what the model will predict.

The final `test_table` has the same schema as the `train_table`.

### 6. Construct and Return `TaskTables`

The generated `train_table` and `test_table` DataFrames are encapsulated in a `TaskTables` dataclass and returned, completing Stage 1 of the pipeline.

## Simplification from `convert_dataset.py`

The `materialize_task_tables` function, guided by the `QuerySpec`, is a significant simplification and enhancement of the workflow in the original `convert_dataset.py` script.

-   **Dynamic Service vs. Static Script**: The `convert` script is a batch process that converts an entire dataset based on a static configuration. `materialize_task_tables` operates within a persistent server, responding to specific, on-demand queries. This shifts from an offline, one-size-fits-all approach to an online, tailored one.

-   **Configuration via `QuerySpec`**: Parameters that were previously command-line arguments are now encapsulated in the `QuerySpec` object, making each request self-contained and explicit.
    -   `input_folder`, `output_folder`: **Eliminated**. The server works with the in-memory `RDBDataset`.
    -   `today_date`: **Replaced** by the more precise `ts_current` in the query.
    -   `sampling_rate`: **Dynamically configurable** per-query.

-   **Elimination of Redundant Processing**:
    -   **Data Loading**: The `convert` script loads all data from disk on every run. The server loads the `RDBDataset` only once at startup, removing this I/O overhead from the query path.
    -   **Target Iteration**: The `convert` script inefficiently iterates over all possible tables to generate tasks. `materialize_task_tables` is highly efficient as it only ever processes the single `target_table` specified in the `QuerySpec`.

-   **Dynamic and Powerful Labeling**: The most critical improvement is the `label_spec`.
    -   In `convert_dataset.py`, label generation was trivial and hardcoded (e.g., random labels), making it unsuitable for real tasks.
    -   The `label_spec` allows the client to define complex, meaningful prediction targets using Python and SQL (via DuckDB). This decouples the task materialization logic from the task *definition*, making the system vastly more flexible and powerful.

-   **Simplified State Management**: The `convert` script's final step was to modify the `metadata.yaml` file with task definitions. In the new design, tasks are ephemeral and defined entirely by queries. This removes the need to write task metadata back to disk, simplifying the overall architecture.

## Sub-API Pseudocode and Examples

Here is a breakdown of the internal functions that `materialize_task_tables` would call, complete
with clear signatures, pseudocode, and examples.

---

### Main Function: `materialize_task_tables`

This is the main entry point that orchestrates the sub-API calls.

**Pseudocode:**
```python
async def materialize_task_tables(server: 'TabularPredictionMCPServer') -> TaskTables:
    # 1. Get context from the server
    query = server.current_query_spec
    rdb = server.rdb_dataset
    target_table_df = rdb.get_table(query.target_table)
    target_table_schema = rdb.get_table_metadata(query.target_table)

    # 2. Generate historical timestamps for training
    timestamps = _generate_timestamps(
        ts_current=query.ts_current,
        rdb_dataset=rdb
    )

    # 3. Build the training table
    train_table = _materialize_training_table(
        query=query,
        rdb=rdb,
        timestamps=timestamps,
        target_table_df=target_table_df,
        target_table_schema=target_table_schema
    )

    # 4. Build the test table
    test_table = _materialize_test_table(
        query=query,
        target_table_df=target_table_df,
        target_table_schema=target_table_schema
    )

    # 5. Return the final TaskTables object
    return TaskTables(train_table=train_table, test_table=test_table)
```

---

### Sub-API 1: `_generate_timestamps`

This function generates historical timestamps for training data using exponential backoff, adapted from [`convert_dataset.py:_generate_exponential_timestamps()`](convert_dataset.py:98).

**Signature:**
```python
def _generate_timestamps(
    ts_current: datetime,
    rdb_dataset: RDBDataset
) -> List[datetime]:
    """
    Generate historical timestamps using exponential backoff strategy.
    
    Args:
        ts_current: Current timestamp from QuerySpec
        rdb_dataset: Server's persistent RDB dataset
        
    Returns:
        List of timestamps in descending order (most recent first)
    """
```

**Pseudocode:**
```python
def _generate_timestamps(ts_current, rdb_dataset):
    # 1. Find earliest date across all temporal columns in RDB
    earliest_date = None
    for table_schema in rdb_dataset.metadata.tables:
        for col_schema in table_schema.columns:
            if col_schema.dtype == "datetime":
                table_df = rdb_dataset.get_table(table_schema.name)
                if col_schema.name in table_df.columns:
                    min_date = table_df[col_schema.name].min()
                    if earliest_date is None or min_date < earliest_date:
                        earliest_date = min_date
    
    # 2. Generate timestamps with exponential backoff
    timestamps = []
    current_date = ts_current
    day_interval = 1
    count_in_current_interval = 0
    
    while current_date >= earliest_date:
        timestamps.append(current_date)
        current_date -= timedelta(days=day_interval)
        count_in_current_interval += 1
        
        # Every 7 timestamps, double the interval
        if count_in_current_interval == 7:
            day_interval *= 2
            count_in_current_interval = 0
    
    return timestamps
```

**Input Example:**
```python
ts_current = datetime(2021, 1, 1)
rdb_dataset = RDBDataset(path="stackexchange_data")
# RDB contains data from 2008-07-01 to 2021-01-01
```

**Output Example:**
```python
[
    datetime(2021, 1, 1),    # Day 0
    datetime(2020, 12, 31),  # Day 1
    datetime(2020, 12, 30),  # Day 1
    # ... 7 daily timestamps
    datetime(2020, 12, 18),  # Week 1, interval = 2 days
    datetime(2020, 12, 16),  # Week 1, interval = 2 days
    # ... 7 bi-daily timestamps
    datetime(2020, 11, 4),   # Week 2, interval = 4 days
    # ... exponential backoff continues
    datetime(2008, 7, 1)     # Earliest date
]
```

---

### Sub-API 2: `_materialize_training_table`

This function creates the training table by executing the label specification and joining entity features, based on the sampling strategy from [`convert_dataset.py:_process_temporal_table()`](convert_dataset.py:183).

**Signature:**
```python
def _materialize_training_table(
    query: QuerySpec,
    rdb: RDBDataset,
    timestamps: List[datetime],
    target_table_df: pd.DataFrame,
    target_table_schema: RDBTableSchema
) -> pd.DataFrame:
    """
    Materialize training table with labels and entity features.
    
    Args:
        query: Current query specification
        rdb: Server's persistent RDB dataset
        timestamps: Historical timestamps for training
        target_table_df: Target table DataFrame
        target_table_schema: Target table schema metadata
        
    Returns:
        Training DataFrame with columns [__id, __timestamp, __label, ...entity_features]
    """
```

**Pseudocode:**
```python
def _materialize_training_table(query, rdb, timestamps, target_table_df, target_table_schema):
    # 1. Determine sampling strategy
    sampling_strategy = _determine_sampling_strategy(
        target_table_schema=target_table_schema,
        rdb_dataset=rdb
    )
    
    # 2. Execute label specification to get labeled data
    labeled_data = _execute_label_spec(
        label_spec=query.label_spec,
        rdb_tables={name: rdb.get_table(name) for name in rdb.table_names},
        timestamps=pd.Series(timestamps)
    )
    
    # 3. Apply sampling based on strategy and sampling_rate
    if sampling_strategy == "creation_based":
        sampled_data = _apply_creation_based_sampling(
            labeled_data, target_table_df, target_table_schema, query.sampling_rate
        )
    elif sampling_strategy == "activity_based":
        sampled_data = _apply_activity_based_sampling(
            labeled_data, rdb, target_table_schema, query.sampling_rate
        )
    else:  # static
        sampled_data = _apply_static_sampling(
            labeled_data, query.sampling_rate
        )
    
    # 4. Get entity features from target table
    entity_features = _get_entity_features(
        target_table_df=target_table_df,
        target_table_schema=target_table_schema
    )
    
    # 5. Join entity features onto sampled labeled data
    train_table = sampled_data.merge(
        entity_features,
        left_on='__id',
        right_on=query.id_column,
        how='left'
    ).drop(columns=[query.id_column])
    
    return train_table
```

**Input Example:**
```python
query = QuerySpec(
    target_table="users",
    id_column="Id",
    ts_current=datetime(2021, 1, 1),
    label_spec="...",  # Complex user engagement prediction
    sampling_rate=0.3
)
timestamps = [datetime(2021, 1, 1), datetime(2020, 12, 31), ...]
target_table_df = pd.DataFrame({
    'Id': [1, 2, 3, 4],
    'DisplayName': ['Alice', 'Bob', 'Charlie', 'David'],
    'Reputation': [100, 200, 50, 300],
    'CreationDate': [datetime(2019, 1, 1), ...]
})
```

**Output Example:**
```python
pd.DataFrame({
    '__timestamp': [datetime(2021, 1, 1), datetime(2020, 12, 31), ...],
    '__id': [2, 1, 4, 1, ...],
    '__label': [1, 0, 1, 0, ...],  # From label_spec execution
    'DisplayName': ['Bob', 'Alice', 'David', 'Alice', ...],
    'Reputation': [200, 100, 300, 100, ...]
    # CreationDate excluded as it's a time column
})
```

---

### Sub-API 3: `_materialize_test_table`

This function creates the test table for prediction, setting up entities with null labels at the current timestamp.

**Signature:**
```python
def _materialize_test_table(
    query: QuerySpec,
    target_table_df: pd.DataFrame,
    target_table_schema: RDBTableSchema
) -> pd.DataFrame:
    """
    Materialize test table for prediction.
    
    Args:
        query: Current query specification
        target_table_df: Target table DataFrame
        target_table_schema: Target table schema metadata
        
    Returns:
        Test DataFrame with columns [__id, __timestamp, __label=NULL, ...entity_features]
    """
```

**Pseudocode:**
```python
def _materialize_test_table(query, target_table_df, target_table_schema):
    # 1. Select test entities
    if query.entity_ids is not None:
        # Filter to specific entities
        test_entities = target_table_df[
            target_table_df[query.id_column].isin(query.entity_ids)
        ]
    else:
        # Use all entities in target table
        test_entities = target_table_df.copy()
    
    # 2. Get entity features
    entity_features = _get_entity_features(
        target_table_df=test_entities,
        target_table_schema=target_table_schema
    )
    
    # 3. Create test table structure
    test_table = entity_features.copy()
    test_table['__timestamp'] = query.ts_current
    test_table['__id'] = test_table[query.id_column]
    test_table['__label'] = None  # NULL labels for prediction
    test_table = test_table.drop(columns=[query.id_column])
    
    # 4. Reorder columns to match training table format
    cols = ['__id', '__timestamp', '__label'] + [
        col for col in test_table.columns
        if col not in ['__id', '__timestamp', '__label']
    ]
    test_table = test_table[cols]
    
    return test_table
```

**Input Example:**
```python
query = QuerySpec(
    target_table="users",
    entity_ids=[2666],  # Specific user for prediction
    id_column="Id",
    ts_current=datetime(2021, 1, 1)
)
target_table_df = pd.DataFrame({
    'Id': [1, 2666, 3, 4],
    'DisplayName': ['Alice', 'TargetUser', 'Charlie', 'David'],
    'Reputation': [100, 150, 50, 300],
    'CreationDate': [datetime(2019, 1, 1), datetime(2020, 6, 1), ...]
})
```

**Output Example:**
```python
pd.DataFrame({
    '__id': [2666],
    '__timestamp': [datetime(2021, 1, 1)],
    '__label': [None],
    'DisplayName': ['TargetUser'],
    'Reputation': [150]
    # CreationDate excluded as it's a time column
})
```

---

### Sub-API 4: `_execute_label_spec`

This function executes the Python code in the label specification to generate ground-truth labels for training data.

**Signature:**
```python
def _execute_label_spec(
    label_spec: str,
    rdb_tables: Dict[str, pd.DataFrame],
    timestamps: pd.Series[pd.Timestamp]
) -> pd.DataFrame:
    """
    Execute label specification code to generate training labels.
    
    Args:
        label_spec: Python code string defining label generation logic
        rdb_tables: Dictionary of all RDB tables
        timestamps: Series of timestamps for label generation
        
    Returns:
        DataFrame with columns [__timestamp, __id, __label]
    """
```

**Pseudocode:**
```python
def _execute_label_spec(label_spec, rdb_tables, timestamps):
    # 1. Set up execution environment
    import duckdb
    
    # Create namespace for label_spec execution
    exec_namespace = {
        'tables': rdb_tables,
        'timestamps': timestamps,
        'pd': pd,
        'duckdb': duckdb,
        'datetime': datetime,
        'timedelta': timedelta
    }
    
    # 2. Execute the label specification code
    exec(label_spec, exec_namespace)
    
    # 3. Call the create_table function defined in label_spec
    labeled_data = exec_namespace['create_table'](
        tables=rdb_tables,
        timestamps=timestamps
    )
    
    # 4. Validate output format
    required_columns = ['__timestamp', '__id', '__label']
    if not all(col in labeled_data.columns for col in required_columns):
        raise ValueError(f"Label spec must return DataFrame with columns {required_columns}")
    
    return labeled_data[required_columns]
```

**Input Example:**
```python
label_spec = '''
def create_table(tables, timestamps):
    users = tables['users']
    comments = tables['comments']
    timestamp_df = pd.DataFrame({'timestamp': timestamps})
    timedelta = pd.Timedelta(days=90)
    
    return duckdb.sql("""
        SELECT
            t.timestamp AS __timestamp,
            u.Id AS __id,
            CASE WHEN COUNT(c.Id) > 0 THEN 1 ELSE 0 END AS __label
        FROM timestamp_df t
        CROSS JOIN users u
        LEFT JOIN comments c ON u.Id = c.UserId
            AND c.CreationDate > t.timestamp
            AND c.CreationDate <= t.timestamp + INTERVAL '90 days'
        WHERE u.CreationDate <= t.timestamp
        GROUP BY t.timestamp, u.Id
    """).df()
'''

rdb_tables = {
    'users': pd.DataFrame({'Id': [1, 2], 'CreationDate': [...]}),
    'comments': pd.DataFrame({'UserId': [1, 1, 2], 'CreationDate': [...]})
}
timestamps = pd.Series([datetime(2021, 1, 1), datetime(2020, 12, 1)])
```

**Output Example:**
```python
pd.DataFrame({
    '__timestamp': [datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2020, 12, 1), ...],
    '__id': [1, 2, 1, ...],
    '__label': [1, 0, 1, ...]  # 1 = user will comment within 90 days, 0 = won't
})
```

---

### Sub-API 5: `_determine_sampling_strategy`

This function determines the appropriate sampling strategy based on the target table's schema and relationships, adapted from [`convert_dataset.py:_determine_strategies()`](convert_dataset.py:140).

**Signature:**
```python
def _determine_sampling_strategy(
    target_table_schema: RDBTableSchema,
    rdb_dataset: RDBDataset
) -> Literal["creation_based", "activity_based", "static"]:
    """
    Determine sampling strategy based on table schema and relationships.
    
    Args:
        target_table_schema: Schema of the target table
        rdb_dataset: Server's persistent RDB dataset
        
    Returns:
        Sampling strategy: "creation_based", "activity_based", or "static"
    """
```

**Pseudocode:**
```python
def _determine_sampling_strategy(target_table_schema, rdb_dataset):
    target_table_name = target_table_schema.name
    
    # 1. Check if target table has temporal column
    has_temporal_column = any(
        col.dtype == "datetime"
        for col in target_table_schema.columns
    )
    
    if has_temporal_column:
        return "creation_based"
    
    # 2. Check if target table is referenced by temporal tables
    relationships = rdb_dataset.get_relationships()
    
    for child_table, child_col, parent_table, parent_col in relationships:
        if parent_table == target_table_name:
            # Check if the child table has temporal columns
            child_table_schema = rdb_dataset.get_table_metadata(child_table)
            child_has_temporal = any(
                col.dtype == "datetime"
                for col in child_table_schema.columns
            )
            if child_has_temporal:
                return "activity_based"
    
    return "static"
```

**Input Example:**
```python
target_table_schema = RDBTableSchema(
    name="users",
    columns=[
        RDBColumnSchema(name="Id", dtype="primary_key"),
        RDBColumnSchema(name="DisplayName", dtype="string"),
        RDBColumnSchema(name="Reputation", dtype="int"),
        # No datetime column
    ]
)

rdb_dataset = RDBDataset(...)  # Contains posts, comments tables that reference users
```

**Output Example:**
```python
"activity_based"  # Users table has no temporal column, but posts/comments reference it and have CreationDate
```

---

### Sub-API 6: `_get_entity_features`

This function extracts static entity features from the target table, excluding keys and temporal columns.

**Signature:**
```python
def _get_entity_features(
    target_table_df: pd.DataFrame,
    target_table_schema: RDBTableSchema
) -> pd.DataFrame:
    """
    Extract entity features from target table.
    
    Args:
        target_table_df: Target table DataFrame
        target_table_schema: Target table schema metadata
        
    Returns:
        DataFrame with entity features (excludes keys and time columns)
    """
```

**Pseudocode:**
```python
def _get_entity_features(target_table_df, target_table_schema):
    # 1. Identify columns to include as features
    feature_columns = []
    exclude_dtypes = {"primary_key", "foreign_key", "datetime"}
    
    for col_schema in target_table_schema.columns:
        if col_schema.dtype not in exclude_dtypes:
            feature_columns.append(col_schema.name)
    
    # 2. Include primary key for joining (will be renamed to __id later)
    primary_key_col = None
    for col_schema in target_table_schema.columns:
        if col_schema.dtype == "primary_key":
            primary_key_col = col_schema.name
            break
    
    if primary_key_col:
        include_columns = [primary_key_col] + feature_columns
    else:
        include_columns = feature_columns
    
    # 3. Select and return feature columns
    available_columns = [col for col in include_columns if col in target_table_df.columns]
    return target_table_df[available_columns].copy()
```

**Input Example:**
```python
target_table_df = pd.DataFrame({
    'Id': [1, 2, 3],                              # primary_key - include for joining
    'DisplayName': ['Alice', 'Bob', 'Charlie'],   # string - include as feature
    'Reputation': [100, 200, 50],                 # int - include as feature
    'CreationDate': [datetime(2019, 1, 1), ...], # datetime - exclude
    'LocationId': [10, 20, 30]                    # foreign_key - exclude
})

target_table_schema = RDBTableSchema(
    name="users",
    columns=[
        RDBColumnSchema(name="Id", dtype="primary_key"),
        RDBColumnSchema(name="DisplayName", dtype="string"),
        RDBColumnSchema(name="Reputation", dtype="int"),
        RDBColumnSchema(name="CreationDate", dtype="datetime"),
        RDBColumnSchema(name="LocationId", dtype="foreign_key")
    ]
)
```

**Output Example:**
```python
pd.DataFrame({
    'Id': [1, 2, 3],                              # Primary key for joining
    'DisplayName': ['Alice', 'Bob', 'Charlie'],   # Feature
    'Reputation': [100, 200, 50]                  # Feature
    # CreationDate and LocationId excluded
})
```

---

## Complete Workflow Example

Here's a comprehensive example showing how all sub-APIs work together:

### Input: User Engagement Prediction Query

```python
# Server state
server.rdb_dataset = RDBDataset("stackexchange_data")  # Loaded at startup
server.current_query_spec = QuerySpec(
    target_table="users",
    entity_ids=[2666],
    id_column="Id",
    ts_current=datetime(2021, 1, 1),
    task_type="classification",
    label_spec="""
def create_table(tables, timestamps):
    users = tables['users']
    comments = tables['comments']
    posts = tables['posts']
    votes = tables['votes']
    timestamp_df = pd.DataFrame({'timestamp': timestamps})
    
    # Define engagement as any comment, post, or vote
    return duckdb.sql('''
        WITH ALL_ACTIVITY AS (
            SELECT UserId, CreationDate FROM comments
            UNION ALL
            SELECT OwnerUserId as UserId, CreationDate FROM posts
            UNION ALL
            SELECT UserId, CreationDate FROM votes
        )
        SELECT
            t.timestamp AS __timestamp,
            u.Id AS __id,
            CASE WHEN COUNT(a.UserId) > 0 THEN 1 ELSE 0 END AS __label
        FROM timestamp_df t
        CROSS JOIN users u
        LEFT JOIN ALL_ACTIVITY a ON u.Id = a.UserId
            AND a.CreationDate > t.timestamp
            AND a.CreationDate <= t.timestamp + INTERVAL '90 days'
        WHERE u.CreationDate <= t.timestamp
        GROUP BY t.timestamp, u.Id
    ''').df()
""",
    sampling_rate=0.3,
    horizon_days=90
)
```

### Step-by-Step Execution:

**Step 1: Generate Timestamps**
```python
timestamps = _generate_timestamps(
    ts_current=datetime(2021, 1, 1),
    rdb_dataset=server.rdb_dataset
)
# Result: [datetime(2021, 1, 1), datetime(2020, 12, 31), ..., datetime(2008, 7, 1)]
```

**Step 2: Execute Label Specification**
```python
labeled_data = _execute_label_spec(
    label_spec=server.current_query_spec.label_spec,
    rdb_tables={name: server.rdb_dataset.get_table(name) for name in server.rdb_dataset.table_names},
    timestamps=pd.Series(timestamps)
)
# Result: DataFrame with ~50,000 rows of (timestamp, user_id, engagement_label)
```

**Step 3: Determine Sampling Strategy**
```python
strategy = _determine_sampling_strategy(
    target_table_schema=server.rdb_dataset.get_table_metadata("users"),
    rdb_dataset=server.rdb_dataset
)
# Result: "activity_based" (users table referenced by temporal tables)
```

**Step 4: Apply Sampling and Get Entity Features**
```python
# Apply activity-based sampling (30% of users, biased toward active users)
sampled_data = labeled_data.sample(frac=0.3, weights=activity_scores)

entity_features = _get_entity_features(
    target_table_df=server.rdb_dataset.get_table("users"),
    target_table_schema=server.rdb_dataset.get_table_metadata("users")
)
# Result: DataFrame with user features (DisplayName, Reputation, etc.)
```

**Step 5: Create Training Table**
```python
train_table = sampled_data.merge(entity_features, left_on='__id', right_on='Id')
# Result: ~15,000 training examples with timestamps, labels, and user features
```

**Step 6: Create Test Table**
```python
test_table = _materialize_test_table(
    query=server.current_query_spec,
    target_table_df=server.rdb_dataset.get_table("users"),
    target_table_schema=server.rdb_dataset.get_table_metadata("users")
)
# Result: 1 test example for user 2666 at timestamp 2021-01-01
```

### Final Output: TaskTables

```python
TaskTables(
    train_table=pd.DataFrame({
        '__id': [1, 5, 8, 12, ...],                    # ~15,000 training examples
        '__timestamp': [datetime(2021, 1, 1), ...],    # Various historical timestamps
        '__label': [1, 0, 1, 0, ...],                  # Engagement labels
        'DisplayName': ['Alice', 'Bob', ...],          # User features
        'Reputation': [100, 200, 50, ...]              # User features
    }),
    
    test_table=pd.DataFrame({
        '__id': [2666],                                 # Target user
        '__timestamp': [datetime(2021, 1, 1)],         # Current timestamp
        '__label': [None],                             # To be predicted
        'DisplayName': ['TargetUser'],                 # User features
        'Reputation': [150]                            # User features
    })
)
```

## QuerySpec Simplifications Summary

The `QuerySpec` design dramatically simplifies the materialization process compared to [`convert_dataset.py`](convert_dataset.py):

### 1. **Eliminates File I/O Completely**
- **Before**: `convert_dataset.py` loads metadata and parquet files from disk on every run
- **After**: `materialize_task_tables` uses server's persistent `RDBDataset`, eliminating all I/O from the query path

### 2. **Single-Table Focus vs. Batch Processing**
- **Before**: Script iterates over all tables to generate multiple tasks
- **After**: Function processes only the specified `target_table`, making it much faster and more focused

### 3. **Dynamic Label Generation vs. Static Random Labels**
- **Before**: Labels are randomly generated: `training_df['__label__'] = np.random.choice([0, 1], size=len(training_df))`
- **After**: Labels are generated via powerful `label_spec` Python/SQL code, enabling real prediction tasks

### 4. **Flexible Entity Selection vs. Bulk Processing**
- **Before**: Script processes all entities in each table
- **After**: `entity_ids` parameter allows targeting specific entities for prediction

### 5. **Query-Specific Configuration vs. Command-Line Arguments**
- **Before**: Configuration via CLI args: `--sampling_rate`, `--today_date`, etc.
- **After**: All parameters encapsulated in `QuerySpec`, enabling per-query customization

### 6. **In-Memory Processing vs. File Output**
- **Before**: Script writes task files to disk in complex directory structures
- **After**: Function returns `TaskTables` directly in memory for immediate use by Stage 2

### 7. **Temporal Precision vs. Date Strings**
- **Before**: Uses string `today_date` parameter
- **After**: Uses precise `datetime` objects with `ts_current` for exact temporal queries

This design transformation enables the system to shift from a slow, batch-oriented data preparation script to a fast, interactive prediction service capable of handling real-time queries with complex, meaningful prediction targets.