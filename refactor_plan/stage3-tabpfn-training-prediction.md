# Stage 3: TabPFN Training & Prediction Workflow

## Overview

The `train_and_predict` function is the third and final stage in the MCP server's prediction pipeline. Its primary responsibility is to take the `FeatureArtifacts` produced by Stage 2 and, using the server's persistent TabPFN manager, train a model on the training data and generate predictions for the test data. This stage represents the culmination of the pipeline, where the prepared data is transformed into actionable predictions.

TabPFN (Tabular Prior-data Fitted Networks) is a powerful foundation model for tabular data that excels at few-shot learning. It can achieve high accuracy with limited training data and requires minimal hyperparameter tuning, making it ideal for the dynamic, query-based prediction tasks in the MCP server architecture.

## API Signature

As defined in the interface design document:

```python
@dataclass
class PredictionArtifact:
    """Prediction artifact - clean data only."""
    test_with_predictions: pd.DataFrame  # Columns: [__id, __timestamp, y_pred, y_prob (if classification)]

async def train_and_predict(
    feature_artifacts: FeatureArtifacts,
    server: 'TabularPredictionMCPServer'
) -> PredictionArtifact:
    """
    Train TabPFN model and generate predictions using server configuration and current query.
    
    Args:
        feature_artifacts: DFS features from Stage 2
        server: MCP server instance (accesses tabpfn_config and current_query_spec)
        
    Returns:
        PredictionArtifact with predictions
    """
```

## Detailed Workflow Steps

The training and prediction process can be broken down into the following steps:

### 1. Extract Inputs from Server Context

The function begins by accessing the key pieces of state from the server instance:
- **`current_query_spec`**: The `QuerySpec` object for the current client request, containing parameters like `task_type` that control model training.
- **`tabpfn_config`**: The server's TabPFN configuration, which may include model type, ensemble settings, and other parameters.

### 2. Prepare Training Data

The function prepares the training data from the feature artifacts:
- **Extract Features and Labels**: Separate the features (X) from the target labels (y) in the training data.
- **Handle Missing Values**: Apply strategies for handling missing values in the feature set.
- **Feature Scaling/Normalization**: Apply appropriate preprocessing to ensure features are in the optimal range for TabPFN.
- **Feature Selection**: Optionally select the most relevant features if the feature set is very large.

### 3. Train TabPFN Model

This is the core step where the TabPFN model is trained:
- **Configure TabPFN**: Set up the TabPFN model with the appropriate parameters based on the task type (classification or regression).
- **Fit Model**: Train the TabPFN model on the prepared training data.
- **Cross-Validation**: Optionally perform cross-validation to assess model performance.

### 4. Generate Predictions for Test Data

Once the model is trained, it is used to generate predictions:
- **Prepare Test Features**: Ensure test features match the format used during training.
- **Generate Predictions**: Apply the trained model to the test features to generate predictions.
- **Format Prediction Output**: Structure the predictions in the required format, including probabilities for classification tasks.

### 5. Generate Model Insights (Optional)

For explainability and debugging:
- **Feature Importance**: Calculate and include feature importance scores.
- **Model Performance Metrics**: Include metrics like accuracy, F1 score, or RMSE depending on the task type.
- **Prediction Confidence**: Include confidence scores for predictions where applicable.

### 6. Construct and Return `PredictionArtifact`

The generated predictions are encapsulated in a `PredictionArtifact` dataclass and returned, completing Stage 3 and the entire pipeline.

## Sub-API Pseudocode and Examples

Here is a breakdown of the internal functions that `train_and_predict` would call, complete with clear signatures, pseudocode, and examples.

---

### Main Function: `train_and_predict`

This is the main entry point that orchestrates the sub-API calls.

**Pseudocode:**
```python
async def train_and_predict(
    feature_artifacts: FeatureArtifacts,
    server: 'TabularPredictionMCPServer'
) -> PredictionArtifact:
    # 1. Get context from the server
    query = server.current_query_spec
    tabpfn_config = _merge_tabpfn_configs(server.tabpfn_config, query)
    
    # 2. Prepare training data
    train_X, train_y = _prepare_training_data(
        feature_artifacts.train_features_table,
        task_tables=server.task_tables,
        task_type=query.task_type
    )
    
    # 3. Train TabPFN model
    model = _train_tabpfn_model(
        train_X=train_X,
        train_y=train_y,
        task_type=query.task_type,
        tabpfn_config=tabpfn_config
    )
    
    # 4. Prepare test data
    test_X = _prepare_test_data(
        feature_artifacts.test_features_table,
        train_X_columns=train_X.columns
    )
    
    # 5. Generate predictions
    predictions = _generate_predictions(
        model=model,
        test_X=test_X,
        task_type=query.task_type
    )
    
    # 6. Format prediction results
    test_with_predictions = _format_prediction_results(
        test_df=feature_artifacts.test_features_table,
        predictions=predictions,
        task_type=query.task_type
    )
    
    # 7. Return the final PredictionArtifact object
    return PredictionArtifact(
        test_with_predictions=test_with_predictions
    )
```

---

### Sub-API 1: `_merge_tabpfn_configs`

This function merges the server's default TabPFN configuration with any query-specific overrides.

**Signature:**
```python
def _merge_tabpfn_configs(
    server_config: Dict[str, Any],
    query: QuerySpec
) -> Dict[str, Any]:
    """
    Merge server TabPFN config with query-specific overrides.
    
    Args:
        server_config: Server's default TabPFN configuration
        query: Current query specification with potential overrides
        
    Returns:
        Merged configuration dictionary
    """
```

**Pseudocode:**
```python
def _merge_tabpfn_configs(server_config, query):
    # Start with server config
    merged_config = server_config.copy()
    
    # Override with query-specific parameters if provided
    if hasattr(query, 'tabpfn_ensemble_size'):
        merged_config['ensemble_size'] = query.tabpfn_ensemble_size
    
    if hasattr(query, 'tabpfn_max_samples'):
        merged_config['max_samples'] = query.tabpfn_max_samples
    
    # Add other query-specific overrides as needed
    
    return merged_config
```

**Input Example:**
```python
server_config = {
    'model_type': 'tabpfn_v2',
    'ensemble_size': 16,
    'max_samples': 10000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

query = QuerySpec(
    target_table="users",
    id_column="Id",
    task_type="classification",
    tabpfn_ensemble_size=32  # Override default ensemble size
)
```

**Output Example:**
```python
{
    'model_type': 'tabpfn_v2',
    'ensemble_size': 32,  # Overridden from query
    'max_samples': 10000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

---

### Sub-API 2: `_prepare_training_data`

This function prepares the training data for the TabPFN model, extracting features and labels.

**Signature:**
```python
def _prepare_training_data(
    train_features_table: pd.DataFrame,
    task_tables: TaskTables,
    task_type: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare training data for TabPFN model.
    
    Args:
        train_features_table: Feature table from Stage 2
        task_tables: Task tables from Stage 1 (for labels)
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        Tuple of (X_train, y_train) for model training
    """
```

**Pseudocode:**
```python
def _prepare_training_data(train_features_table, task_tables, task_type):
    # 1. Extract entity IDs and features
    entity_ids = train_features_table['__id'].values
    
    # 2. Remove non-feature columns
    feature_columns = [col for col in train_features_table.columns 
                      if col not in ['__id', 'cutoff_time']]
    X_train = train_features_table[feature_columns].copy()
    
    # 3. Get labels from task_tables
    train_labels = task_tables.train_table[['__id', '__label']].copy()
    
    # 4. Merge labels with entity IDs
    merged_data = pd.merge(
        pd.DataFrame({'__id': entity_ids}),
        train_labels,
        on='__id',
        how='left'
    )
    
    y_train = merged_data['__label'].values
    
    # 5. Handle missing values in features
    X_train = _handle_missing_values(X_train)
    
    # 6. Apply feature scaling if needed
    X_train = _scale_features(X_train)
    
    # 7. Convert labels to appropriate type based on task
    if task_type == 'classification':
        y_train = y_train.astype(int)
    else:  # regression
        y_train = y_train.astype(float)
    
    return X_train, y_train
```

**Input Example:**
```python
train_features_table = pd.DataFrame({
    '__id': [1, 5, 8, 12],
    'cutoff_time': [datetime(2021, 1, 1), datetime(2020, 12, 1), datetime(2020, 11, 1), datetime(2020, 10, 1)],
    'users.Reputation': [100, 200, 50, 300],
    'COUNT(posts)': [5, 2, 10, 3],
    'MEAN(posts.Score)': [3.2, 4.5, 2.1, 3.8]
})

task_tables = TaskTables(
    train_table=pd.DataFrame({
        '__id': [1, 5, 8, 12],
        '__timestamp': [datetime(2021, 1, 1), datetime(2020, 12, 1), datetime(2020, 11, 1), datetime(2020, 10, 1)],
        '__label': [1, 0, 1, 0]
    }),
    test_table=pd.DataFrame({...})
)

task_type = "classification"
```

**Output Example:**
```python
# X_train
pd.DataFrame({
    'users.Reputation': [100, 200, 50, 300],
    'COUNT(posts)': [5, 2, 10, 3],
    'MEAN(posts.Score)': [3.2, 4.5, 2.1, 3.8]
})

# y_train
np.array([1, 0, 1, 0])
```

---

### Sub-API 3: `_handle_missing_values`

This function handles missing values in the feature set.

**Signature:**
```python
def _handle_missing_values(
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Handle missing values in feature DataFrame.
    
    Args:
        features_df: DataFrame with features
        
    Returns:
        DataFrame with handled missing values
    """
```

**Pseudocode:**
```python
def _handle_missing_values(features_df):
    # Make a copy to avoid modifying the original
    df = features_df.copy()
    
    # For each column, handle missing values appropriately
    for col in df.columns:
        # If numeric column, fill with median
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        # If categorical, fill with most frequent value
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "MISSING")
        # For other types, fill with a placeholder
        else:
            df[col] = df[col].fillna("MISSING")
    
    return df
```

**Input Example:**
```python
features_df = pd.DataFrame({
    'users.Reputation': [100, 200, None, 300],
    'COUNT(posts)': [5, 2, 10, None],
    'MEAN(posts.Score)': [3.2, None, 2.1, 3.8]
})
```

**Output Example:**
```python
pd.DataFrame({
    'users.Reputation': [100, 200, 200, 300],  # Filled with median (200)
    'COUNT(posts)': [5, 2, 10, 5],             # Filled with median (5)
    'MEAN(posts.Score)': [3.2, 3.05, 2.1, 3.8] # Filled with median (3.05)
})
```

---

### Sub-API 4: `_scale_features`

This function applies feature scaling to ensure optimal performance with TabPFN.

**Signature:**
```python
def _scale_features(
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply feature scaling to ensure optimal performance with TabPFN.
    
    Args:
        features_df: DataFrame with features
        
    Returns:
        DataFrame with scaled features
    """
```

**Pseudocode:**
```python
def _scale_features(features_df):
    # Make a copy to avoid modifying the original
    df = features_df.copy()
    
    # For each numeric column, apply scaling
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # TabPFN works well with features in a reasonable range
            # Apply robust scaling to handle outliers
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Avoid division by zero
            if iqr > 0:
                df[col] = (df[col] - q1) / iqr
            else:
                # If IQR is zero, use standard scaling
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
    
    return df
```

**Input Example:**
```python
features_df = pd.DataFrame({
    'users.Reputation': [100, 200, 200, 300],
    'COUNT(posts)': [5, 2, 10, 5],
    'MEAN(posts.Score)': [3.2, 3.05, 2.1, 3.8]
})
```

**Output Example:**
```python
pd.DataFrame({
    'users.Reputation': [0.0, 0.5, 0.5, 1.0],  # Scaled based on IQR
    'COUNT(posts)': [0.375, 0.0, 1.0, 0.375],  # Scaled based on IQR
    'MEAN(posts.Score)': [0.647, 0.559, 0.0, 1.0]  # Scaled based on IQR
})
```

---

### Sub-API 5: `_train_tabpfn_model`

This function trains the TabPFN model on the prepared training data.

**Signature:**
```python
def _train_tabpfn_model(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    task_type: str,
    tabpfn_config: Dict[str, Any]
) -> Any:
    """
    Train TabPFN model on prepared training data.
    
    Args:
        train_X: Training features
        train_y: Training labels
        task_type: Type of task ('classification' or 'regression')
        tabpfn_config: Configuration for TabPFN
        
    Returns:
        Trained TabPFN model
    """
```

**Pseudocode:**
```python
def _train_tabpfn_model(train_X, train_y, task_type, tabpfn_config):
    # 1. Configure TabPFN based on task type
    if task_type == 'classification':
        model = TabPFNClassifier(
            device=tabpfn_config.get('device', 'cpu'),
            N_ensemble_configurations=tabpfn_config.get('ensemble_size', 16)
        )
    else:  # regression
        model = TabPFNRegressor(
            device=tabpfn_config.get('device', 'cpu'),
            N_ensemble_configurations=tabpfn_config.get('ensemble_size', 16)
        )
    
    # 2. Handle sample size limits
    max_samples = tabpfn_config.get('max_samples', 10000)
    if len(train_X) > max_samples:
        # Sample data if it exceeds TabPFN's limits
        indices = np.random.choice(len(train_X), max_samples, replace=False)
        X_subset = train_X.iloc[indices]
        y_subset = train_y[indices]
    else:
        X_subset = train_X
        y_subset = train_y
    
    # 3. Fit the model
    model.fit(X_subset, y_subset)
    
    # 4. Optionally evaluate on a validation set
    if tabpfn_config.get('evaluate_performance', False):
        # Simple validation using a holdout set
        X_train, X_val, y_train, y_val = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        
        if task_type == 'classification':
            val_score = model.score(X_val, y_val)  # Accuracy
            model.validation_score = val_score
        else:  # regression
            val_pred = model.predict(X_val)
            val_score = mean_squared_error(y_val, val_pred)
            model.validation_score = val_score
        
        # Refit on full data
        model.fit(X_subset, y_subset)
    
    return model
```

**Input Example:**
```python
train_X = pd.DataFrame({
    'users.Reputation': [0.0, 0.5, 0.5, 1.0],
    'COUNT(posts)': [0.375, 0.0, 1.0, 0.375],
    'MEAN(posts.Score)': [0.647, 0.559, 0.0, 1.0]
})

train_y = np.array([1, 0, 1, 0])

task_type = "classification"

tabpfn_config = {
    'model_type': 'tabpfn_v2',
    'ensemble_size': 32,
    'max_samples': 10000,
    'device': 'cpu',
    'evaluate_performance': True
}
```

**Output Example:**
```python
# A trained TabPFNClassifier model
TabPFNClassifier(
    device='cpu',
    N_ensemble_configurations=32,
    validation_score=0.85  # Added during evaluation
)
```

---

### Sub-API 6: `_prepare_test_data`

This function prepares the test data for prediction, ensuring it matches the format of the training data.

**Signature:**
```python
def _prepare_test_data(
    test_features_table: pd.DataFrame,
    train_X_columns: pd.Index
) -> pd.DataFrame:
    """
    Prepare test data for prediction.
    
    Args:
        test_features_table: Test feature table from Stage 2
        train_X_columns: Columns from the training features
        
    Returns:
        Prepared test features DataFrame
    """
```

**Pseudocode:**
```python
def _prepare_test_data(test_features_table, train_X_columns):
    # 1. Extract entity IDs and features
    entity_ids = test_features_table['__id'].values
    
    # 2. Select only the columns that were used in training
    feature_columns = [col for col in train_X_columns if col in test_features_table.columns]
    
    # 3. Create a DataFrame with the same columns as train_X
    test_X = pd.DataFrame(index=range(len(entity_ids)), columns=train_X_columns)
    
    # 4. Fill in the values for columns that exist in test_features_table
    for col in feature_columns:
        test_X[col] = test_features_table[col].values
    
    # 5. Handle missing columns (that were in train but not in test)
    for col in train_X_columns:
        if col not in feature_columns:
            # Fill with median from training data (would need to be passed in)
            test_X[col] = 0  # Placeholder, ideally use training statistics
    
    # 6. Handle missing values
    test_X = _handle_missing_values(test_X)
    
    # 7. Apply the same scaling as was applied to training data
    test_X = _scale_features(test_X)
    
    return test_X
```

**Input Example:**
```python
test_features_table = pd.DataFrame({
    '__id': [2666],
    'cutoff_time': [datetime(2021, 1, 1)],
    'users.Reputation': [150],
    'COUNT(posts)': [7],
    'MEAN(posts.Score)': [4.1]
})

train_X_columns = pd.Index(['users.Reputation', 'COUNT(posts)', 'MEAN(posts.Score)'])
```

**Output Example:**
```python
pd.DataFrame({
    'users.Reputation': [0.25],  # Scaled based on training data statistics
    'COUNT(posts)': [0.625],     # Scaled based on training data statistics
    'MEAN(posts.Score)': [0.882] # Scaled based on training data statistics
})
```

---

### Sub-API 7: `_generate_predictions`

This function generates predictions using the trained model on the test data.

**Signature:**
```python
def _generate_predictions(
    model: Any,
    test_X: pd.DataFrame,
    task_type: str
) -> Dict[str, np.ndarray]:
    """
    Generate predictions using the trained model.
    
    Args:
        model: Trained TabPFN model
        test_X: Prepared test features
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        Dictionary with predictions and probabilities (for classification)
    """
```

**Pseudocode:**
```python
def _generate_predictions(model, test_X, task_type):
    # Generate predictions
    if task_type == 'classification':
        # For classification, get both class predictions and probabilities
        y_pred = model.predict(test_X)
        y_prob = model.predict_proba(test_X)
        
        # For binary classification, extract probability of positive class
        if y_prob.shape[1] == 2:
            y_prob_positive = y_prob[:, 1]
        else:
            # For multiclass, keep all probabilities
            y_prob_positive = y_prob
        
        return {
            'y_pred': y_pred,
            'y_prob': y_prob_positive
        }
    else:  # regression
        # For regression, just get predictions
        y_pred = model.predict(test_X)
        return {
            'y_pred': y_pred
        }
```

**Input Example:**
```python
model = TabPFNClassifier(...)  # Trained model from _train_tabpfn_model

test_X = pd.DataFrame({
    'users.Reputation': [0.25],
    'COUNT(posts)': [0.625],
    'MEAN(posts.Score)': [0.882]
})

task_type = "classification"
```

**Output Example:**
```python
{
    'y_pred': np.array([1]),  # Predicted class
    'y_prob': np.array([0.87])  # Probability of positive class
}
```

---

### Sub-API 8: `_format_prediction_results`

This function formats the prediction results into the required output structure.

**Signature:**
```python
def _format_prediction_results(
    test_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    task_type: str
) -> pd.DataFrame:
    """
    Format prediction results into the required output structure.
    
    Args:
        test_df: Original test DataFrame
        predictions: Dictionary with predictions from model
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        DataFrame with predictions in the required format
    """
```

**Pseudocode:**
```python
def _format_prediction_results(test_df, predictions, task_type):
    # 1. Extract entity IDs and timestamps
    result_df = test_df[['__id', 'cutoff_time']].copy()
    result_df.rename(columns={'cutoff_time': '__timestamp'}, inplace=True)
    
    # 2. Add predictions
    result_df['y_pred'] = predictions['y_pred']
    
    # 3. Add probabilities for classification tasks
    if task_type == 'classification' and 'y_prob' in predictions:
        if isinstance(predictions['y_prob'], np.ndarray) and predictions['y_prob'].ndim == 2:
            # Multi-class case
            for i in range(predictions['y_prob'].shape[1]):
                result_df[f'y_prob_{i}'] = predictions['y_prob'][:, i]
        else:
            # Binary classification case
            result_df['y_prob'] = predictions['y_prob']
    
    return result_df
```

**Input Example:**
```python
test_df = pd.DataFrame({
    '__id': [2666],
    'cutoff_time': [datetime(2021, 1, 1)],
    'users.Reputation': [150],
    'COUNT(posts)': [7],
    'MEAN(posts.Score)': [4.1]
})

predictions = {
    'y_pred': np.array([1]),
    'y_prob': np.array([0.87])
}

task_type = "classification"
```

**Output Example:**
```python
pd.DataFrame({
    '__id': [2666],
    '__timestamp': [datetime(2021, 1, 1)],
    'y_pred': [1],
    'y_prob': [0.87]
})
```

---

## Complete Workflow Example

Here's a comprehensive example showing how all sub-APIs work together:

### Input: User Engagement Prediction Query

```python
# Server state
server.rdb_dataset = RDBDataset("stackexchange_data")  # Loaded at startup
server.tabpfn_config = {
    'model_type': 'tabpfn_v2',
    'ensemble_size': 16,
    'max_samples': 10000,
    'device': 'cpu'
}
server.current_query_spec = QuerySpec(
    target_table="users",
    entity_ids=[2666],
    id_column="Id",
    ts_current=datetime(2021, 1, 1),
    task_type="classification",
    query_id="user_engagement_classification_2666"
)

# Input from Stage 2
feature_artifacts = FeatureArtifacts(
    train_features_table=pd.DataFrame({
        '__id': [1, 5, 8, 12, ...],                    # ~15,000 training examples
        'cutoff_time': [datetime(2021, 1, 1), ...],    # Various historical timestamps
        'users.Reputation': [100, 200, 50, ...],       # Original features
        # Generated features (100+ columns)
        'COUNT(posts WHERE posts.OwnerUserId = users.Id)': [5, 2, 10, ...],
        'MEAN(posts.Score WHERE posts.OwnerUserId = users.Id)': [3.2, 4.5, 2.1, ...],
        'COUNT(comments WHERE comments.UserId = users.Id)': [12, 5, 8, ...],
        # ... many more features ...
    }),
    
    test_features_table=pd.DataFrame({
        '__id': [2666],                                # Target user
        'cutoff_time': [datetime(2021, 1, 1)],         # Current timestamp
        'users.Reputation': [150],                     # Original features
        # Same generated features as train set
        'COUNT(posts WHERE posts.OwnerUserId = users.Id)': [7],
        'MEAN(posts.Score WHERE posts.OwnerUserId = users.Id)': [4.1],
        'COUNT(comments WHERE comments.UserId = users.Id)': [9],
        # ... many more features ...
    })
)

# Task tables from Stage 1 (needed for labels)
server.task_tables = TaskTables(
    train_table=pd.DataFrame({
        '__id': [1, 5, 8, 12, ...],
        '__timestamp': [datetime(2021, 1, 1), ...],
        '__label': [1, 0, 1, 0, ...],  # Engagement labels
        # ... entity features ...
    }),
    test_table=pd.DataFrame({
        '__id': [2666],
        '__timestamp': [datetime(2021, 1, 1)],
        '__label': [None],  # To be predicted
        # ... entity features ...
    })
)
```

### Step-by-Step Execution:

**Step 1: Merge TabPFN Configurations**
```python
tabpfn_config = _merge_tabpfn_configs(
    server_config=server.tabpfn_config,
    query=server.current_query_spec
)
# Result: {'model_type': 'tabpfn_v2', 'ensemble_size': 16, 'max_samples': 10000, 'device': 'cpu'}
```

**Step 2: Prepare Training Data**
```python
train_X, train_y = _prepare_training_data(
    train_features_table=feature_artifacts.train_features_table,
    task_tables=server.task_tables,
    task_type=server.current_query_spec.task_type
)
# Result: 
# train_X: DataFrame with 100+ feature columns, 15,000 rows (scaled and cleaned)
# train_y: Array with 15,000 binary labels [1, 0, 1, 0, ...]
```

**Step 3: Train TabPFN Model**
```python
model = _train_tabpfn_model(
    train_X=train_X,
    train_y=train_y,
    task_type=server.current_query_spec.task_type,
    tabpfn_config=tabpfn_config
)
# Result: Trained TabPFNClassifier with validation_score=0.89
```

**Step 4: Prepare Test Data**
```python
test_X = _prepare_test_data(
    test_features_table=feature_artifacts.test_features_table,
    train_X_columns=train_X.columns
)
# Result: DataFrame with same 100+ feature columns, 1 row (user 2666), scaled appropriately
```

**Step 5: Generate Predictions**
```python
predictions = _generate_predictions(
    model=model,
    test_X=test_X,
    task_type=server.current_query_spec.task_type
)
# Result: {'y_pred': array([1]), 'y_prob': array([0.87])}
```

**Step 6: Format Prediction Results**
```python
test_with_predictions = _format_prediction_results(
    test_df=feature_artifacts.test_features_table,
    predictions=predictions,
    task_type=server.current_query_spec.task_type
)
# Result: DataFrame with columns ['__id', '__timestamp', 'y_pred', 'y_prob']
```

**Final Output: PredictionArtifact**
```python
return PredictionArtifact(
    test_with_predictions=pd.DataFrame({
        '__id': [2666],
        '__timestamp': [datetime(2021, 1, 1)],
        'y_pred': [1],           # Predicted class: engaged user
        'y_prob': [0.87]         # 87% probability of engagement
    })
)
```
