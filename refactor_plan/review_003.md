There are a few places where the current design introduces churn or redundancy that can be smoothed out for consistency and simplicity. Here’s what I’d change and why.

Key inconsistencies and redundancies observed
- Time column rename churn
  - Stage 1 uses `__timestamp`.
  - Stage 2 renames to `cutoff_time`.
  - Stage 3 renames back to `__timestamp`.
  - This adds repeated renames and increases the risk of bugs.

- Static entity features joined in Stage 1, then ignored
  - Stage 1 joins “static entity features” from the target table onto both train and test tables.
  - Stage 2 discards these and rebuilds features via DFS from the RDB.
  - Result: duplicated work and unused data.

- Labels are not part of features in Stage 2
  - Stage 3 needs labels. The example relies on `server.task_tables` to get them.
  - This is global, mutable state and not concurrency-safe. It also couples Stage 3 to the server object.

- Feature names vs reserved columns
  - Stage 2’s `_generate_features` can return `__id` and `cutoff_time` among the columns, and Stage 3 later has to exclude them again. Sometimes test feature generation uses all of `train_features.columns`, which may accidentally include reserved columns rather than only feature columns.

- Config naming mismatches and override sprawl
  - `QuerySpec.dfs_depth` overrides server DFS config `max_depth`.
  - TabPFN overrides may appear as separate top-level fields (e.g., `tabpfn_ensemble_size`), while `server.tabpfn_config` uses different names. This requires hand-written mapping logic in merge functions.

- `id_column` redundancy
  - You can infer the primary key from `RDBDataset` metadata for most cases. Keeping `id_column` in `QuerySpec` is useful as an override but should be optional; right now it’s a required field.

- Link prediction schema inconsistencies
  - Stage 1 sometimes generates `__target_id` for link tasks, but Stage 2 and 3 only consider `__id`. There isn’t a consistent multi-key policy across stages.

- Transform pipeline and key mappings rebuilt per request
  - The transform pipeline and key mappings are deterministic for a given RDB/schema and `target_table`. They should be cached to avoid recomputation.

- Concurrency hazards on server state
  - `server.current_query_spec` and `server.task_tables` are mutated per request. This is unsafe with concurrent requests.

- Sampling and TabPFN max_samples not coordinated
  - Stage 1 samples training rows. Stage 3 may sub-sample again to meet TabPFN limits. This can lead to effective sampling more than intended.

- Timezone/dtype policy not explicit
  - Timestamps appear naive; DFS and DuckDB benefit from clear timezone normalization (e.g., UTC).

- Feature column naming contains spaces and punctuation
  - DFS-generated column names can be unfriendly for downstream ML or serialization. Canonicalizing names consistently (and storing a mapping) improves robustness.


How to improve consistency (actionable changes)

1) Canonicalize reserved column names across all stages
- Use a single internal schema everywhere:
  - `__entity_id` for entity id (or `__id` if you want to keep it short).
  - `__time` for timestamp.
  - `__label` for label.
  - Optional for link tasks: `__target_id` (and if needed, `__edge_id` or `__pair_id`).
- Keep these names unchanged in Stage 1, 2, and 3.
- Only adapt to external library requirements at the boundaries with a thin mapping:
  - Example: a helper `_to_fastdfs(df)` that projects `__time -> cutoff_time` and back, without changing the canonical in-memory schema elsewhere.

2) Make Stage 1 minimal; defer all features to Stage 2
- Remove the join of “static entity features” in Stage 1.
- Stage 1 should return only:
  - Train: `[__id, __time, __label]` (+ `__target_id` for link tasks).
  - Test: `[__id, __time]` (+ `__target_id` for link tasks).
- Let Stage 2 compute all features (including those from the target table) to avoid duplication and ambiguity.

3) Ship labels alongside features to Stage 3
- Add labels to `FeatureArtifacts` (for training) or pass `TaskTables` explicitly into Stage 3.
- Avoid relying on `server.task_tables` as mutable server state.
- Example:
  - `FeatureArtifacts.train_labels`: `pd.Series` aligned to `train_features_table` row order.
  - Or pass `(feature_artifacts, task_tables)` to Stage 3 and have a strict join key policy.

4) Centralize feature schema and reserved columns
- In `FeatureArtifacts`, include:
  - `feature_columns`: list of actual feature columns (excluding reserved columns).
  - `reserved_columns`: list of reserved columns present in the tables.
  - Optionally `dtypes`, `scaler_stats`, `feature_name_map` (see next point).
- Stage 3 should never have to recompute which columns are features; it should use `feature_columns` directly.

5) Canonicalize feature names and preserve a mapping
- Introduce a function to canonicalize feature names (snake_case, no spaces or punctuation).
- Keep a reversible `name_map: Dict[str, str]` from canonical name -> original human-readable name.
- Store `name_map` in `FeatureArtifacts` so you can map back for explainability.

6) Standardize config override mechanism
- Use nested overrides to avoid per-field top-level entries and name mismatches.
- In `QuerySpec`, add:
  - `dfs_overrides: Dict[str, Any]` (keys match `dfs_config` keys exactly, e.g., `max_depth`).
  - `tabpfn_overrides: Dict[str, Any]` (keys match `tabpfn_config` keys exactly, e.g., `ensemble_size`, `max_samples`).
- Remove bespoke fields like `dfs_depth`, `tabpfn_max_samples`, etc., and the need for mapping functions.

7) Infer `id_column` by default
- Make `QuerySpec.id_column` optional. Infer from `RDBDataset` metadata (the primary key).
- Keep `id_column` as an optional override for edge cases.

8) First-class link prediction support
- Define a `TaskKind` and `KeySpec`:
  - `task_type`: Literal["classification", "regression", "link"]
  - `KeySpec`: `{'primary': '__id', 'secondary': '__target_id'}` or a tuple of keys.
- Update Stage 2 `_generate_key_mappings` to support multiple keys:
  - Example: `{"__id": "users.Id", "__target_id": "posts.Id"}`
- Stage 3 `prepare_*` methods should accept and maintain composite keys, and metrics/predictions should keep them intact.

9) Cache transform pipeline, transformed RDB, and key mappings
- Compile the transform pipeline at server startup; keep it in `server`.
- Cache “transformed RDB” per dataset version.
- Cache key mappings by `target_table`.

10) Replace server-level mutable context with per-request context
- Define a `RequestContext` (immutable per request):
  - `query_spec`, `rdb_dataset`, `dfs_config_effective`, `tabpfn_config_effective`, and optional handles to cached resources.
- Pass `ctx` explicitly to all stages instead of reading/writing `server.current_query_spec` and `server.task_tables`.

11) Coordinate sampling across stages
- Decide where sampling happens and make it visible downstream:
  - If Stage 1 does sampling, Stage 3 should respect it and avoid re-subsampling unless required by TabPFN limits; if it must, compute an “effective sample size” and log/report it.
- Add `sampling_info` metadata in `TaskTables` or `FeatureArtifacts` to track what fraction and how was sampled.

12) Explicit timezone and dtype policy
- Normalize all timestamps to UTC at ingestion.
- Document dtypes for reserved columns:
  - `__id`: integer or string
  - `__target_id`: integer or string (only for link)
  - `__time`: pandas datetime64[ns, UTC]
  - `__label`: int (classification), float (regression)

13) Make async story consistent
- Either:
  - Keep stages async and run I/O- or compute-heavy operations concurrently; or
  - Make stages synchronous and run them in executors when needed.
- Mix-and-match works, but pick one approach and document it to avoid confusion.

14) Standardize output schema and enrich artifacts
- `PredictionArtifact` should include:
  - `predictions_df` with reserved columns (`__id`, `__time`, optional `__target_id`) and predicted outputs.
  - Optional: `metrics`, `model_info`, `feature_schema`, `query_id`, timing, and any caching hit/miss info.
- This gives a consistent, debuggable output contract.


Small example adjustments to illustrate the improvements

- Reserved columns and mapping adapter
```python
RESERVED = {
    "id": "__id",
    "target_id": "__target_id",       # optional
    "time": "__time",
    "label": "__label",
}

def to_fastdfs_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    # Do not mutate input; thin view for FastDFS
    cutoff = df[[RESERVED["id"], RESERVED["time"]]].copy()
    cutoff.rename(columns={RESERVED["time"]: "cutoff_time"}, inplace=True)
    return cutoff

def from_fastdfs_time(df: pd.DataFrame) -> pd.DataFrame:
    if "cutoff_time" in df.columns and RESERVED["time"] not in df.columns:
        df = df.rename(columns={"cutoff_time": RESERVED["time"]})
    return df
```

- Minimal Stage 1 outputs (no entity features)
```python
@dataclass
class TaskTables:
    train_table: pd.DataFrame  # [__id, __time, __label] (+ __target_id for link)
    test_table: pd.DataFrame   # [__id, __time] (+ __target_id for link)
```

- FeatureArtifacts carries schema and labels
```python
@dataclass
class FeatureArtifacts:
    train_features_table: pd.DataFrame   # includes reserved columns
    test_features_table: pd.DataFrame
    feature_columns: list                # excludes reserved columns
    train_labels: Optional[pd.Series]    # aligned to train_features_table
    name_map: Optional[Dict[str, str]] = None  # canonical -> original
```

- Config overrides in QuerySpec
```python
@dataclass
class QuerySpec:
    target_table: str
    entity_ids: Optional[list]
    id_column: Optional[str]  # optional; infer by default
    ts_current: datetime
    task_type: Literal["classification", "regression", "link"]
    label_spec: str
    horizon_days: int = 90
    sampling_rate: float = 0.3
    dfs_overrides: Dict[str, Any] = field(default_factory=dict)
    tabpfn_overrides: Dict[str, Any] = field(default_factory=dict)
    query_id: Optional[str] = None
```

- Stage 2 uses adapter without renaming core schema
```python
train_cutoff_df = to_fastdfs_cutoff(task_tables.train_table)
test_cutoff_df = to_fastdfs_cutoff(task_tables.test_table)
train_features = from_fastdfs_time(_generate_features(...))
test_features  = from_fastdfs_time(_generate_features(..., feature_names=feature_columns))
```

- Stage 3 consumes `FeatureArtifacts` without accessing server state
```python
train_X = feature_artifacts.train_features_table[feature_artifacts.feature_columns]
train_y = feature_artifacts.train_labels.to_numpy()
```

Migration plan (incremental, low risk)
1) Introduce reserved column constants and the FastDFS mapping adapter. Keep old names temporarily and deprecate them.
2) Remove Stage 1’s entity-feature join; adjust Stage 2 to compute all features.
3) Add `feature_columns` and `train_labels` to `FeatureArtifacts`; refactor Stage 3 to use them; stop reading labels from `server.task_tables`.
4) Replace per-field overrides with `dfs_overrides` and `tabpfn_overrides`; update merge logic to simple dict overlays.
5) Make `id_column` optional; infer from schema; keep override behavior if provided.
6) Add link-prediction multi-key support through a `KeySpec` and update Stage 2 mappings.
7) Cache transform pipeline and key mappings on the server; introduce a `RequestContext` passed down the stages; remove `server.current_query_spec` mutations.
8) Add canonical feature name mapping; pass `name_map` for explainability; document timezone/dtype policy; tighten async/sync story.

Expected benefits
- Fewer renames and less glue code.
- Clearer, safer data flow (no global mutable state).
- Consistent schemas that are easy to validate and test.
- Easier concurrency handling.
- Cleaner configuration model with fewer special cases.
- Better support for link prediction and explainability.

If you want, I can provide a patch-style diff for your current `refactor_interface_design` to reflect these changes concretely.