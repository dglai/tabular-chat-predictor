# Background (why we’re changing it)

* In the current write-up, DFS preprocessing is done **upfront** and produces fixed `.npz` feature tensors before the query is analyzed; then an LLM-led “Predictive Workflow” defines labels, fits TabPFN, and runs predictions. &#x20;
* That sequence assumes **predefined target tables** and a **query routed over an already materialized DFS dataset**.&#x20;
* We want to support **flexible, query-driven data generation**. As a first step (while we design a richer query parser later), we’ll accept a normalized **`TableInfo`** and refactor the internals to **materialize task tables → build DFS features → train/predict with TabPFN** on demand (no pre-declared DFS table). This reorders parts of the “Predictive Workflow Execution” while keeping the same ML core (labels → fit → predict → optional SHAP) from the doc. &#x20;

# What we’re delivering now

We will implement/refactor these **three stages**, driven by a single `TableInfo` input (task\_type included):

1. **Task Table Materialization (train/test)**
2. **Task-scoped DFS Feature Engineering** (build features for those task rows only)
3. **TabPFN Training & Prediction** (classification or regression)

> Out of scope (for now): NL query parsing, task definition heuristics, or LLM agent routing. Those remain as future work and are intentionally decoupled.

---

# What stays aligned with exsiting

* We keep the **core pipeline: labels → model training → batch prediction → optional explanations**, as summarized in “Predictive Workflow Execution” and subsequent output stages. &#x20;
* We keep **TabPFN** as the model and support **both classification and regression**.&#x20;
* We keep the notion of **returning a predicted table with timestamps and values/probabilities**.&#x20;

# What changes 

* **No predeclared/monolithic DFS dataset**: instead of “run DFS first, then query,” we **derive DFS features from the task tables** we just materialized. This removes the requirement to have `.npz` per target table beforehand. (In the MD, DFS precedes query processing.)&#x20;
* **Explicit task tables** become the contract between labeling/feature generation and model training—this is not spelled out as a hard boundary in the doc, but we elevate it as a first-class stage to enable flexible, query-driven data generation.&#x20;

---

# Refactor plan (actions to take)

## A. Define stable I/O contracts (internal APIs)

### A1) `TableInfo` (input)

* Minimal fields used in Step 1:
  `entity_type`, `entity_ids` (optional for batch), `id_column`, `base_table`, `ts_current`, `label_spec` (contains event source, horizon/lookback, time column), `task_type` ∈ {classification, regression}.
  (This mirrors the “task definition”/labeling idea but arrives **pre-parsed** for now.)

### A2) `TaskTables` (output of Stage 1)

* **Train table**: one row per (entity\_id, snapshot\_ts) with label (classification: 0/1; regression: continuous).
* **Test table**: same schema, `label = NULL`, `snapshot_ts = ts_current`.
* These become the single source of truth for both **DFS windows** and **model fitting** (replacing reliance on global `.npz`).
* Expected downstream usage aligns with the “training labels” and “batch predictions” steps in the doc. &#x20;

### A3) `FeatureArtifacts` (output of Stage 2)

* `train_features_table`, `test_features_table`, shared `feature_list`.
* Feature windows are derived from each row’s `(entity_id, snapshot_ts)` and the **lookback** in `label_spec`.
* Matches the MD’s intent of producing **temporal feature matrices**, but scoped to our task rows (not global `.npz`).&#x20;

### A4) `PredictionArtifact` (output of Stage 3)

* `test_with_predictions` (id, snapshot\_ts, y\_pred / y\_hat, probability if classification) + minimal metrics (e.g., CV AUC for classification or RMSE for regression).
* Same user-facing semantics as “Predicted Table with Labels” in the doc.&#x20;