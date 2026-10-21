# ml_core — contracts for the LLM-agent and infrastructure teammates

This is the integration contract for `ml_core`, the ML backend. It is a
plain Python package (`import ml_core` or `from ml_core import ...`), not a
service — every function is a normal, synchronous Python call. All shared
objects are [pydantic](https://docs.pydantic.dev/) v2 models with
`extra="forbid"`, so a typo'd field or a hallucinated config key raises
`pydantic.ValidationError` immediately instead of being silently ignored.

Every model is JSON-serializable (`.model_dump_json()` /
`Model.model_validate_json(...)`), which matters because we don't know how
the final `solution.py` will be structured. The organizers' own
`baseline.py` runs the agent's Python through a tool (`run_python`) that
spawns **a brand-new interpreter process for every call** — "Каждый вызов —
новый процесс. Файлы сохраняются, переменные — нет." If the real agent ends
up built the same way, nothing in this package can rely on in-memory state
surviving between calls: artifacts (fold indices, experiment logs, model
files) are always written to disk and reloaded by path. This also means
`ml_core` works fine if the agent instead runs as one long-lived process —
the disk-based design is a superset, not a special case.

## The four operations

```python
from ml_core import profiling, validation, training, prediction

task_spec   = profiling.infer_task_spec(train_path, dict_path, sample_submission_path)
data_profile = profiling.profile_data(train_path, dict_path, test_path)
val_plan    = validation.create_validation(task_spec, train_path, out_dir="runs/val")
result      = training.run_experiment(task_spec, config, val_plan, data_profile, train_path, "runs/experiments")
pred_result = prediction.refit_and_predict(task_spec, config, data_profile, train_path, test_path, "submission.csv")
```

1. **`profiling.infer_task_spec(...) -> TaskSpec`** — reads `data_dictionary.csv`'s
   `role` column (`id`/`feature`/`target`) and `sample_submission.csv`'s
   columns. This is deliberately **not** based on parsing `task.md`'s free
   text — that file is prose for an LLM, not a guaranteed schema, and the
   PDF explicitly says private-task metrics "may differ" from the two public
   ones. `metric_name`/`greater_is_better` default by `task_type` (`rmse`/
   `macro_f1`) and the default is recorded in `TaskSpec.notes` — **read that
   field**, and pass `metric_name=`/`greater_is_better=` explicitly whenever
   you actually know them (from `task.md`, or because you're looking at one
   of the two public tasks).

2. **`profiling.profile_data(...) -> DataProfile`** — per-column missingness,
   cardinality, a train/test categorical-overlap check, and a heuristic flag
   for "this low-cardinality numeric column's train/test value sets barely
   overlap — might be a time period or group id, consider `time_split` /
   `group_kfold`". This heuristic is what caught the real regression task's
   `Year_Factor` (train ∈ {1..5}, test ∈ {6} exactly) — see README for the
   concrete effect that has on validation honesty. It also has occasional
   false positives on columns with high missingness or just few categories
   sampled unevenly; treat it as a prompt to look, not a verdict. Confirmed
   false positive on the real multiclass task: `swin`/`dwin` get flagged, but
   they're just near-constant TCP-window-size features (>99.9% of rows are
   `0` or `255`) whose handful of other values each occur exactly once —
   train/test disagreeing on those one-off values is noise, not a real
   temporal/group signal. Compare against the genuine hit on the same task
   (`state`, below) before trusting a flag.

   **`profile_data` only *reports* — it never modifies, drops, or imputes
   anything, on disk or anywhere else.** Concretely, for both real tasks it
   flags a large number of fully duplicated rows (33 in the regression
   train.csv, **29,342 — a third of all rows** — in the classification
   train.csv) and does not remove a single one. That's deliberate, not an
   oversight: for the classification task specifically, repeated network
   flows with identical feature values are a plausible property of the
   domain, not obviously noise, and the rarest class (`Worms`) has only 65
   rows total — an automatic dedup could silently delete exactly the signal
   that matters most. If your agent decides duplicates (or outliers, or
   anything else) should be removed, that has to be an explicit, logged
   hypothesis your agent makes — e.g. a feature step (see
   `ExperimentConfig.feature_steps` below; no dedup/outlier-removal step is
   implemented, since dropping rows would break the fold row-positions), never
   something that happens for you inside `profile_data` or `run_experiment`.

3. **`validation.create_validation(...) -> ValidationPlan`** — persists fold
   row-positions to an `.npz` file plus a fingerprint of the train file used.
   Strategies: `"kfold"`, `"stratified_kfold"` (default for classification;
   auto-caps `n_splits` to the rarest class's count, with a warning — never
   silently), `"time_split"` (needs `time_column`; last `n_splits` distinct
   values each become a held-out fold, training on everything earlier),
   `"group_kfold"` (needs `group_column`).
   **`validation.load_fold_indices(plan, train_path)` re-checks the
   fingerprint and row count before returning anything** — reusing a plan
   against data it wasn't built for raises `ValidationMismatchError` instead
   of silently scoring against the wrong split.

4. **`training.run_experiment(...) -> ExperimentResult`** — fits
   `experiment_config.model_name` under K-fold CV, fitting every
   preprocessing step **only on that fold's training rows**, and returns a
   result that **never raises** for an ML-level failure (bad model name,
   unavailable library, a fit that throws, a config/validation mismatch —
   all come back as `status in {"failed", "unsupported", "timeout"}` with a
   human-readable `error_message`). It *can* raise `pydantic.ValidationError`,
   but only while you're constructing the `ExperimentConfig` itself, before
   `run_experiment` is ever called.
   The metric is **always** `task_spec.metric_name` — `ExperimentConfig` has
   no way to change the evaluator. If `experiments_dir` is given, every call
   appends one line to `runs/.../experiments.jsonl` (see
   `experiment_log.select_best(...)`).

5. **`prediction.refit_and_predict(...) -> PredictionResult`** — refits the
   chosen config on the *entire* train file, predicts on test, runs format
   checks (row count, id order, column names, no NaN/Inf, classification
   labels within the observed training set), and **only then** writes
   `submission.csv`, atomically (`os.replace`, never a half-written file).
   If checks fail, nothing is written — a bad result can never clobber a
   previously good `submission.csv` at the same path.
   **`prediction.write_fallback_submission(...)`** does the same thing with
   the cheapest possible model (median / most-frequent). **Call this first,
   before spending any real time budget** — the platform hard-kills the
   process after 3 hours, so a guaranteed-valid file should exist on disk as
   early as possible, then get overwritten by better ones as the research
   loop finds them.

   Note on refitting a boosted tree model on the *entire* training set (not
   a CV fold): it genuinely takes longer than any single fold did — on the
   real classification task, ~125s for CatBoost against the ~97s/fold seen
   during 5-fold CV — and `refit_and_predict` prints nothing while it runs.
   This is exactly the atomic-write design paying off: if the caller is
   killed mid-refit (by a human assuming the process hung, or by the
   platform's own timeout), the previous file at that path — the fallback,
   or an earlier better submission — is untouched, never corrupted or
   half-written. Observed firsthand building this: a terminal that looks
   frozen for ~2 minutes during the final refit is normal, not a bug: don't
   interrupt it, and if the agent/infra loop logs progress, log something
   right before this call so a human watching doesn't assume a hang.

## Contract objects (see `ml_core/contracts.py` for the authoritative, typed definitions)

- **`TaskSpec`** — `task_type` (`"regression"` | `"classification"`),
  `target_column`, `id_columns`, `metric_name`, `greater_is_better`,
  `class_labels` (required for classification), `prediction_requirements`
  (`output_columns` — the exact columns/order for `submission.csv`).
  Build it once per task with `infer_task_spec`, reuse it for every
  experiment on that task.

- **`ExperimentConfig`** — `model_name` (see below), `hyperparameters` (dict,
  passed straight through to the underlying estimator), `preprocessing`
  (`PreprocessingConfig`: `mode` is the field that matters most — see "modes"
  below), `random_seed`, `validation_id` (must match the `ValidationPlan`
  you're scoring against), `time_budget_s` (soft, see "time budget" below),
  `n_jobs`.

- **`ExperimentResult`** — `status`, `metric_value`/`metric_std`/
  `fold_metrics`, `n_folds_completed`/`n_folds_requested` (compare these
  before trusting a result — `ExperimentResult.is_fully_comparable` is
  `True` only when every requested fold actually ran), `validation_id`
  (**two results are only safe to compare head-to-head if this matches** —
  different splits are not comparable), `diagnostics` (free-form dict,
  includes `fold_warnings` for things like an early stop from `time_budget_s`),
  `error_message`.

### Models (`model_name`)

| name | library | mode | notes |
|---|---|---|---|
| `baseline` | sklearn `Dummy*` | either | median (regression) / most-frequent (classification); always available, this is the reference point every real config should beat |
| `ridge` | sklearn | **linear only** | |
| `logistic_regression` | sklearn | **linear only** | `class_weight="balanced"` by default |
| `random_forest` | sklearn | **linear only** | sklearn's RandomForest needs a dense numeric matrix, unlike the others below |
| `hist_gradient_boosting` | sklearn | native | `categorical_features="from_dtype"` (sklearn ≥1.4) |
| `catboost` | catboost | native | `cat_features` passed as column names; `auto_class_weights="Balanced"` for classification |
| `lightgbm` | lightgbm | native | relies on pandas `category` dtype auto-detection |
| `xgboost` | xgboost | native | `enable_categorical=True, tree_method="hist"` |

`model_registry.is_available(name)` checks importability at call time —
`run_experiment`/`refit_and_predict` return `status="unsupported"` rather
than crashing if a library isn't installed in a given environment.
`autogluon` / `lightautoml` / `flaml` are in `requirements.txt` but **not**
wrapped here on purpose — see "Next improvements" below.

### Probability-scored metrics (`roc_auc`, `log_loss`) — binary classification only

Found by stress-testing against a synthetic task shaped like a plausible
private one (imbalanced binary target, metric stated as ROC-AUC in
`task.md`): `macro_f1`/`weighted_f1`/`accuracy` all score a *hard*
prediction, but `roc_auc` and `log_loss` need a probability. Before this,
`run_experiment`/`refit_and_predict` always called `model.predict()` for
classification — under real class imbalance this collapses to predicting
the majority class for every row, which is a valid *label* but a useless
*probability* (ROC-AUC ≈ 0.5, i.e. no better than random), even though the
underlying model may be fine.

When `task_spec.metric_name` is `"roc_auc"` or `"log_loss"`
(`metrics.metric_needs_proba(task_spec)`), both functions now call
`model.predict_proba(X)` instead and submit `P(label == task_spec.class_labels[1])`
as a float in `[0, 1]` — looked up by position in the fitted model's
`classes_`, not assumed to be column 1 (`model_registry.positive_class_proba`).
`TaskSpec.prediction_requirements.output_columns` doesn't change (still
`[id_col, target_col]`); only the *values* in the target column become
probabilities instead of class-label strings, and `_check_submission`
validates them as finite numbers in `[0, 1]` rather than against the
allowed-label set. `infra/solution.py`'s independent CSV check reads the
same decision (`infra_task_spec.json`'s `needs_proba` field, written after
`run_research` settles on the real metric — task.md/the LLM can pick a
different metric than the preliminary guess) so it doesn't wrongly reject a
valid probability as "not one of train's labels".

**Scope: binary classification only.** A metric needing probabilities on a
3+-class task would need a genuinely different submission shape (one
probability column per class, driven by `sample_submission.csv`'s own
columns) that isn't implemented — `check_proba_metric_supported` raises
before any data is touched, which `run_experiment`/`refit_and_predict` turn
into `status="unsupported"`/`status="failed"` rather than silently emitting
a wrong-shaped file.

### `preprocessing.mode`

- `"native"` (default): for the four tree models. Numeric NaNs are left as
  NaN — every one of those libraries splits on missingness natively.
  Categorical NaNs and genuinely unseen categories (confirmed in this
  competition's own data: `state` has values in `test.csv` that never
  appear in `train.csv`) become an explicit `"__missing__"` token, never an
  exception.
- `"linear"`: for `ridge`/`logistic_regression`/`random_forest`. Median/
  most-frequent imputation, one-hot for low-cardinality categoricals
  (`handle_unknown="ignore"`), a small frequency-encoder for high-cardinality
  ones (target encoding is intentionally **deferred** — see below).
- **`ridge`/`logistic_regression`/`random_forest` + `mode="native"` is
  rejected up front** with a clear `status="failed"` message — they can't
  consume raw NaN/category-dtype input, and mode is pydantic's own default
  of `"native"`, so this combination is an easy footgun for an
  agent-generated config. `model_registry.requires_linear_mode(name)` is the
  check you can run yourself before even calling `run_experiment`.

### `feature_steps` (`ml_core/feature_steps.py`)

Applied **inside** the CV loop (fit on the training fold only, then
transform the validation fold / test), to feature columns only — steps never
see the target or the id. The validation split is untouched, so results
with and without steps share a `validation_id` and are directly comparable.

| step | what it does |
|---|---|
| `drop:<column>` | drop one feature column |
| `missing_indicators` | `<col>__isna` flags for columns with NaN in the training fold (max 50) |
| `count_encode` | `<col>__count` = category frequency in the training fold (unseen → 0) |
| `row_nan_count` | `__row_nan_count` per row |
| `py:<path>` | generated module with a stateless `make_features(df) -> df` (same rows, same order; may add/drop columns; ±inf become NaN) |

New columns are typed by dtype (numeric → numeric, otherwise categorical).
An unknown step or a broken `make_features` comes back as
`status="failed"` with the reason, never an exception. `py:` code runs
in-process: checking it is safe and deterministic is the caller's job
(`agent/features.py` does that in a sandboxed subprocess first).

### Time budget — what's enforced and what isn't

`ExperimentConfig.time_budget_s` is checked **once per CV fold**, not inside
a fold's own `.fit()` call. A fold that hangs or runs unexpectedly long
cannot be interrupted from inside `run_experiment` — this is a soft budget.
**If a given model/hyperparameter combination needs a hard timeout, the
infrastructure layer has to run `run_experiment` in a subprocess and kill it
at the OS level.** A `subprocess.run(..., timeout=...)` is a reasonable way
to do that, but it is not a resource sandbox (no memory cap, no filesystem
isolation) — don't describe it as one in whatever you build around it.

Nobody inside `ml_core` knows when the platform's 3-hour clock started —
`run_experiment`/`refit_and_predict` take time budgets as explicit
parameters from the caller, they never read a wall clock of their own.

### Leakage / unseen-category guarantees, and what's explicitly deferred

- Every fit (imputer, encoder, scaler, the model itself) happens inside the
  per-fold loop, on that fold's training rows only — `tests/test_preprocessing_leakage.py`
  verifies this behaviorally (fit on a 90/10-split fold, confirm the
  transform of an all-"rare-value" held-out fold reflects the *fold's*
  statistics, not its own).
- **Target encoding is not implemented.** The task brief explicitly allows
  deferring it ("Target encoding делай с корректным out-of-fold механизмом
  либо отложи") — it needs extra nested-CV machinery to stay leak-safe, and
  the frequency encoder already used for `"linear"` mode covers the
  high-cardinality case without that risk. `category-encoders` is in
  `requirements.txt` if a teammate wants to add it properly later.
- Macro/weighted F1 is computed with an **explicit, fixed label set**
  (`metrics.build_task_metric_fn`), not sklearn's default behavior of
  averaging only over labels present in a given fold's `y_true`/`y_pred`.
  This matters concretely here: the classification task's rarest class
  (`Worms`) has 65 rows out of 87,670 — a fold that happened to miss it
  would otherwise silently average over 9 classes instead of 10 and read as
  better than it is. See `tests/test_metrics.py::test_macro_f1_is_computed_over_the_full_fixed_label_set`.

## What infrastructure needs to provide

- **A hard per-experiment timeout** — the LLM-agent teammate's
  `agent.executor.SubprocessExecutor` (see `docs/AGENT_CONTRACT.md` in their
  branch) already runs every `ml_core` call in its own process and kills it
  on timeout. Infra doesn't need to build this again; only swap in a
  different `Executor` if that behavior doesn't fit.
- **The overall 3-hour deadline**, translated into `time_budget_s` values
  passed into `ExperimentConfig` (or, once wired to the agent, into
  `run_research(task_md, deadline_ts=...)`'s absolute deadline — the agent
  never reads the platform clock itself) and into when
  `write_fallback_submission` gets called (as early as possible) and when
  the loop stops trying new configs and calls `refit_and_predict` with
  whatever `experiment_log.select_best(...)` currently returns.
- **CPU/thread budgeting** across whatever else runs concurrently — `n_jobs`
  on `ExperimentConfig` is passed straight through to the underlying
  estimator (`-1` by default, meaning "let the library decide", normally all
  cores).
- A decision on where `experiments_dir`/validation `out_dir` live inside the
  task's working directory (relative paths are resolved from the task.md's
  directory per the platform's own convention).

### ZIP packaging — flat layout, confirmed by the organizers' own instructions, now implemented

From the organizers' own upload instructions (not just the case PDF, which
only says solution.py must be at the archive root and *can* import other
modules — it doesn't spell out the packaging constraint below):

> ZIP: все файлы в корне архива, без папок. Копируются только корневые
> `.py` и `requirements.txt`; вложенные папки и прочие файлы не
> копируются → модуль из подпапки будет недоступен → можно получить 0.

In translation: the platform's unzip step only takes root-level `.py` files
and `requirements.txt` — any file inside a folder is silently dropped. Both
`ml_core/` and `agent/` are real Python packages, so zipping the repo as-is
would not work.

**Implemented** by the infrastructure teammate — not by renaming/rewriting
every module import (the approach this section used to suggest), but by
bundling the untouched source of every file in `ml_core/*.py` and
`agent/*.py` as plain Python string literals inside one root file
(`bundle_sources.py`, with a SHA-256 per file checked before use). The
actual submission ZIP has exactly three root files: `solution.py`,
`bundle_sources.py`, `requirements.txt`. At startup, `solution.py` writes
those strings back out as real files under the original `ml_core/`/`agent/`
package layout in a fresh temp directory inside the task's working
directory, then runs from there — so every existing import, `python -m
agent._worker`, and the feature-sandbox's sibling-file lookup keep working
completely unmodified. Build it with `python3 scripts/build_submission.py
<output.zip>`; verify the built archive (including a mode that imitates the
platform's root-files-only copy) with `scripts/verify_submission.py`. Full
details, the time/memory watchdog around the agent process, and the
independent final CSV check: `docs/INFRASTRUCTURE.md`.

## What the LLM agent needs to provide

- A `TaskSpec` (call `infer_task_spec`, override `metric_name`/
  `greater_is_better` if `task.md`'s prose gives a different metric than the
  default).
- Each hypothesis as an `ExperimentConfig` — `model_name`, `hyperparameters`,
  `preprocessing.mode` matched to that model (see table above),
  `validation_id` pointing at a `ValidationPlan` built once per task.
- Reading `ExperimentResult.diagnostics`/`error_message` to decide the next
  hypothesis — `run_experiment` never crashes the agent's loop, so a bad
  guess just comes back with `status != "success"` and a reason.
- Calling `experiment_log.select_best(experiments_dir, validation_id=...)`
  rather than tracking "the best so far" itself — this recomputes from the
  append-only log every time, so it can't go stale if a later experiment
  config object isn't retained in memory (see the fresh-subprocess caveat
  above).

## Next improvements (deliberately out of v1 scope, in priority order)

1. **Out-of-fold target encoding** for the high-cardinality categoricals
   (`category_encoders`, already in `requirements.txt`) — likely upside on
   `facility_type` (60 categories) and `proto` (133 categories).
2. ~~Feature engineering step registry~~ — done, see `feature_steps` above.
3. **A thin `autogluon`/`flaml` adapter model**, run through the *same*
   `run_experiment` contract (so its result is directly comparable to the
   hand-rolled candidates) rather than letting it do its own internal
   validation.
4. **Hard per-experiment timeout wrapper** (subprocess + OS kill) as a small
   helper in this package, if infra wants it provided rather than built
   itself.
5. **`sample_weight` plumbing** for XGBoost classification imbalance (it has
   no `class_weight` parameter, unlike the other three classifiers).
