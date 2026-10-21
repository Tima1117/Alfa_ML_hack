# ml_core — ML backend for the AutoResearch agent

## Platform infrastructure

The flat submission ZIP is built with `python3 scripts/build_submission.py /path/to/alfa_solution.zip`.
The launcher restores the unchanged packages in the task directory, supervises time/memory/processes,
and independently validates `submission.csv`. See [docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md)
for Russian instructions, settings, clean-ZIP verification and optional Linux container checks.

> Карта репозитория (что где лежит, для человека и LLM): [project_context/repo_map.md](project_context/repo_map.md) ·
> презентация решения: [docs/presentation/agent_deck.html](docs/presentation/agent_deck.html) · вход для LLM-ассистентов: [AGENTS.md](AGENTS.md)

My part of the hackathon team's agent: a plain Python package that profiles
a tabular ML task, builds an honest validation split, trains/evaluates one
hypothesis at a time, and produces a checked, atomically-written
`submission.csv`. It has no LLM/agent logic and no CLI of its own — it's a
library the LLM-agent teammate calls to test hypotheses, and that the
infrastructure teammate wraps with the actual time/resource budget and
timeout enforcement.

Full integration contract (objects, operations, what each side needs to
provide): **[docs/CONTRACTS.md](docs/CONTRACTS.md)**.

## LLM research agent (`agent/`)

The second layer of the team's solution: an AutoResearch loop on top of
`ml_core` that understands the task, proposes hypotheses (LLM or a generic
playbook when no LLM is configured), runs them under one honest validation
split, analyses the results and writes the best `submission.csv`. Entry point
for infrastructure: `agent.research.run_research(task_md, deadline_ts)`;
contract and env variables: **[docs/AGENT_CONTRACT.md](docs/AGENT_CONTRACT.md)**.

```bash
python scripts/run_agent_local.py "Base/Datasets/Регрессия/task.md" --budget-s 600 --no-llm
```

## Install

```bash
python -m venv .venv
source .venv/Scripts/activate   # or .venv/bin/activate on Linux/macOS
pip install -r Base/requirements.txt
pip install pytest               # dev-only, not needed at submission time
```

Every library this package imports is in `Base/requirements.txt`, pinned to
the exact versions the platform will use — verified by installing that exact
file into a clean venv (`catboost 1.2.8`, `lightgbm 4.6.0`, `xgboost 2.1.4`,
`scikit-learn 1.5.2`, `pydantic 2.10.6`, …) and running everything below
against it.

## Run it

```bash
python examples/run_without_llm_regression.py
python examples/run_without_llm_multiclass.py
python -m pytest tests/ -q
```

Each example script profiles one of the two real provided datasets, runs a
handful of candidate configs, picks the best by validation score, and writes
a real `submission.csv` under `examples/runs/<task>/`.

## What's actually implemented

```
ml_core/
  contracts.py        TaskSpec, ExperimentConfig, ExperimentResult, DataProfile, ValidationPlan, PredictionResult
  metrics.py           rmse / macro_f1 / mae / r2 / accuracy / weighted_f1 / roc_auc / log_loss, direction fixed per metric
  profiling.py          profile_data(), infer_task_spec() — from data_dictionary.csv + sample_submission.csv
  validation.py          create_validation() — kfold / stratified_kfold / time_split / group_kfold
  preprocessing.py        native (NaN/category-dtype, for tree models) and linear (impute+encode+scale) modes
  model_registry.py        baseline, ridge, logistic_regression, random_forest, hist_gradient_boosting,
                            catboost, lightgbm, xgboost
  training.py               run_experiment() — leak-safe CV loop, soft time-budget check between folds
  prediction.py              refit_and_predict(), write_fallback_submission()
  experiment_log.py          append-only jsonl ledger, select_best()
tests/     29 tests: contracts, metrics, validation, preprocessing leakage, training, prediction format
examples/  two runnable, no-LLM demonstrations on the real competition data
docs/CONTRACTS.md  the integration contract for the other two teammates
```

## Read this before wiring up the agent: `profile_data` never cleans anything

`profiling.profile_data(...)` is read-only. It flags things — missing
values, duplicates, unseen test categories, a column that might be a
disguised time/group id — it does **not** drop rows, impute values, or touch
`train.csv`/`test.csv` in any way, ever. Concretely: it flags **29,342 fully
duplicated rows (a third of the whole classification train set!)** and
leaves every one of them in place, on purpose — see "Real findings" below
for why that's a deliberate call, not an oversight.

If the agent decides something in `DataProfile.warnings` should actually be
acted on (drop duplicates, cap outliers, whatever), that has to be an
explicit step it takes — currently there is no built-in dedup/outlier-removal
step (`ExperimentConfig.feature_steps` is reserved for exactly this but
nothing is implemented behind it yet). Also: the "looks like a time/group
column" heuristic has a confirmed false positive on the real data (`swin`/
`dwin` in the classification task — see below) — treat every warning as "go
look", not as "this is true". Full details: [docs/CONTRACTS.md](docs/CONTRACTS.md).

## Real findings from profiling the provided data (not hypothetical)

These drove concrete design decisions, not just EDA prose:

- **Regression task — `Year_Factor` is a disguised time column.** Train has
  values `{1, 2, 3, 4, 5}`; test has **only `{6}`** — an entirely unseen
  period. `profiling.profile_data` flags this automatically (generic
  heuristic: a low-cardinality numeric column whose train/test value sets
  barely overlap), not because the column name is hardcoded anywhere.
  Measured effect on the same CatBoost config:

  | validation | RMSE |
  |---|---|
  | plain 5-fold KFold | 44.80 ± 1.75 |
  | `time_split` on `Year_Factor` (train on periods 1–4, validate on 5) | **47.28** |

  Plain K-fold is ~5% optimistic here because it lets the model see rows
  from the same period as validation. `create_validation(..., strategy="time_split", time_column="Year_Factor")`
  is the one actually used in `examples/run_without_llm_regression.py`.

- **Classification task — real unseen-category and severe-imbalance risk.**
  `state` has values in `test.csv` (`CLO`, `ACC`) that never appear in
  `train.csv`; `proto` has 133 train categories. The rarest class (`Worms`)
  has 65 rows out of 87,670. Both are handled, not just noted: unseen
  categories become an explicit missing token rather than erroring
  (`tests/test_preprocessing_leakage.py`), and Macro F1 is computed over the
  task's full fixed 10-class label set rather than sklearn's default of
  "whatever classes happen to appear in this fold"
  (`tests/test_metrics.py::test_macro_f1_is_computed_over_the_full_fixed_label_set`).
  Measured: CatBoost macro F1 ≈ **0.537 ± 0.006** vs. a most-frequent-class
  baseline of 0.048.

- **Same classification task — a confirmed false positive from the
  time/group-column heuristic.** It also flags `swin`/`dwin`, but they're
  just near-constant TCP-window-size features (>99.9% of rows are `0` or
  `255`); the handful of other values each occur exactly once, so train/test
  disagreeing on them is noise, not a real signal. Don't trust a flag without
  checking the column — this is exactly why `profile_data` reports instead of
  acting.

- **The data dictionary's declared dtype is sometimes wrong.** `is_ftp_login`
  is declared `binary` but actually takes values `{0, 1, 2, 4}`. `profile_data`
  cross-checks declared vs. observed dtype and flags mismatches instead of
  trusting the dictionary blindly — this exact case is what it catches.

- **A two-category *string* column is not the same as a numeric 0/1 flag.**
  `building_class` (`"Commercial"`/`"Residential"`) was initially
  misclassified as a numeric "binary" feature by an early cardinality-only
  heuristic and handed to CatBoost as a float column, which crashed
  (`CatBoostError: Cannot convert 'Commercial' to float`). Fixed by checking
  dtype before cardinality in `profiling._infer_column_kind`; a regression
  test (`tests/conftest.py::regression_dataset` includes exactly this kind
  of column) now guards it.

## Experiments actually run (see `examples/runs/*/experiments/experiments.jsonl` for the raw log)

### Regression (RMSE, lower is better) — `time_split` on `Year_Factor`, 1 fold

| model | RMSE | notes |
|---|---|---|
| `baseline` (median) | 56.59 | reference point |
| `ridge` (linear mode, scaled) | 62.23 | worse than the median baseline — linear model underfits these features |
| `hist_gradient_boosting` | 54.36 | |
| `catboost` (iterations=400, depth=6) | 47.84 | |
| `catboost` + `log1p` target transform | **46.09** | best run; target is right-skewed (mean 84, max 993) so this is the expected direction |

(The 44.80/47.28 KFold-vs-time_split comparison above used a slightly
different CatBoost config than this table's row; both are in the jsonl log.)

### Classification (Macro F1, higher is better) — `stratified_kfold`, 5 folds

| model | Macro F1 | notes |
|---|---|---|
| `baseline` (most-frequent) | 0.048 | |
| `catboost` (iterations=300, depth=6, auto class weights) | **0.537 ± 0.006** | |

(Exact numbers for every run, including the ones elided above, are in the
jsonl logs produced by `examples/run_without_llm_*.py` — run them to
regenerate; nothing here is invented, everything above was actually executed
against `Base/Datasets/`.)

## Known limitations / explicitly deferred (priority order)

See "Next improvements" in [docs/CONTRACTS.md](docs/CONTRACTS.md) for the
full list and reasoning. Short version: no out-of-fold target encoding yet
(deferring it is explicitly allowed by the task brief), no hard
per-experiment timeout (only a soft between-folds check — infra needs to
wrap `run_experiment` in a subprocess + OS timeout if a candidate's fit time
is a real risk), `autogluon`/`lightautoml`/`flaml` are installed but
intentionally not wrapped (they'd run their own internal validation,
fighting this module's leakage/comparability controls instead of composing
with them), and XGBoost classification has no `class_weight` plumbing yet
(the other three classifiers do).

## What I did not do

I did not implement the LLM-agent (hypothesis generation / planning) or the
infrastructure layer (the actual `solution.py` entry point, deadline
tracking, hard timeouts, zip packaging). `docs/CONTRACTS.md` has a worked
example of exactly how those two pieces are expected to call into this
module.
