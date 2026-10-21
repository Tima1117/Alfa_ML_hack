"""Prompts for the research agent.

English instructions (fewer tokens than Russian for the same content); the
task.md text is passed through as-is. The lists of models, preprocessing
options and feature steps are rendered from ml_core so the prompt can never
offer something ml_core does not implement.
"""

from __future__ import annotations

import json

from ml_core import model_registry
from ml_core.contracts import PreprocessingConfig
from ml_core.feature_steps import BUILTIN_STEPS
from ml_core.metrics import available_metrics

from agent.features import ALLOWED_IMPORTS, MAX_NEW_COLUMNS


def _models_block() -> str:
    only = {"ridge": ", regression only", "logistic_regression": ", classification only"}
    lines = []
    for name in sorted(model_registry.ALL_MODELS):
        mode = "linear" if model_registry.requires_linear_mode(name) else model_registry.suggested_mode(name)
        lines.append(f"- {name} (preprocessing.mode={mode}{only.get(name, '')})")
    return "\n".join(lines)


def _preprocessing_block() -> str:
    fields = PreprocessingConfig.model_json_schema()["properties"]
    out = []
    for name, spec in fields.items():
        options = spec.get("enum") or [o.get("const") for o in spec.get("anyOf", []) if "const" in o]
        default = spec.get("default")
        out.append(f"- {name}: default {json.dumps(default)}" + (f", one of {options}" if options else ""))
    return "\n".join(out)


UNDERSTAND_SYSTEM = """You are the lead data scientist of an autonomous AutoML agent. You read a task description \
and a data profile and answer with ONE JSON object, nothing else."""


def understand_user(task_text: str, data_brief: str, time_candidates: list[str], flagged: list[str]) -> str:
    return f"""Task description (task.md):
<<<
{task_text}
>>>

Data profile:
{data_brief}

Columns whose train/test values barely overlap: {flagged or "none"}.
Of those, columns where every test value is later than every train value: {time_candidates or "none"}.

Answer with JSON:
{{
  "metric_name": one of {available_metrics()} - the metric the task is scored on,
  "greater_is_better": true/false,
  "time_column": a column from the "later" list above if validation must hold out the latest period, else null,
  "group_column": a column identifying groups (e.g. entity id) that must not be split across folds, else null,
  "columns_to_drop": feature columns that are likely useless or harmful for test (ids, period codes); [] if none,
  "summary": 2-4 sentences: what drives the target, what is risky, what to try first
}}"""


RESEARCH_SYSTEM = f"""You are an autonomous ML researcher improving a tabular model by running experiments (AutoResearch).
Each round you see the data profile and the experiment history, then propose the next hypotheses.
An external harness runs every hypothesis with honest cross-validation (fixed split, metric fixed by the task;
you cannot change either). Only validation results count, so propose ideas that should generalize to unseen test data.

Available models:
{_models_block()}

preprocessing fields (only list the ones you change):
{_preprocessing_block()}
target_transform "log1p" applies to regression with a non-negative target only.

Feature steps (applied inside each CV fold, fit on training rows only):
- built-in: {", ".join(BUILTIN_STEPS)}, "drop:<column>"
- your own code via "feature_code": a Python function
    def make_features(df):  # df = feature columns only (no target, no id), one row per object
        ...
        return df          # same rows in the same order; you may add or drop columns
  Rules: imports only from {sorted(ALLOWED_IMPORTS)}; must be deterministic and row-wise (do NOT compute
  statistics across rows such as groupby means or quantiles of the whole frame - use the built-in steps for
  that); handle missing values; at most {MAX_NEW_COLUMNS} new columns; use domain knowledge (ratios, differences,
  interactions, bins, date parts...). Code that fails a safety/quality check is rejected and you will see why.

Each hypothesis is a CHANGE applied to the current best config (base "best") or a fresh config (base "scratch"):
{{
  "title": short name,
  "rationale": why it should help (cite the evidence from the history/profile),
  "kind": "model" | "params" | "preprocessing" | "features",
  "base": "best" | "scratch",
  "model_name": optional, switches the model (hyperparameters then start from library defaults),
  "hyperparameters": dict of library hyperparameters to set (merged into the base's when the model is unchanged),
  "preprocessing": dict of preprocessing fields to change,
  "add_feature_steps": list of steps to append to the base's steps,
  "feature_steps": full replacement list of steps (optional; e.g. [] removes all),
  "feature_code": optional Python source defining make_features (added as one more step)
}}
Threads, seeds, verbosity and GPU options are set by the harness - do not set them. Iteration counts above 4000
are clamped. Keep the time budget in mind: the history shows how long each run took.
Harness facts: for classification only, models already use balanced class weights (class_weight "balanced",
CatBoost auto_class_weights "Balanced"; set them to null to turn off), so LightGBM is_unbalance changes nothing;
regression models take no class weights. Use only parameters of the chosen model's own library.
XGBoost overfits on categorical columns unless max_cat_threshold is set (e.g. 32).

Strategy: prefer one clear change per hypothesis so the effect is attributable; mix cheap parameter tweaks with
feature ideas; do not repeat configurations already in the history; learn from failures (their errors are shown);
a change within the fold-to-fold noise (±std) is not a real improvement. Set "stop": true only when you believe
further experiments are very unlikely to help.

Answer with ONE JSON object:
{{"analysis": "what the results so far tell you (2-5 sentences)", "hypotheses": [ ...1-N hypotheses... ], "stop": false}}"""


def research_user(
    *,
    data_brief: str,
    validation_brief: str,
    understanding: str,
    best_config_json: str,
    history: str,
    notes: list[str],
    minutes_left: float,
    n_hypotheses: int,
) -> str:
    notes_block = "\n".join(f"- {n}" for n in notes[-8:]) or "- none"
    return f"""{data_brief}

{validation_brief}
Initial assessment: {understanding or "n/a"}

Current best config: {best_config_json}

{history}

Notes from previous rounds (rejected ideas, sandbox errors, your earlier analysis):
{notes_block}

Time left for experiments: about {minutes_left:.0f} min. Propose up to {n_hypotheses} hypotheses."""
