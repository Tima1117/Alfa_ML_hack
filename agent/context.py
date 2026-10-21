"""Compact text summaries of the task and of the experiment history for prompts.

Everything here is about spending as few tokens as possible while still
giving the LLM what it needs to reason: one line per column, top results
plus the most recent attempts (with their error messages) instead of the
whole log.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml_core.contracts import DataProfile, ExperimentConfig, ExperimentResult, PreprocessingConfig, TaskSpec, ValidationPlan

_DEFAULT_PREPROCESSING = PreprocessingConfig().model_dump()
_MAX_COLUMNS = 150
_DESCRIPTION_CHARS = 70


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def load_descriptions(data_dictionary_path: str | Path) -> dict[str, str]:
    try:
        dd = pd.read_csv(data_dictionary_path)
    except Exception:
        return {}
    desc_col = next((c for c in ("description_ru", "description", "desc") if c in dd.columns), None)
    if desc_col is None or "feature" not in dd.columns:
        return {}
    out = {}
    for name, desc, unit in zip(dd["feature"], dd[desc_col], dd["unit"] if "unit" in dd.columns else [None] * len(dd)):
        text = "" if pd.isna(desc) else str(desc)
        if unit is not None and not pd.isna(unit) and str(unit).strip():
            text = f"{text} [{unit}]"
        out[str(name)] = text[:_DESCRIPTION_CHARS]
    return out


def describe_metric(task_spec: TaskSpec) -> str:
    direction = "higher is better" if task_spec.greater_is_better else "lower is better"
    return f"Metric being optimized: {task_spec.metric_name} ({direction})."


def describe_task(task_spec: TaskSpec, profile: DataProfile, descriptions: dict[str, str]) -> str:
    """Data brief (deliberately without the metric: it is decided after the understanding step)."""
    lines = [
        f"Task type: {task_spec.task_type}. Target: `{task_spec.target_column}`.",
        f"Rows: train={profile.n_rows_train}, test={profile.n_rows_test}. "
        f"Fully duplicated train rows: {profile.duplicated_rows_train}.",
    ]
    if task_spec.task_type == "classification":
        counts = ", ".join(f"{k}={v}" for k, v in sorted(profile.target_summary.items(), key=lambda kv: -kv[1]))
        lines.append(f"Class counts: {counts}")
    else:
        ts = profile.target_summary
        keys = ("mean", "std", "min", "25%", "50%", "75%", "max", "n_nonpositive")
        lines.append("Target summary: " + ", ".join(f"{k}={_fmt(ts[k])}" for k in keys if k in ts))

    features = [c for c in profile.columns if c.role == "feature"]
    lines.append(f"Features ({len(features)}): name | kind | missing% | n_unique | notes | description")
    for c in features[:_MAX_COLUMNS]:
        notes = []
        if c.possible_time_or_group_column:
            notes.append("train/test values barely overlap")
        if c.train_test_category_overlap is not None and c.train_test_category_overlap < 1:
            notes.append(f"unseen test categories ({c.train_test_category_overlap:.0%} overlap)")
        if c.is_high_cardinality:
            notes.append("high-cardinality")
        lines.append(
            f"- {c.name} | {c.inferred_kind} | {c.missing_rate * 100:.1f} | {c.n_unique} | "
            f"{'; '.join(notes)} | {descriptions.get(c.name, '')}"
        )
    if len(features) > _MAX_COLUMNS:
        rest = [c.name for c in features[_MAX_COLUMNS:]]
        lines.append(f"- … and {len(rest)} more: {', '.join(rest)}")
    return "\n".join(lines)


def describe_validation(plan: ValidationPlan) -> str:
    extra = ""
    if plan.time_column:
        extra = f" on `{plan.time_column}` (latest period held out)"
    if plan.group_column:
        extra = f" grouped by `{plan.group_column}`"
    return f"Validation: {plan.strategy}{extra}, {plan.n_splits} fold(s), fixed for the whole run."


def compact_config(config: ExperimentConfig) -> dict[str, Any]:
    """Only the parts of a config that differ from defaults — what the LLM needs to see."""
    pre = {k: v for k, v in config.preprocessing.model_dump().items() if _DEFAULT_PREPROCESSING.get(k) != v}
    out: dict[str, Any] = {"model_name": config.model_name}
    if config.hyperparameters:
        out["hyperparameters"] = config.hyperparameters
    if pre:
        out["preprocessing"] = pre
    if config.feature_steps:
        out["feature_steps"] = [short_step(s) for s in config.feature_steps]
    return out


def short_step(step: str) -> str:
    if step.startswith("py:"):
        return "py:" + Path(step[3:]).name
    return step


def _result_line(r: ExperimentResult, title: str | None) -> str:
    cfg = json.dumps(compact_config(r.config), ensure_ascii=False, separators=(",", ":"))
    if r.status == "success" and r.metric_value is not None:
        score = f"{r.metric_value:.5g} ± {(r.metric_std or 0):.2g}"
    else:
        score = f"{r.status.upper()}: {(r.error_message or '')[:200]}"
    label = f" «{title}»" if title else ""
    return f"- [{r.experiment_id}]{label} {score} | {r.duration_s:.0f}s | {cfg}"


def describe_history(
    results: list[ExperimentResult],
    titles: dict[str, str],
    greater_is_better: bool,
    *,
    top_k: int = 8,
    recent_k: int = 6,
) -> str:
    if not results:
        return "No experiments yet."
    ok = [r for r in results if r.status == "success" and r.metric_value is not None]
    ok.sort(key=lambda r: r.metric_value if greater_is_better else -r.metric_value, reverse=True)
    top = ok[:top_k]
    top_ids = {r.experiment_id for r in top}
    recent = [r for r in results[-recent_k:] if r.experiment_id not in top_ids]
    lines = [f"Experiments run: {len(results)} ({len(ok)} successful). Best first:"]
    lines += [_result_line(r, titles.get(r.experiment_id)) for r in top]
    if recent:
        lines.append("Most recent other attempts:")
        lines += [_result_line(r, titles.get(r.experiment_id)) for r in recent]
    return "\n".join(lines)
