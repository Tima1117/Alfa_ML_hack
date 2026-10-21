"""Phase 1: what is the metric, and how do we validate honestly?

Deterministic rules decide first; the LLM can refine, but only with answers
that pass the same checks (a metric must exist in ml_core's registry, a time
column must really separate train from test). This keeps a hallucinated
answer from silently changing what the whole run optimizes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ml_core.contracts import DataProfile, TaskSpec
from ml_core.metrics import available_metrics, get_metric

from agent.schemas import TaskUnderstanding

# (metric, regex) in the order ml_core supports them
_METRIC_PATTERNS = [
    ("macro_f1", r"macro[\s_-]*f1|f1[\s_-]*macro|average\s*=\s*[\"']macro"),
    ("weighted_f1", r"weighted[\s_-]*f1|f1[\s_-]*weighted"),
    ("rmse", r"\brmse\b|root[\s-]+mean[\s-]+squared"),
    ("mae", r"\bmae\b|mean[\s-]+absolute[\s-]+error"),
    ("r2", r"\br2\b|r²|coefficient of determination|коэффициент детерминации"),
    ("accuracy", r"\baccuracy\b"),
    # roc_auc/log_loss need a probability, and ml_core only supports that for binary
    # classification (ml_core/metrics.py Metric.needs_proba) — _fits_task below falls
    # back to the task-type default rather than matching these on a 3+-class task.
    ("roc_auc", r"roc[\s_-]*auc|\bauc\b|\bgini\b"),
    ("log_loss", r"log[\s_-]*loss|cross[\s-]*entropy"),
]
# metrics ml_core cannot optimize directly -> closest supported one
_UNSUPPORTED = [
    ("rmsle", r"\brmsle\b|root[\s-]+mean[\s-]+squared[\s-]+log", "rmse"),
    ("mape", r"\bs?mape\b", "mae"),
]


@dataclass
class MetricDecision:
    metric_name: str
    greater_is_better: bool
    source: str
    notes: list[str] = field(default_factory=list)


@dataclass
class ValidationDecision:
    strategy: str
    n_splits: int
    time_column: str | None = None
    group_column: str | None = None
    reason: str = ""


def _first_match(text: str, patterns: list[tuple[str, str]]) -> str | None:
    lowered = text.lower()
    metric_lines = [line for line in lowered.splitlines() if "метрик" in line or "metric" in line]
    for scope in (metric_lines, [lowered]):
        best: tuple[int, str] | None = None
        for line_no, line in enumerate(scope):
            for name, pattern in patterns:
                m = re.search(pattern, line)
                if m and (best is None or (line_no, m.start()) < best[0]):
                    best = ((line_no, m.start()), name)  # type: ignore[assignment]
        if best is not None:
            return best[1]
    return None


def guess_metric_from_text(text: str) -> str | None:
    return _first_match(text, _METRIC_PATTERNS)


def unsupported_metric_in_text(text: str) -> tuple[str, str | None] | None:
    found = _first_match(text, [(name, pattern) for name, pattern, _ in _UNSUPPORTED])
    if found is None:
        return None
    return found, next(sub for name, _, sub in _UNSUPPORTED if name == found)


def _fits_task(metric: str, default_spec: TaskSpec) -> bool:
    regression = {"rmse", "mae", "r2"}
    if (metric in regression) != (default_spec.task_type == "regression"):
        return False
    if get_metric(metric).needs_proba:
        # ml_core only has a probability-submission format for binary classification
        # (ml_core/metrics.py Metric.needs_proba) — a 3+-class task falls back to the
        # type default below instead of looping on status="unsupported" all run.
        return len(default_spec.class_labels or []) == 2
    return True


def decide_metric(task_text: str, default_spec: TaskSpec, llm: TaskUnderstanding | None) -> MetricDecision:
    task_type = default_spec.task_type
    notes: list[str] = []
    from_text = guess_metric_from_text(task_text)
    unsupported = unsupported_metric_in_text(task_text)
    if unsupported and (from_text is None or unsupported[0] == "rmsle"):
        name, substitute = unsupported
        notes.append(
            f"task.md mentions metric {name!r}, which ml_core cannot compute; optimizing "
            f"{substitute or default_spec.metric_name!r} instead"
            + (" (log1p target transform is the natural match)" if name == "rmsle" else "")
        )
        if substitute and _fits_task(substitute, default_spec):
            return MetricDecision(substitute, get_metric(substitute).greater_is_better, "text-substitute", notes)

    llm_metric = (llm.metric_name or "").strip().lower() if llm else ""
    if from_text and _fits_task(from_text, default_spec):
        if llm_metric and llm_metric != from_text:
            notes.append(f"LLM suggested metric {llm_metric!r}, text says {from_text!r}; using the text")
        return MetricDecision(from_text, get_metric(from_text).greater_is_better, "task.md", notes)
    if from_text and not _fits_task(from_text, default_spec):
        notes.append(
            f"task.md says {from_text!r}, but this task doesn't fit it "
            f"(needs binary classification for a probability metric); using the default instead"
        )
    if llm_metric in available_metrics() and _fits_task(llm_metric, default_spec):
        return MetricDecision(llm_metric, get_metric(llm_metric).greater_is_better, "llm", notes)
    notes.append(f"metric not recognized in task.md; default {default_spec.metric_name!r} for {task_type}")
    return MetricDecision(default_spec.metric_name, default_spec.greater_is_better, "default", notes)


def time_split_candidates(profile: DataProfile, train_path: Path, test_path: Path) -> list[str]:
    """Flagged numeric columns where every test value lies after every train value (a true later period)."""
    flagged = [c.name for c in profile.columns if c.role == "feature" and c.possible_time_or_group_column]
    if not flagged:
        return []
    train = pd.read_csv(train_path, usecols=flagged)
    test = pd.read_csv(test_path, usecols=[c for c in flagged])
    out = []
    for col in flagged:
        tr, te = train[col].dropna(), test[col].dropna()
        if not (pd.api.types.is_numeric_dtype(tr) and pd.api.types.is_numeric_dtype(te)) or tr.empty or te.empty:
            continue
        if tr.nunique() >= 2 and tr.max() < te.min():
            out.append(col)
    return out


def decide_validation(
    task_spec: TaskSpec, profile: DataProfile, time_candidates: list[str], llm: TaskUnderstanding | None
) -> ValidationDecision:
    default_folds = 5 if profile.n_rows_train <= 30_000 else 3
    if time_candidates:
        chosen = llm.time_column if llm and llm.time_column in time_candidates else time_candidates[0]
        return ValidationDecision(
            "time_split", 1, time_column=chosen,
            reason=f"all test rows have {chosen} later than any train row; validate on the latest train period",
        )
    if llm and llm.group_column:
        col = next((c for c in profile.columns if c.name == llm.group_column and c.role == "feature"), None)
        if col is not None and col.n_unique >= default_folds * 2 and (col.train_test_category_overlap or 0) < 0.5:
            return ValidationDecision("group_kfold", default_folds, group_column=col.name,
                                      reason=f"LLM: rows grouped by {col.name}, test groups are new")
    strategy = "stratified_kfold" if task_spec.task_type == "classification" else "kfold"
    return ValidationDecision(strategy, default_folds, reason="no time/group structure detected")


def drop_candidates(profile: DataProfile, validation: ValidationDecision, llm: TaskUnderstanding | None) -> list[str]:
    features = {c.name for c in profile.columns if c.role == "feature"}
    out = []
    if validation.time_column:
        out.append(validation.time_column)
    for col in (llm.columns_to_drop if llm else []):
        if col in features and col not in out:
            out.append(col)
    return out[:5]
