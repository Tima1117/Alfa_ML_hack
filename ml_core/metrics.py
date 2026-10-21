"""Metric registry.

The metric for a task is fixed in TaskSpec (metric_name + greater_is_better)
and is never read from ExperimentConfig — an experiment is not allowed to
pick its own evaluator. training.run_experiment always resolves the metric
through get_metric(task_spec.metric_name) and cross-checks
greater_is_better against the registry, so a mismatched TaskSpec fails loudly
instead of silently flipping the comparison direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

from ml_core.errors import UnknownMetricError

if TYPE_CHECKING:
    from ml_core.contracts import TaskSpec


@dataclass(frozen=True)
class Metric:
    name: str
    greater_is_better: bool
    fn: Callable[[np.ndarray, np.ndarray], float]
    # True for metrics that score a continuous probability/confidence (roc_auc, log_loss)
    # rather than a hard prediction. Binary classification only (see build_task_metric_fn) —
    # a private task pairing one of these with 3+ classes needs a real multi-column
    # probability submission format, which isn't implemented yet; run_experiment reports
    # status="unsupported" for that case instead of guessing at a format.
    needs_proba: bool = False


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """y_score: probability of the positive class (the higher-coded binary label), shape (n,)."""
    return float(roc_auc_score(y_true, y_score))


def _log_loss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """y_score: probability of the positive class, shape (n,) — sklearn infers the complement."""
    return float(log_loss(y_true, y_score, labels=[0, 1]))


_REGISTRY: dict[str, Metric] = {
    "rmse": Metric("rmse", greater_is_better=False, fn=_rmse),
    "mae": Metric("mae", greater_is_better=False, fn=_mae),
    "r2": Metric("r2", greater_is_better=True, fn=_r2),
    "macro_f1": Metric("macro_f1", greater_is_better=True, fn=_macro_f1),
    "weighted_f1": Metric("weighted_f1", greater_is_better=True, fn=_weighted_f1),
    "accuracy": Metric("accuracy", greater_is_better=True, fn=_accuracy),
    "roc_auc": Metric("roc_auc", greater_is_better=True, fn=_roc_auc, needs_proba=True),
    "log_loss": Metric("log_loss", greater_is_better=False, fn=_log_loss, needs_proba=True),
}

# Metrics computed for context even when they are not the competition metric.
REGRESSION_AUX_METRICS = ("mae", "r2")
CLASSIFICATION_AUX_METRICS = ("accuracy", "weighted_f1")


def get_metric(name: str) -> Metric:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownMetricError(f"Unknown metric {name!r}. Known metrics: {known}") from exc


def available_metrics() -> list[str]:
    return sorted(_REGISTRY)


def metric_needs_proba(task_spec: "TaskSpec") -> bool:
    return get_metric(task_spec.metric_name).needs_proba


def check_proba_metric_supported(task_spec: "TaskSpec") -> None:
    """Raise UnknownMetricError if metric_name needs probabilities on a task shape
    we don't support yet (anything but binary classification — see Metric.needs_proba)."""
    if not metric_needs_proba(task_spec):
        return
    if task_spec.task_type != "classification" or len(task_spec.class_labels or []) != 2:
        raise UnknownMetricError(
            f"metric {task_spec.metric_name!r} needs a probability output, which this package only "
            f"supports for binary classification; task_type={task_spec.task_type!r} with "
            f"{len(task_spec.class_labels or [])} class(es) would need a multi-column probability "
            "submission format that isn't implemented."
        )


def build_task_metric_fn(task_spec: "TaskSpec") -> Callable[[np.ndarray, np.ndarray], float]:
    """Metric function bound to the task's full, fixed label set.

    For classification this matters more than it looks: sklearn's
    f1_score(average="macro") without an explicit `labels=` argument only
    averages over classes that actually appear in the y_true/y_pred given to
    it. On a small CV fold a rare class (this competition has one with only
    65 training rows total) can be absent by chance, which would silently
    average over fewer classes and read as a better score than it should.
    Binding `labels` to every class in task_spec.class_labels makes every
    fold's macro F1 computed over the same fixed denominator.
    """
    metric = get_metric(task_spec.metric_name)
    if metric.greater_is_better != task_spec.greater_is_better:
        raise UnknownMetricError(
            f"TaskSpec.greater_is_better={task_spec.greater_is_better} disagrees with the registered "
            f"direction for metric {task_spec.metric_name!r} ({metric.greater_is_better}). "
            "This looks like a contract bug, not a modeling choice — fix TaskSpec."
        )
    if task_spec.task_type == "classification" and task_spec.metric_name in ("macro_f1", "weighted_f1"):
        average = "macro" if task_spec.metric_name == "macro_f1" else "weighted"
        n_classes = len(task_spec.class_labels or [])
        return partial(f1_score, average=average, labels=list(range(n_classes)), zero_division=0)
    return metric.fn
