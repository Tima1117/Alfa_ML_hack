import numpy as np
import pytest

from ml_core.contracts import PredictionRequirements, TaskSpec
from ml_core.errors import UnknownMetricError
from ml_core.metrics import build_task_metric_fn, check_proba_metric_supported, get_metric, metric_needs_proba


def test_direction_is_fixed_per_metric():
    assert get_metric("rmse").greater_is_better is False
    assert get_metric("macro_f1").greater_is_better is True


def test_proba_metrics_are_flagged_and_only_these_are():
    assert get_metric("roc_auc").needs_proba is True
    assert get_metric("roc_auc").greater_is_better is True
    assert get_metric("log_loss").needs_proba is True
    assert get_metric("log_loss").greater_is_better is False
    for name in ("rmse", "mae", "r2", "macro_f1", "weighted_f1", "accuracy"):
        assert get_metric(name).needs_proba is False


def _binary_task_spec(metric_name: str, greater_is_better: bool) -> TaskSpec:
    return TaskSpec(
        task_type="classification",
        target_column="y",
        metric_name=metric_name,
        greater_is_better=greater_is_better,
        class_labels=["No", "Yes"],
        prediction_requirements=PredictionRequirements(output_columns=["id", "y"]),
    )


def test_proba_metric_supported_for_binary_classification():
    check_proba_metric_supported(_binary_task_spec("roc_auc", True))  # must not raise
    assert metric_needs_proba(_binary_task_spec("roc_auc", True)) is True
    assert metric_needs_proba(_binary_task_spec("macro_f1", True)) is False


def test_proba_metric_rejected_for_multiclass():
    task_spec = TaskSpec(
        task_type="classification",
        target_column="y",
        metric_name="roc_auc",
        greater_is_better=True,
        class_labels=["A", "B", "C"],
        prediction_requirements=PredictionRequirements(output_columns=["id", "y"]),
    )
    with pytest.raises(UnknownMetricError):
        check_proba_metric_supported(task_spec)


def test_proba_metric_rejected_for_regression_task_type():
    task_spec = TaskSpec(
        task_type="regression",
        target_column="y",
        metric_name="roc_auc",
        greater_is_better=True,
        prediction_requirements=PredictionRequirements(output_columns=["id", "y"]),
    )
    with pytest.raises(UnknownMetricError):
        check_proba_metric_supported(task_spec)


def test_unknown_metric_raises():
    with pytest.raises(UnknownMetricError):
        get_metric("not_a_metric")


def test_task_spec_metric_direction_mismatch_is_rejected():
    task_spec = TaskSpec(
        task_type="regression",
        target_column="y",
        metric_name="rmse",
        greater_is_better=True,  # wrong: rmse is lower-is-better
        prediction_requirements=PredictionRequirements(output_columns=["id", "y"]),
    )
    with pytest.raises(UnknownMetricError):
        build_task_metric_fn(task_spec)


def test_macro_f1_is_computed_over_the_full_fixed_label_set():
    """A fold whose y_true happens to miss a rare class must not silently
    average over fewer classes than the task actually has."""
    task_spec = TaskSpec(
        task_type="classification",
        target_column="y",
        metric_name="macro_f1",
        greater_is_better=True,
        class_labels=["A", "B", "RARE"],
        prediction_requirements=PredictionRequirements(output_columns=["id", "y"]),
    )
    metric_fn = build_task_metric_fn(task_spec)

    # y_true never contains class index 2 ("RARE"); a naive f1_score(average="macro")
    # without an explicit labels= would average over only {0, 1} here.
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    import sklearn.metrics as skm

    naive = skm.f1_score(y_true, y_pred, average="macro")
    bound = metric_fn(y_true, y_pred)
    assert bound != pytest.approx(naive)
    # The missing-class term contributes an F1 of 0, which must pull the
    # 3-class macro average below the naive 2-class one.
    assert bound < naive
