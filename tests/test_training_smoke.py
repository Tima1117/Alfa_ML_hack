import numpy as np

from ml_core.contracts import ExperimentConfig, PreprocessingConfig
from ml_core.model_registry import classification_sample_weight
from ml_core.profiling import infer_task_spec, profile_data
from ml_core.training import run_experiment
from ml_core.validation import create_validation


def _roc_auc_setup(paths, tmp_path):
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"], metric_name="roc_auc", greater_is_better=True)
    data_profile = profile_data(paths["train"], paths["dict"], paths["test"])
    val_plan = create_validation(task_spec, paths["train"], tmp_path / "val", n_splits=3, seed=0)
    return task_spec, data_profile, val_plan


def _setup(paths, tmp_path, strategy=None):
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    data_profile = profile_data(paths["train"], paths["dict"], paths["test"])
    val_plan = create_validation(task_spec, paths["train"], tmp_path / "val", n_splits=3, seed=0, strategy=strategy)
    return task_spec, data_profile, val_plan


def test_regression_end_to_end_with_linear_and_native_models(regression_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(regression_dataset, tmp_path)

    ridge_cfg = ExperimentConfig(
        model_name="ridge",
        validation_id=val_plan.validation_id,
        preprocessing=PreprocessingConfig(mode="linear"),
    )
    ridge_res = run_experiment(task_spec, ridge_cfg, val_plan, data_profile, regression_dataset["train"])
    assert ridge_res.status == "success"
    assert ridge_res.n_folds_completed == 3
    assert ridge_res.metric_name == "rmse"

    hgb_cfg = ExperimentConfig(model_name="hist_gradient_boosting", validation_id=val_plan.validation_id)
    hgb_res = run_experiment(task_spec, hgb_cfg, val_plan, data_profile, regression_dataset["train"])
    assert hgb_res.status == "success"

    baseline_cfg = ExperimentConfig(model_name="baseline", validation_id=val_plan.validation_id)
    baseline_res = run_experiment(task_spec, baseline_cfg, val_plan, data_profile, regression_dataset["train"])
    assert baseline_res.status == "success"
    # a fitted model should beat (lower RMSE than) the median-only baseline on signal we built in
    assert hgb_res.metric_value < baseline_res.metric_value


def test_classification_end_to_end_handles_unseen_test_category(classification_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(classification_dataset, tmp_path)
    cfg = ExperimentConfig(model_name="random_forest", validation_id=val_plan.validation_id)
    # random_forest uses the linear-style ColumnTransformer path under the hood via "native"? no —
    # random_forest is registered as a LINEAR_MODELS entry (needs a dense numeric matrix).
    cfg = cfg.model_copy(update={"preprocessing": PreprocessingConfig(mode="linear")})
    res = run_experiment(task_spec, cfg, val_plan, data_profile, classification_dataset["train"])
    assert res.status == "success"
    assert res.metric_name == "macro_f1"
    assert 0.0 <= res.metric_value <= 1.0


def test_run_experiment_rejects_native_mode_for_linear_only_model(regression_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(regression_dataset, tmp_path)
    cfg = ExperimentConfig(model_name="random_forest", validation_id=val_plan.validation_id)  # default mode="native"
    res = run_experiment(task_spec, cfg, val_plan, data_profile, regression_dataset["train"])
    assert res.status == "failed"
    assert "linear" in res.error_message


def test_run_experiment_never_raises_on_unknown_model(regression_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(regression_dataset, tmp_path)
    cfg = ExperimentConfig(model_name="definitely_not_a_model", validation_id=val_plan.validation_id)
    res = run_experiment(task_spec, cfg, val_plan, data_profile, regression_dataset["train"])
    assert res.status == "unsupported"
    assert res.error_message


def test_run_experiment_reports_failed_on_validation_id_mismatch(regression_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(regression_dataset, tmp_path)
    cfg = ExperimentConfig(model_name="ridge", validation_id="not-the-real-id")
    res = run_experiment(task_spec, cfg, val_plan, data_profile, regression_dataset["train"])
    assert res.status == "failed"
    assert "validation_id" in res.error_message


def test_run_experiment_respects_time_budget(regression_dataset, tmp_path):
    """A tiny time_budget_s must cut the CV loop short. The check runs once
    per fold (not inside a fold's own fit), so exactly which fold index it
    stops on is a timing race, not a contract — what's guaranteed is "fewer
    folds than requested, and a clear note why" rather than "zero folds"."""
    task_spec, data_profile, val_plan = _setup(regression_dataset, tmp_path)
    cfg = ExperimentConfig(
        model_name="ridge",
        validation_id=val_plan.validation_id,
        preprocessing=PreprocessingConfig(mode="linear"),
        time_budget_s=1e-9,  # expires almost immediately
    )
    res = run_experiment(task_spec, cfg, val_plan, data_profile, regression_dataset["train"])
    assert res.status in ("success", "timeout")
    assert res.n_folds_completed < res.n_folds_requested
    assert res.is_fully_comparable is False
    assert any("time_budget_s exceeded" in w for w in res.diagnostics.get("fold_warnings", []))


def test_run_experiment_reports_timeout_when_budget_already_expired(regression_dataset, tmp_path):
    """A budget that's already in the past (deterministic, unlike a razor-thin
    positive one) must yield zero completed folds and status="timeout"."""
    task_spec, data_profile, val_plan = _setup(regression_dataset, tmp_path)
    cfg = ExperimentConfig(
        model_name="ridge",
        validation_id=val_plan.validation_id,
        preprocessing=PreprocessingConfig(mode="linear"),
        time_budget_s=-3600.0,  # expired an hour "ago"
    )
    res = run_experiment(task_spec, cfg, val_plan, data_profile, regression_dataset["train"])
    assert res.status == "timeout"
    assert res.n_folds_completed == 0


def test_repeated_run_is_deterministic(regression_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(regression_dataset, tmp_path)
    cfg = ExperimentConfig(
        model_name="ridge",
        validation_id=val_plan.validation_id,
        preprocessing=PreprocessingConfig(mode="linear"),
        random_seed=123,
    )
    res_a = run_experiment(task_spec, cfg, val_plan, data_profile, regression_dataset["train"])
    res_b = run_experiment(task_spec, cfg, val_plan, data_profile, regression_dataset["train"])
    assert res_a.metric_value == res_b.metric_value
    assert res_a.fold_metrics == res_b.fold_metrics


def test_roc_auc_experiment_uses_real_probabilities_not_a_collapsed_hard_label(binary_classification_dataset, tmp_path):
    """Regression test for a real finding: scoring on model.predict() under imbalance
    collapses to the majority class for every row, which reads as a valid label but
    gives roc_auc ~= 0.5 (useless) no matter how good the model actually is. A fold score
    well above 0.5 here proves run_experiment is actually using predict_proba."""
    task_spec, data_profile, val_plan = _roc_auc_setup(binary_classification_dataset, tmp_path)
    cfg = ExperimentConfig(model_name="hist_gradient_boosting", validation_id=val_plan.validation_id)
    res = run_experiment(task_spec, cfg, val_plan, data_profile, binary_classification_dataset["train"])
    assert res.status == "success", res.error_message
    assert res.metric_name == "roc_auc"
    assert res.metric_value > 0.7  # the fixture's signal is strong; a collapsed prediction would read ~0.5


def test_roc_auc_on_multiclass_task_is_unsupported_not_a_crash(classification_dataset, tmp_path):
    task_spec = infer_task_spec(
        classification_dataset["train"], classification_dataset["dict"], classification_dataset["sub"],
        metric_name="roc_auc", greater_is_better=True,
    )
    data_profile = profile_data(classification_dataset["train"], classification_dataset["dict"], classification_dataset["test"])
    val_plan = create_validation(task_spec, classification_dataset["train"], tmp_path / "val", n_splits=3, seed=0)
    cfg = ExperimentConfig(model_name="hist_gradient_boosting", validation_id=val_plan.validation_id)
    res = run_experiment(task_spec, cfg, val_plan, data_profile, classification_dataset["train"])
    assert res.status == "unsupported"
    assert "roc_auc" in res.error_message


def test_classification_sample_weight_balances_xgboost_only():
    """Every other classifier this registry builds balances classes through its own
    constructor param (class_weight='balanced' or catboost's auto_class_weights) — xgboost
    has no such param, so it alone needs sample_weight at fit time to get the same effect.

    strategy defaults to 'sqrt_balanced', not sklearn's plain 'balanced': measured on the
    real UNSW-NB15 multiclass task (65 to 28000 rows/class), plain 'balanced' actually made
    mean macro F1 *worse* (0.611 -> 0.596) by overcorrecting so hard that mid-sized minority
    classes collapsed while fixing only the rarest one; 'sqrt_balanced' was a net improvement
    (0.611 -> 0.618) — see classification_sample_weight()'s docstring for the full numbers."""
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8:2 imbalance

    assert classification_sample_weight("catboost", y) is None
    assert classification_sample_weight("lightgbm", y) is None
    assert classification_sample_weight("logistic_regression", y) is None
    assert classification_sample_weight("xgboost", y, strategy=None) is None
    assert classification_sample_weight("xgboost", y, strategy="none") is None

    sw_default = classification_sample_weight("xgboost", y)
    sw_sqrt = classification_sample_weight("xgboost", y, strategy="sqrt_balanced")
    sw_balanced = classification_sample_weight("xgboost", y, strategy="balanced")
    for sw in (sw_default, sw_sqrt, sw_balanced):
        assert sw is not None and len(sw) == len(y)
        assert sw[y == 1][0] > sw[y == 0][0]  # the minority class gets the heavier weight
    assert np.array_equal(sw_default, sw_sqrt)  # the default strategy is sqrt_balanced
    assert sw_balanced[y == 1][0] > sw_sqrt[y == 1][0]  # plain balanced overcorrects harder


def test_xgboost_classification_runs_end_to_end_with_sample_weight(classification_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(classification_dataset, tmp_path)
    cfg = ExperimentConfig(model_name="xgboost", validation_id=val_plan.validation_id)
    res = run_experiment(task_spec, cfg, val_plan, data_profile, classification_dataset["train"])
    assert res.status == "success", res.error_message


def test_ensemble_experiment_soft_votes_regression(regression_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(regression_dataset, tmp_path)
    ridge = ExperimentConfig(model_name="ridge", validation_id=val_plan.validation_id, preprocessing=PreprocessingConfig(mode="linear"))
    hgb = ExperimentConfig(model_name="hist_gradient_boosting", validation_id=val_plan.validation_id)
    ens = ExperimentConfig(model_name="ensemble", validation_id=val_plan.validation_id, ensemble_of=[ridge, hgb])
    res = run_experiment(task_spec, ens, val_plan, data_profile, regression_dataset["train"])
    assert res.status == "success", res.error_message
    assert res.n_folds_completed == 3
    assert res.diagnostics["component_models"] == ["ridge", "hist_gradient_boosting"]


def test_ensemble_experiment_soft_votes_hard_label_classification(classification_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(classification_dataset, tmp_path)
    a = ExperimentConfig(model_name="xgboost", validation_id=val_plan.validation_id)
    b = ExperimentConfig(model_name="lightgbm", validation_id=val_plan.validation_id)
    ens = ExperimentConfig(model_name="ensemble", validation_id=val_plan.validation_id, ensemble_of=[a, b])
    res = run_experiment(task_spec, ens, val_plan, data_profile, classification_dataset["train"])
    assert res.status == "success", res.error_message
    assert res.metric_name == "macro_f1"


def test_ensemble_experiment_soft_votes_proba_metric(binary_classification_dataset, tmp_path):
    task_spec, data_profile, val_plan = _roc_auc_setup(binary_classification_dataset, tmp_path)
    a = ExperimentConfig(model_name="xgboost", validation_id=val_plan.validation_id)
    b = ExperimentConfig(model_name="hist_gradient_boosting", validation_id=val_plan.validation_id)
    ens = ExperimentConfig(model_name="ensemble", validation_id=val_plan.validation_id, ensemble_of=[a, b])
    res = run_experiment(task_spec, ens, val_plan, data_profile, binary_classification_dataset["train"])
    assert res.status == "success", res.error_message
    assert res.metric_name == "roc_auc" and res.metric_value > 0.7  # same sanity bar as the single-model test


def test_ensemble_experiment_rejects_fewer_than_two_or_nested_configs(regression_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(regression_dataset, tmp_path)
    ridge = ExperimentConfig(model_name="ridge", validation_id=val_plan.validation_id, preprocessing=PreprocessingConfig(mode="linear"))
    hgb = ExperimentConfig(model_name="hist_gradient_boosting", validation_id=val_plan.validation_id)

    one = ExperimentConfig(model_name="ensemble", validation_id=val_plan.validation_id, ensemble_of=[ridge])
    res = run_experiment(task_spec, one, val_plan, data_profile, regression_dataset["train"])
    assert res.status == "failed" and "at least 2" in res.error_message

    inner = ExperimentConfig(model_name="ensemble", validation_id=val_plan.validation_id, ensemble_of=[ridge, hgb])
    nested = ExperimentConfig(model_name="ensemble", validation_id=val_plan.validation_id, ensemble_of=[inner, hgb])
    res = run_experiment(task_spec, nested, val_plan, data_profile, regression_dataset["train"])
    assert res.status == "failed" and "nested" in res.error_message
