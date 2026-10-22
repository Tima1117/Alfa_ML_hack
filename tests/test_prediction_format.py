import pandas as pd

from ml_core.contracts import ExperimentConfig, PreprocessingConfig
from ml_core.prediction import refit_and_predict, write_fallback_submission
from ml_core.profiling import infer_task_spec, profile_data


def test_refit_and_predict_regression_matches_test_order_and_columns(regression_dataset, tmp_path):
    paths = regression_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    data_profile = profile_data(paths["train"], paths["dict"], paths["test"])
    cfg = ExperimentConfig(model_name="catboost", validation_id="refit", hyperparameters={"iterations": 50})
    # fall back to hist_gradient_boosting if catboost isn't installed in this environment
    from ml_core import model_registry

    if not model_registry.is_available("catboost"):
        cfg = cfg.model_copy(update={"model_name": "hist_gradient_boosting"})

    out_path = tmp_path / "submission.csv"
    result = refit_and_predict(task_spec, cfg, data_profile, paths["train"], paths["test"], out_path)

    assert result.status == "success", result.error_message
    assert out_path.exists()
    assert all(result.checks_passed.values())

    submission = pd.read_csv(out_path)
    test = pd.read_csv(paths["test"])
    assert list(submission.columns) == ["id", "target"]
    assert list(submission["id"]) == list(test["id"])  # exact row order preserved
    assert submission["target"].isna().sum() == 0


def test_refit_and_predict_classification_labels_within_allowed_set(classification_dataset, tmp_path):
    paths = classification_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    data_profile = profile_data(paths["train"], paths["dict"], paths["test"])
    cfg = ExperimentConfig(
        model_name="logistic_regression",
        validation_id="refit",
        preprocessing=PreprocessingConfig(mode="linear"),
    )
    out_path = tmp_path / "submission.csv"
    result = refit_and_predict(task_spec, cfg, data_profile, paths["train"], paths["test"], out_path)

    assert result.status == "success", result.error_message
    submission = pd.read_csv(out_path)
    # test.csv's "flow_state" includes category "D", unseen at train time — this must not crash
    # the fit/predict path, and predicted labels must still be only the ones seen in train's target.
    assert set(submission["label"].unique()).issubset(set(task_spec.class_labels))


def test_fallback_submission_is_always_valid(classification_dataset, tmp_path):
    paths = classification_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    data_profile = profile_data(paths["train"], paths["dict"], paths["test"])
    out_path = tmp_path / "submission.csv"
    result = write_fallback_submission(task_spec, data_profile, paths["train"], paths["test"], out_path)
    assert result.status == "success"
    assert all(result.checks_passed.values())


def test_refit_and_predict_does_not_write_file_when_checks_fail(regression_dataset, tmp_path, monkeypatch):
    paths = regression_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    data_profile = profile_data(paths["train"], paths["dict"], paths["test"])

    # Force a submission-format violation: wrong output_columns order/content.
    bad_task_spec = task_spec.model_copy(
        update={
            "prediction_requirements": task_spec.prediction_requirements.model_copy(
                update={"output_columns": ["id", "target", "extra_column_that_does_not_exist"]}
            )
        }
    )
    cfg = ExperimentConfig(model_name="ridge", validation_id="refit", preprocessing=PreprocessingConfig(mode="linear"))
    out_path = tmp_path / "submission.csv"
    result = refit_and_predict(bad_task_spec, cfg, data_profile, paths["train"], paths["test"], out_path)

    assert result.status == "failed"
    assert not out_path.exists()


def test_refit_and_predict_roc_auc_writes_probabilities_not_labels(binary_classification_dataset, tmp_path):
    paths = binary_classification_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"], metric_name="roc_auc", greater_is_better=True)
    data_profile = profile_data(paths["train"], paths["dict"], paths["test"])
    cfg = ExperimentConfig(model_name="hist_gradient_boosting", validation_id="refit")
    out_path = tmp_path / "submission.csv"
    result = refit_and_predict(task_spec, cfg, data_profile, paths["train"], paths["test"], out_path)

    assert result.status == "success", result.error_message
    assert result.checks_passed.get("probabilities_in_unit_range") is True
    submission = pd.read_csv(out_path)
    values = submission["churn"]
    assert pd.api.types.is_float_dtype(values)
    assert values.between(0, 1).all()
    assert values.nunique() > 5  # real per-row probabilities, not a constant/collapsed column


def test_refit_and_predict_ensemble_regression(regression_dataset, tmp_path):
    paths = regression_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    data_profile = profile_data(paths["train"], paths["dict"], paths["test"])
    ridge = ExperimentConfig(model_name="ridge", validation_id="refit", preprocessing=PreprocessingConfig(mode="linear"))
    hgb = ExperimentConfig(model_name="hist_gradient_boosting", validation_id="refit")
    ens = ExperimentConfig(model_name="ensemble", validation_id="refit", ensemble_of=[ridge, hgb])
    out_path = tmp_path / "submission.csv"
    result = refit_and_predict(task_spec, ens, data_profile, paths["train"], paths["test"], out_path)

    assert result.status == "success", result.error_message
    assert all(result.checks_passed.values())
    submission = pd.read_csv(out_path)
    test = pd.read_csv(paths["test"])
    assert list(submission["id"]) == list(test["id"])
    assert submission["target"].isna().sum() == 0


def test_refit_and_predict_ensemble_hard_label_classification(classification_dataset, tmp_path):
    paths = classification_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    data_profile = profile_data(paths["train"], paths["dict"], paths["test"])
    a = ExperimentConfig(model_name="xgboost", validation_id="refit")
    b = ExperimentConfig(model_name="lightgbm", validation_id="refit")
    ens = ExperimentConfig(model_name="ensemble", validation_id="refit", ensemble_of=[a, b])
    out_path = tmp_path / "submission.csv"
    result = refit_and_predict(task_spec, ens, data_profile, paths["train"], paths["test"], out_path)

    assert result.status == "success", result.error_message
    assert result.checks_passed.get("labels_within_allowed_set") is True
    submission = pd.read_csv(out_path)
    assert submission["label"].isin(["Yes", "Maybe", "No"]).all()
