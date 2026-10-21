import numpy as np
import pandas as pd

from ml_core.contracts import ExperimentConfig, PreprocessingConfig
from ml_core.feature_steps import apply_feature_steps
from ml_core.prediction import refit_and_predict
from ml_core.preprocessing import FeatureColumns
from ml_core.profiling import infer_task_spec, profile_data
from ml_core.training import run_experiment
from ml_core.validation import create_validation


def _setup(paths, tmp_path):
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    data_profile = profile_data(paths["train"], paths["dict"], paths["test"])
    val_plan = create_validation(task_spec, paths["train"], tmp_path / "val", n_splits=3, seed=0)
    return task_spec, data_profile, val_plan


def test_builtin_steps_are_fit_on_the_fit_frame_only():
    fit_df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "c": ["x", "x", "y", "x"], "target": [1, 2, 3, 4]})
    other = pd.DataFrame({"a": [np.nan, 2.0], "c": ["y", "zzz"], "target": [0, 0]})
    cols = FeatureColumns(numeric=["a"], categorical=["c"], low_card_categorical=["c"])

    fit_out, (other_out,), new_cols = apply_feature_steps(
        ["missing_indicators", "count_encode", "row_nan_count"], fit_df, [other], cols, 20
    )

    assert "target" not in fit_out.columns and "target" not in other_out.columns
    # counts come from the fit frame: "y" appeared once there, unseen "zzz" -> 0
    assert other_out["c__count"].tolist() == [1.0, 0.0]
    assert other_out["a__isna"].tolist() == [1, 0]
    assert {"a__isna", "c__count", "__row_nan_count"} <= set(new_cols.numeric)
    assert new_cols.categorical == ["c"]
    assert list(other_out.columns) == list(fit_out.columns)


def test_drop_step_removes_column_from_feature_set():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    cols = FeatureColumns(numeric=["a", "b"])
    out, _, new_cols = apply_feature_steps(["drop:b"], df, [], cols, 20)
    assert list(out.columns) == ["a"]
    assert new_cols.numeric == ["a"]


def test_generated_python_step_runs_inside_cv_and_refit(regression_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(regression_dataset, tmp_path)
    step_file = tmp_path / "feat.py"
    step_file.write_text(
        "import numpy as np\n"
        "def make_features(df):\n"
        "    df['size_log'] = np.log1p(df['size'])\n"
        "    df['kind_size'] = df['building_kind'].astype(str) + '_' + (df['size'] > 1000).astype(str)\n"
        "    return df\n"
    )
    cfg = ExperimentConfig(
        model_name="hist_gradient_boosting",
        validation_id=val_plan.validation_id,
        feature_steps=[f"py:{step_file}", "missing_indicators"],
    )
    res = run_experiment(task_spec, cfg, val_plan, data_profile, regression_dataset["train"])
    assert res.status == "success", res.error_message
    # 3 original + 2 generated + isna flags for size and size_log
    assert res.diagnostics["n_features_after_steps"] == 3 + 2 + 2

    linear_cfg = cfg.model_copy(update={"model_name": "ridge", "preprocessing": PreprocessingConfig(mode="linear")})
    assert run_experiment(task_spec, linear_cfg, val_plan, data_profile, regression_dataset["train"]).status == "success"

    out = tmp_path / "submission.csv"
    pred = refit_and_predict(task_spec, cfg, data_profile, regression_dataset["train"], regression_dataset["test"], out)
    assert pred.status == "success", pred.error_message
    assert len(pd.read_csv(out)) == len(pd.read_csv(regression_dataset["test"]))


def test_unknown_or_broken_step_is_a_failed_result_not_an_exception(regression_dataset, tmp_path):
    task_spec, data_profile, val_plan = _setup(regression_dataset, tmp_path)
    cfg = ExperimentConfig(model_name="baseline", validation_id=val_plan.validation_id, feature_steps=["nope"])
    res = run_experiment(task_spec, cfg, val_plan, data_profile, regression_dataset["train"])
    assert res.status == "failed" and "unknown feature step" in res.error_message

    bad = tmp_path / "bad.py"
    bad.write_text("def make_features(df):\n    return df.iloc[:5]\n")
    cfg = cfg.model_copy(update={"feature_steps": [f"py:{bad}"]})
    res = run_experiment(task_spec, cfg, val_plan, data_profile, regression_dataset["train"])
    assert res.status == "failed" and "row count" in res.error_message
