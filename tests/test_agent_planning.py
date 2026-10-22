from pathlib import Path

import pandas as pd
import pytest

from agent.features import FeatureSandbox, static_check
from agent.hypotheses import HypothesisError, materialize, sanitize_hyperparameters
from agent.schemas import Hypothesis, TaskUnderstanding
from agent.task_files import locate_task_files
from agent.understand import decide_metric, decide_validation, guess_metric_from_text, time_split_candidates
from ml_core.contracts import ExperimentConfig, PreprocessingConfig
from ml_core.profiling import infer_task_spec, profile_data

REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "Base" / "Datasets"


def test_materialize_merges_on_same_model_and_resets_on_switch():
    best = ExperimentConfig(
        model_name="lightgbm", hyperparameters={"n_estimators": 500, "num_leaves": 31},
        preprocessing=PreprocessingConfig(target_transform="log1p"), feature_steps=["count_encode"], validation_id="v",
    )
    cfg, _ = materialize(Hypothesis(hyperparameters={"num_leaves": 63}, add_feature_steps=["row_nan_count"]), best, "v")
    assert cfg.hyperparameters == {"n_estimators": 500, "num_leaves": 63}
    assert cfg.feature_steps == ["count_encode", "row_nan_count"]
    assert cfg.preprocessing.target_transform == "log1p"

    cfg, notes = materialize(Hypothesis(model_name="ridge", hyperparameters={"alpha": 3.0}), best, "v")
    assert cfg.hyperparameters == {"alpha": 3.0}
    assert cfg.preprocessing.mode == "linear" and cfg.preprocessing.target_transform == "log1p"

    same, _ = materialize(Hypothesis(model_name="ridge", hyperparameters={"alpha": 3.0}), best, "v")
    assert same.experiment_id == cfg.experiment_id  # stable id -> duplicates are detectable

    with pytest.raises(HypothesisError):
        materialize(Hypothesis(model_name="gpt_boost"), best, "v")
    with pytest.raises(HypothesisError):
        materialize(Hypothesis(preprocessing={"mode": "quantum"}), best, "v")


def test_sanitize_removes_reserved_keys_and_clamps():
    hp, notes = sanitize_hyperparameters("catboost", {"iterations": 100000, "thread_count": 64, "task_type": "GPU", "depth": 14})
    assert hp == {"iterations": 4000, "depth": 10, "allow_writing_files": False}
    assert len(notes) == 4


def test_metric_from_the_real_task_texts():
    reg = (DATASETS / "Регрессия" / "task.md").read_text(encoding="utf-8")
    cls = (DATASETS / "Многоклассовая классификация" / "task.md").read_text(encoding="utf-8")
    assert guess_metric_from_text(reg) == "rmse"
    assert guess_metric_from_text(cls) == "macro_f1"
    assert guess_metric_from_text("Метрика: MAE.") == "mae"


def test_decide_metric_validates_llm_answers(regression_dataset):
    spec = infer_task_spec(regression_dataset["train"], regression_dataset["dict"], regression_dataset["sub"])
    assert decide_metric("Метрика — **MAE**", spec, TaskUnderstanding(metric_name="r2")).metric_name == "mae"
    assert decide_metric("no metric words", spec, TaskUnderstanding(metric_name="r2")).metric_name == "r2"
    # a classification metric for a regression task is ignored
    assert decide_metric("no metric words", spec, TaskUnderstanding(metric_name="macro_f1")).metric_name == "rmse"
    rmsle = decide_metric("Метрика: RMSLE", spec, None)
    assert rmsle.metric_name == "rmse" and "log1p" in rmsle.notes[0]


def test_roc_auc_detected_in_text_for_binary_classification(binary_classification_dataset):
    spec = infer_task_spec(
        binary_classification_dataset["train"], binary_classification_dataset["dict"], binary_classification_dataset["sub"]
    )
    assert guess_metric_from_text("Метрика оценки — ROC-AUC.") == "roc_auc"
    decision = decide_metric("Метрика оценки — ROC-AUC.", spec, None)
    assert decision.metric_name == "roc_auc" and decision.source == "task.md"
    assert decide_metric("Метрика: AUC.", spec, None).metric_name == "roc_auc"
    assert decide_metric("Метрика: log loss.", spec, None).metric_name == "log_loss"


def test_roc_auc_falls_back_to_default_on_a_multiclass_task(classification_dataset):
    """ml_core only supports a probability submission for binary classification (see
    ml_core/metrics.py); on a 3+-class task, text asking for ROC-AUC must not get accepted —
    that would make every single experiment come back status="unsupported" for the whole run,
    never actually falling back to a metric the task can be optimized on."""
    spec = infer_task_spec(classification_dataset["train"], classification_dataset["dict"], classification_dataset["sub"])
    decision = decide_metric("Метрика оценки — ROC-AUC.", spec, None)
    assert decision.metric_name == spec.metric_name  # the task-type default, not roc_auc
    assert decision.source == "default"
    assert any("doesn't fit it" in n for n in decision.notes)


def test_time_split_only_for_a_truly_later_period(tmp_path):
    train = pd.DataFrame({"id": range(8), "period": [1, 1, 2, 2, 3, 3, 4, 4], "noise_flag": [0, 255] * 4,
                          "x": range(8), "y": [1.0, 2, 3, 4, 5, 6, 7, 8]})
    test = pd.DataFrame({"id": range(8, 12), "period": [5, 5, 5, 5], "noise_flag": [7, 9, 11, 13], "x": range(4)})
    dd = pd.DataFrame({"feature": ["id", "period", "noise_flag", "x", "y"], "role": ["id", "feature", "feature", "feature", "target"],
                       "dtype": ["numeric"] * 5})
    for name, df in {"train": train, "test": test, "data_dictionary": dd}.items():
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    pd.DataFrame({"id": test["id"], "y": 0.0}).to_csv(tmp_path / "sample_submission.csv", index=False)

    profile = profile_data(tmp_path / "train.csv", tmp_path / "data_dictionary.csv", tmp_path / "test.csv")
    candidates = time_split_candidates(profile, tmp_path / "train.csv", tmp_path / "test.csv")
    assert candidates == ["period"]  # noise_flag is flagged too, but test values are not "later"
    spec = infer_task_spec(tmp_path / "train.csv", tmp_path / "data_dictionary.csv", tmp_path / "sample_submission.csv")
    vd = decide_validation(spec, profile, candidates, None)
    # 4 distinct periods in train -> walk-forward over min(3, 4-2)=2 folds (periods 4 and 3 held out,
    # each trained only on strictly earlier periods — see agent/understand.py's decide_validation).
    assert (vd.strategy, vd.time_column, vd.n_splits) == ("time_split", "period", 2)


def test_time_split_folds_scale_with_available_periods_and_are_capped(tmp_path):
    def build(n_periods, rows_per_period=4):
        periods = [p for p in range(1, n_periods + 1) for _ in range(rows_per_period)]
        n = len(periods)
        train = pd.DataFrame({"id": range(n), "period": periods, "x": range(n), "y": range(n)})
        test = pd.DataFrame({"id": range(n, n + 4), "period": [n_periods + 1] * 4, "x": range(4)})
        dd = pd.DataFrame({"feature": ["id", "period", "x", "y"], "role": ["id", "feature", "feature", "target"],
                           "dtype": ["numeric"] * 4})
        for name, df in {"train": train, "test": test, "data_dictionary": dd}.items():
            df.to_csv(tmp_path / f"{name}_{n_periods}.csv", index=False)
        pd.DataFrame({"id": test["id"], "y": 0.0}).to_csv(tmp_path / f"sample_submission_{n_periods}.csv", index=False)
        profile = profile_data(tmp_path / f"train_{n_periods}.csv", tmp_path / f"data_dictionary_{n_periods}.csv", tmp_path / f"test_{n_periods}.csv")
        candidates = time_split_candidates(profile, tmp_path / f"train_{n_periods}.csv", tmp_path / f"test_{n_periods}.csv")
        spec = infer_task_spec(tmp_path / f"train_{n_periods}.csv", tmp_path / f"data_dictionary_{n_periods}.csv", tmp_path / f"sample_submission_{n_periods}.csv")
        return decide_validation(spec, profile, candidates, None)

    assert build(2).n_splits == 1  # too few periods for more than one fold
    assert build(3).n_splits == 1
    assert build(5).n_splits == 3  # the real building-energy task's shape (Year_Factor: 5 train periods)
    assert build(10).n_splits == 3  # capped, not "more periods -> unboundedly more folds"


def test_locate_files_platform_layout_and_flat_layout(tmp_path):
    platform = tmp_path / "platform"
    data = platform / "regression" / "participants"
    data.mkdir(parents=True)
    for name in ("train", "test", "data_dictionary", "sample_submission"):
        (data / f"{name}.csv").write_text("id\n1\n")
    (platform / "task.md").write_text("- `regression/participants/train.csv` — rows\n", encoding="utf-8")
    files = locate_task_files(platform / "task.md")
    assert files.train == (data / "train.csv").resolve() and files.test.parent == data.resolve()

    flat = locate_task_files(DATASETS / "Регрессия" / "task.md")  # paths in this task.md do not exist here
    assert flat.train.name == "train.csv" and flat.train.parent == (DATASETS / "Регрессия").resolve()


def test_feature_code_static_and_dynamic_checks(tmp_path):
    assert any("os" in p for p in static_check("import os\ndef make_features(df):\n    return df"))
    assert any("open" in p for p in static_check("def make_features(df):\n    open('x')\n    return df"))
    assert any("make_features" in p for p in static_check("x = 1"))

    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0] * 50, "b": ["x", "y", "x", "z"] * 50})
    sandbox = FeatureSandbox(tmp_path, train, train.head(40))
    good = sandbox.check("import numpy as np\ndef make_features(df):\n    df['a2'] = df['a'] ** 2\n    return df\n")
    assert good.ok and good.new_columns == ["a2"] and good.step.startswith("py:")
    random_code = "import numpy as np\ndef make_features(df):\n    df['r'] = np.random.rand(len(df))\n    return df\n"
    assert any("deterministic" in p for p in sandbox.check(random_code).problems)
    shrink = "def make_features(df):\n    return df.head(3)\n"
    assert any("rows" in p for p in sandbox.check(shrink).problems)
    const = "def make_features(df):\n    df['c'] = 1\n    return df\n"
    assert any("constant" in p for p in sandbox.check(const).problems)


def test_local_search_moves_stay_in_bounds_and_build_valid_configs():
    import random

    from agent.hypotheses import local_search_hypotheses

    best = ExperimentConfig(model_name="catboost", hyperparameters={"iterations": 2900, "depth": 10}, validation_id="v")
    hyps = local_search_hypotheses(best, 20, random.Random(1))
    assert hyps
    saw_max_ctr_complexity = False
    for h in hyps:
        cfg, _ = materialize(h, best, "v")
        assert cfg.model_name == "catboost"
        assert cfg.hyperparameters["iterations"] <= 3000 and 3 <= cfg.hyperparameters["depth"] <= 10
        if "max_ctr_complexity" in h.hyperparameters:
            saw_max_ctr_complexity = True
            assert 1 <= cfg.hyperparameters["max_ctr_complexity"] <= 4
    assert saw_max_ctr_complexity  # local-search can discover this real speed/quality lever on its own
    assert local_search_hypotheses(ExperimentConfig(model_name="ridge", validation_id="v"), 3, random.Random(0)) == []


def test_materialize_drops_parameters_the_model_does_not_take():
    # both seen from Qwen on the regression task: classifier-only weights and LightGBM params on CatBoost
    hyp = Hypothesis(title="t", model_name="catboost",
                     hyperparameters={"depth": 6, "auto_class_weights": "Balanced", "colsample_bytree": 0.7, "subsample_freq": 1})
    config, notes = materialize(hyp, None, "v", task_type="regression")
    assert config.hyperparameters == {"depth": 6, "allow_writing_files": False}
    assert sum("dropped" in n for n in notes) == 3

    config, _ = materialize(Hypothesis(title="t", model_name="catboost", hyperparameters={"auto_class_weights": None}),
                            None, "v", task_type="classification")
    assert "auto_class_weights" in config.hyperparameters  # valid for a classifier: kept
