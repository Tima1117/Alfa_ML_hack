import time

import pandas as pd

from agent.executor import ExperimentContext, SubprocessExecutor
from ml_core import experiment_log
from ml_core.contracts import ExperimentConfig
from ml_core.profiling import infer_task_spec, profile_data
from ml_core.validation import create_validation


def _ctx(paths, tmp_path):
    spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    profile = profile_data(paths["train"], paths["dict"], paths["test"])
    plan = create_validation(spec, paths["train"], tmp_path / "val", n_splits=3, seed=0)
    return ExperimentContext(spec, profile, plan, paths["train"], paths["test"], tmp_path / "experiments")


def test_subprocess_executor_runs_and_refits(regression_dataset, tmp_path):
    ctx = _ctx(regression_dataset, tmp_path)
    cfg = ExperimentConfig(model_name="hist_gradient_boosting", validation_id=ctx.validation_plan.validation_id, n_jobs=1)
    result = SubprocessExecutor().run_experiment(ctx, cfg, timeout_s=120)
    assert result.status == "success" and result.n_folds_completed == 3
    assert experiment_log.select_best(ctx.experiments_dir).experiment_id == result.experiment_id

    out = tmp_path / "sub.csv"
    pred = SubprocessExecutor().refit_and_predict(ctx, cfg, out, timeout_s=120)
    assert pred.status == "success" and len(pd.read_csv(out)) == len(pd.read_csv(regression_dataset["test"]))


def test_subprocess_executor_kills_a_hanging_experiment(regression_dataset, tmp_path):
    ctx = _ctx(regression_dataset, tmp_path)
    hang = tmp_path / "hang.py"
    hang.write_text("import time\ndef make_features(df):\n    time.sleep(600)\n    return df\n")
    cfg = ExperimentConfig(
        model_name="baseline", validation_id=ctx.validation_plan.validation_id, feature_steps=[f"py:{hang}"],
        experiment_id="hang",
    )
    start = time.monotonic()
    result = SubprocessExecutor(grace_s=1).run_experiment(ctx, cfg, timeout_s=5)
    assert time.monotonic() - start < 60
    assert result.status == "timeout" and "hard timeout" in result.error_message
    logged = experiment_log.load_all(ctx.experiments_dir)
    assert [r.status for r in logged] == ["timeout"]
