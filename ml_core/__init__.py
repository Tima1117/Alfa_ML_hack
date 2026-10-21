"""Reusable ML backend: profile data, build validation, run experiments, predict.

Typical sequence (see examples/ for full runnable scripts):

    from ml_core import contracts, profiling, validation, training, prediction

    task_spec = profiling.infer_task_spec(train_path, dict_path, sample_sub_path)
    data_profile = profiling.profile_data(train_path, dict_path, test_path)
    val_plan = validation.create_validation(task_spec, train_path, out_dir="runs/val")
    cfg = contracts.ExperimentConfig(model_name="catboost", validation_id=val_plan.validation_id)
    result = training.run_experiment(task_spec, cfg, val_plan, data_profile, train_path, "runs/experiments")
    prediction.refit_and_predict(task_spec, cfg, data_profile, train_path, test_path, "submission.csv")
"""

from ml_core import (  # noqa: F401
    contracts,
    experiment_log,
    errors,
    feature_steps,
    metrics,
    model_registry,
    prediction,
    preprocessing,
    profiling,
    training,
    validation,
)

__all__ = [
    "contracts",
    "errors",
    "experiment_log",
    "feature_steps",
    "metrics",
    "model_registry",
    "prediction",
    "preprocessing",
    "profiling",
    "training",
    "validation",
]
