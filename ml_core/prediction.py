"""refit_and_predict(): train the chosen config on full train, predict test,
write a checked, atomic submission.csv.

write_fallback_submission() is the safety net for the hard 3-hour kill: call
it once, early, with the cheapest possible config so a *valid* submission.csv
exists on disk no matter what happens afterwards, then let the research loop
overwrite it with something better via further refit_and_predict() calls.
Checks run before the file is written, never after — a failing submission is
never written over a previous, already-checked-good one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_core import model_registry
from ml_core.contracts import DataProfile, ExperimentConfig, PredictionResult, TaskSpec
from ml_core.feature_steps import apply_feature_steps
from ml_core.io_utils import atomic_write_csv
from ml_core.preprocessing import (
    CategoryVocabulary,
    build_linear_preprocessor,
    forward_target_transform,
    inverse_target_transform,
    prepare_native_frame,
    split_feature_columns,
)


def refit_and_predict(
    task_spec: TaskSpec,
    experiment_config: ExperimentConfig,
    data_profile: DataProfile,
    train_path: str | Path,
    test_path: str | Path,
    out_path: str | Path,
) -> PredictionResult:
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    except Exception as exc:
        return PredictionResult(status="failed", output_path=str(out_path), error_message=f"{type(exc).__name__}: {exc}")

    if (
        model_registry.requires_linear_mode(experiment_config.model_name)
        and experiment_config.preprocessing.mode == "native"
    ):
        return PredictionResult(
            status="failed",
            output_path=str(out_path),
            error_message=(
                f"model_name={experiment_config.model_name!r} needs preprocessing.mode='linear' "
                "(it cannot be fit on raw NaN / pandas-category input) — got mode='native'."
            ),
        )

    feature_cols = split_feature_columns(data_profile.columns, experiment_config.preprocessing)

    try:
        if task_spec.task_type == "classification":
            class_labels = task_spec.class_labels or []
            label_to_int = {label: i for i, label in enumerate(class_labels)}
            unknown = set(train[task_spec.target_column].astype(str).unique()) - set(label_to_int)
            if unknown:
                raise ValueError(f"train target has labels outside TaskSpec.class_labels: {unknown}")
            y = train[task_spec.target_column].astype(str).map(label_to_int).to_numpy()
        else:
            y = train[task_spec.target_column].to_numpy(dtype=float)

        train_features, test_features = train, test
        if experiment_config.feature_steps:
            train_features, (test_features,), feature_cols = apply_feature_steps(
                experiment_config.feature_steps,
                train,
                [test],
                feature_cols,
                experiment_config.preprocessing.high_cardinality_threshold,
            )

        mode = experiment_config.preprocessing.mode
        if mode == "native":
            vocab = CategoryVocabulary().fit(train_features, feature_cols.categorical)
            X_train = prepare_native_frame(train_features, feature_cols, vocab, experiment_config.model_name)
            X_test = prepare_native_frame(test_features, feature_cols, vocab, experiment_config.model_name)
            cat_names = feature_cols.categorical
        else:
            preproc = build_linear_preprocessor(feature_cols, experiment_config.preprocessing)
            X_train = preproc.fit_transform(train_features)
            X_test = preproc.transform(test_features)
            cat_names = None

        model = model_registry.build_model(
            experiment_config.model_name,
            task_spec.task_type,
            experiment_config.hyperparameters,
            experiment_config.random_seed,
            experiment_config.n_jobs,
            cat_feature_names=cat_names,
        )

        if task_spec.task_type == "regression":
            y_fit = forward_target_transform(y, experiment_config.preprocessing.target_transform)
            model.fit(X_train, y_fit)
            raw_pred = model.predict(X_test)
            predictions = inverse_target_transform(raw_pred, experiment_config.preprocessing.target_transform)
        else:
            model.fit(X_train, y)
            pred_int = model.predict(X_test)
            class_labels = task_spec.class_labels or []
            predictions = np.array([class_labels[int(i)] for i in pred_int], dtype=object)
    except Exception as exc:
        return PredictionResult(status="failed", output_path=str(out_path), error_message=f"{type(exc).__name__}: {exc}")

    id_col = task_spec.id_columns[0]
    try:
        submission = pd.DataFrame({id_col: test[id_col].to_numpy()})
        submission[task_spec.target_column] = predictions
        submission = submission[task_spec.prediction_requirements.output_columns]
    except Exception as exc:
        return PredictionResult(
            status="failed",
            output_path=str(out_path),
            error_message=f"could not assemble output_columns {task_spec.prediction_requirements.output_columns}: {type(exc).__name__}: {exc}",
        )

    checks, warnings = _check_submission(submission, task_spec, test, id_col)
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        return PredictionResult(
            status="failed",
            output_path=str(out_path),
            n_rows=len(submission),
            checks_passed=checks,
            warnings=warnings,
            error_message=f"submission failed checks: {failed}",
        )

    atomic_write_csv(submission, out_path)
    return PredictionResult(
        status="success",
        output_path=str(out_path),
        n_rows=len(submission),
        checks_passed=checks,
        warnings=warnings,
    )


def _check_submission(
    submission: pd.DataFrame, task_spec: TaskSpec, test: pd.DataFrame, id_col: str
) -> tuple[dict[str, bool], list[str]]:
    warnings: list[str] = []
    row_count_matches = len(submission) == len(test)
    ids_match = (
        bool((submission[id_col].to_numpy() == test[id_col].to_numpy()).all()) if row_count_matches else False
    )
    checks = {
        "row_count_matches_test": row_count_matches,
        "columns_match_requirements": list(submission.columns) == task_spec.prediction_requirements.output_columns,
        "ids_match_test_order": ids_match,
        "no_nan": task_spec.prediction_requirements.allow_nan or not bool(submission.isna().any().any()),
    }

    target_col = task_spec.target_column
    if task_spec.task_type == "regression":
        values = submission[target_col].to_numpy(dtype=float)
        checks["no_inf"] = task_spec.prediction_requirements.allow_inf or bool(np.isfinite(values).all())
    else:
        allowed = set(task_spec.class_labels or [])
        predicted = set(submission[target_col].astype(str).unique())
        checks["labels_within_allowed_set"] = predicted.issubset(allowed)
        if not checks["labels_within_allowed_set"]:
            warnings.append(f"unexpected labels in predictions: {predicted - allowed}")

    if not bool(test[id_col].is_unique):
        warnings.append("test.csv id column is not unique — not an output error, but worth flagging upstream")

    return checks, warnings


def write_fallback_submission(
    task_spec: TaskSpec,
    data_profile: DataProfile,
    train_path: str | Path,
    test_path: str | Path,
    out_path: str | Path,
    validation_id: str = "fallback",
) -> PredictionResult:
    """Cheapest possible valid submission (median/most-frequent), written
    before spending any real time budget — guarantees a scorable file exists
    even if the process is killed later without finishing a better model."""
    from ml_core.contracts import PreprocessingConfig

    fallback_config = ExperimentConfig(
        model_name="baseline",
        hyperparameters={},
        preprocessing=PreprocessingConfig(mode="native"),
        validation_id=validation_id,
    )
    return refit_and_predict(task_spec, fallback_config, data_profile, train_path, test_path, out_path)
