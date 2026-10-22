"""run_experiment(): fit + honestly evaluate one ExperimentConfig.

Never raises for an agent-driven failure (bad model name, a library missing
from this environment, a fit that throws, a config mismatched with its
ValidationPlan) — every one of those comes back as an ExperimentResult with
status in {"failed", "unsupported", "timeout"} and a short error_message, so
a calling agent can test a bad hypothesis without wrapping every call in
try/except. It CAN still raise pydantic.ValidationError, but only while
*constructing* an ExperimentConfig/TaskSpec, before run_experiment is ever
called — that's deliberate fail-fast validation of the config shape itself.

Time budget is soft: checked once per fold, not inside a fold's .fit() call.
A fold whose own fit() hangs or runs long cannot be interrupted from here —
if that risk matters for a given model/hyperparameter combination, the
infrastructure layer needs to run run_experiment in a subprocess with an OS
-level timeout. See docs/CONTRACTS.md.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml_core import model_registry
from ml_core.model_registry import positive_class_proba
from ml_core.contracts import DataProfile, ExperimentConfig, ExperimentResult, TaskSpec, ValidationPlan
from ml_core.errors import UnsupportedTaskTypeError
from ml_core.experiment_log import record as log_record
from ml_core.feature_steps import apply_feature_steps, parse_steps
from ml_core.target_encoding import STEP_NAME as TARGET_ENCODE_STEP
from ml_core.target_encoding import maybe_apply_target_encoding
from ml_core.io_utils import config_hash
from ml_core.metrics import build_task_metric_fn, check_proba_metric_supported, metric_needs_proba
from ml_core.preprocessing import (
    CategoryVocabulary,
    build_linear_preprocessor,
    forward_target_transform,
    inverse_target_transform,
    prepare_native_frame,
    split_feature_columns,
)
from ml_core.validation import load_fold_indices


def _encode_labels(y: pd.Series, class_labels: list[str]) -> np.ndarray:
    label_to_int = {label: i for i, label in enumerate(class_labels)}
    unknown = set(y.astype(str).unique()) - set(label_to_int)
    if unknown:
        raise ValueError(f"train target contains labels not in TaskSpec.class_labels: {unknown}")
    return y.astype(str).map(label_to_int).to_numpy()


def run_experiment(
    task_spec: TaskSpec,
    experiment_config: ExperimentConfig,
    validation_plan: ValidationPlan,
    data_profile: DataProfile,
    train_path: str | Path,
    experiments_dir: str | Path | None = None,
) -> ExperimentResult:
    if experiment_config.ensemble_of:
        return _run_ensemble_experiment(
            task_spec, experiment_config, validation_plan, data_profile, train_path, experiments_dir
        )
    start = time.monotonic()
    experiment_id = experiment_config.experiment_id or config_hash(
        experiment_config.model_dump(mode="json", exclude={"experiment_id"})
    )
    n_requested = validation_plan.n_splits

    def _result(status: str, **kwargs) -> ExperimentResult:
        res = ExperimentResult(
            status=status,  # type: ignore[arg-type]
            experiment_id=experiment_id,
            metric_name=task_spec.metric_name,
            greater_is_better=task_spec.greater_is_better,
            validation_id=validation_plan.validation_id,
            n_folds_requested=n_requested,
            duration_s=time.monotonic() - start,
            config=experiment_config,
            **kwargs,
        )
        if experiments_dir is not None:
            log_record(res, experiments_dir)
        return res

    if experiment_config.validation_id != validation_plan.validation_id:
        return _result(
            "failed",
            error_message=(
                f"experiment_config.validation_id={experiment_config.validation_id!r} does not match "
                f"validation_plan.validation_id={validation_plan.validation_id!r}"
            ),
        )

    try:
        metric_fn = build_task_metric_fn(task_spec)
    except Exception as exc:  # metric/direction contract mismatch
        return _result("failed", error_message=f"{type(exc).__name__}: {exc}")

    try:
        check_proba_metric_supported(task_spec)
    except Exception as exc:  # metric needs probabilities on a shape we don't support (e.g. 3+ classes)
        return _result("unsupported", error_message=f"{type(exc).__name__}: {exc}")
    needs_proba = metric_needs_proba(task_spec)

    # Fail fast (before touching data) if the model/library isn't usable here.
    try:
        model_registry.build_model(
            experiment_config.model_name,
            task_spec.task_type,
            experiment_config.hyperparameters,
            experiment_config.random_seed,
            experiment_config.n_jobs,
        )
    except UnsupportedTaskTypeError as exc:
        return _result("unsupported", error_message=str(exc))
    except Exception as exc:
        return _result("failed", error_message=f"model construction failed: {type(exc).__name__}: {exc}")

    if (
        model_registry.requires_linear_mode(experiment_config.model_name)
        and experiment_config.preprocessing.mode == "native"
    ):
        return _result(
            "failed",
            error_message=(
                f"model_name={experiment_config.model_name!r} needs preprocessing.mode='linear' "
                "(it cannot be fit on raw NaN / pandas-category input) — got mode='native'."
            ),
        )

    try:
        # "target_encode" is intercepted by maybe_apply_target_encoding() before the remaining
        # steps ever reach apply_feature_steps()/parse_steps() — see ml_core/target_encoding.py's
        # module docstring for why it isn't a regular FeatureStep.
        parse_steps([s for s in experiment_config.feature_steps if s != TARGET_ENCODE_STEP])
    except Exception as exc:
        return _result("failed", error_message=f"feature_steps: {type(exc).__name__}: {exc}")

    try:
        folds = load_fold_indices(validation_plan, train_path)
        df = pd.read_csv(train_path)
    except Exception as exc:
        return _result("failed", error_message=f"{type(exc).__name__}: {exc}")

    feature_cols = split_feature_columns(data_profile.columns, experiment_config.preprocessing)
    diagnostics: dict = {
        "n_rows": len(df),
        "n_numeric_features": len(feature_cols.numeric),
        "n_categorical_features": len(feature_cols.categorical),
        "n_high_cardinality_features": len(feature_cols.high_card_categorical),
        "fold_warnings": [],
    }

    try:
        if task_spec.task_type == "classification":
            y_all = _encode_labels(df[task_spec.target_column], task_spec.class_labels or [])
        else:
            y_all = df[task_spec.target_column].to_numpy(dtype=float)
    except Exception as exc:
        return _result("failed", error_message=f"target encoding failed: {type(exc).__name__}: {exc}")

    deadline = start + experiment_config.time_budget_s if experiment_config.time_budget_s else None
    fold_metrics: list[float] = []

    try:
        for i, fold in enumerate(folds):
            if deadline is not None and time.monotonic() > deadline:
                diagnostics["fold_warnings"].append(
                    f"stopped after {i}/{n_requested} folds: time_budget_s exceeded"
                )
                break

            train_idx, val_idx = fold.train_idx, fold.val_idx
            if len(train_idx) == 0 or len(val_idx) == 0:
                diagnostics["fold_warnings"].append(f"fold {i} has an empty train or val set; skipped")
                continue
            fold_train, fold_val = df.iloc[train_idx], df.iloc[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]

            remaining_steps, fold_train, (fold_val,), fold_cols = maybe_apply_target_encoding(
                experiment_config.feature_steps, fold_train, y_train, [fold_val], feature_cols,
                task_spec.task_type, len(task_spec.class_labels or []), experiment_config.random_seed,
            )
            if remaining_steps:
                fold_train, (fold_val,), fold_cols = apply_feature_steps(
                    remaining_steps,
                    fold_train,
                    [fold_val],
                    fold_cols,
                    experiment_config.preprocessing.high_cardinality_threshold,
                )
            if experiment_config.feature_steps:
                diagnostics["n_features_after_steps"] = len(fold_cols.numeric) + len(fold_cols.categorical)

            mode = experiment_config.preprocessing.mode
            if mode == "native":
                vocab = CategoryVocabulary().fit(fold_train, fold_cols.categorical)
                X_train = prepare_native_frame(fold_train, fold_cols, vocab, experiment_config.model_name)
                X_val = prepare_native_frame(fold_val, fold_cols, vocab, experiment_config.model_name)
                cat_names = fold_cols.categorical
            else:
                preproc = build_linear_preprocessor(fold_cols, experiment_config.preprocessing)
                X_train = preproc.fit_transform(fold_train)
                X_val = preproc.transform(fold_val)
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
                y_fit = forward_target_transform(y_train, experiment_config.preprocessing.target_transform)
                model.fit(X_train, y_fit)
                pred_val = model.predict(X_val)
                pred_val = inverse_target_transform(pred_val, experiment_config.preprocessing.target_transform)
            elif needs_proba:
                cw_strategy = experiment_config.hyperparameters.get("class_weight_strategy", "sqrt_balanced")
                sw = model_registry.classification_sample_weight(experiment_config.model_name, y_train, cw_strategy)
                model.fit(X_train, y_train, sample_weight=sw)
                pred_val = positive_class_proba(model, X_val)
            else:
                cw_strategy = experiment_config.hyperparameters.get("class_weight_strategy", "sqrt_balanced")
                sw = model_registry.classification_sample_weight(experiment_config.model_name, y_train, cw_strategy)
                model.fit(X_train, y_train, sample_weight=sw)
                pred_val = model.predict(X_val)

            fold_metrics.append(float(metric_fn(y_val, pred_val)))
    except Exception as exc:
        return _result(
            "failed",
            error_message=f"{type(exc).__name__}: {exc}",
            fold_metrics=[float(x) for x in fold_metrics],
            n_folds_completed=len(fold_metrics),
            diagnostics=diagnostics,
        )

    if not fold_metrics:
        return _result("timeout", diagnostics=diagnostics)

    return _result(
        "success",
        metric_value=float(np.mean(fold_metrics)),
        metric_std=float(np.std(fold_metrics)),
        fold_metrics=[float(x) for x in fold_metrics],
        n_folds_completed=len(fold_metrics),
        diagnostics=diagnostics,
    )


def _fit_component_fold(task_spec, c: ExperimentConfig, data_profile, fold_train, fold_val, y_train, needs_proba, n_classes):
    """One ensemble member's fit+predict on one fold. Returns an array shaped like a single
    model's own fold prediction (regression: raw values; needs_proba: positive-class proba;
    hard-label classification: full per-class probability matrix, so averaging stays a real
    soft vote instead of averaging incomparable integer class indices)."""
    feature_cols = split_feature_columns(data_profile.columns, c.preprocessing)
    remaining_steps, fold_train, (fold_val,), fold_cols = maybe_apply_target_encoding(
        c.feature_steps, fold_train, y_train, [fold_val], feature_cols, task_spec.task_type, n_classes, c.random_seed,
    )
    if remaining_steps:
        fold_train, (fold_val,), fold_cols = apply_feature_steps(
            remaining_steps, fold_train, [fold_val], fold_cols, c.preprocessing.high_cardinality_threshold
        )
    if c.preprocessing.mode == "native":
        vocab = CategoryVocabulary().fit(fold_train, fold_cols.categorical)
        X_train = prepare_native_frame(fold_train, fold_cols, vocab, c.model_name)
        X_val = prepare_native_frame(fold_val, fold_cols, vocab, c.model_name)
        cat_names = fold_cols.categorical
    else:
        preproc = build_linear_preprocessor(fold_cols, c.preprocessing)
        X_train = preproc.fit_transform(fold_train)
        X_val = preproc.transform(fold_val)
        cat_names = None

    model = model_registry.build_model(
        c.model_name, task_spec.task_type, c.hyperparameters, c.random_seed, c.n_jobs, cat_feature_names=cat_names
    )
    if task_spec.task_type == "regression":
        y_fit = forward_target_transform(y_train, c.preprocessing.target_transform)
        model.fit(X_train, y_fit)
        raw = model.predict(X_val)
        return inverse_target_transform(raw, c.preprocessing.target_transform)

    cw_strategy = c.hyperparameters.get("class_weight_strategy", "sqrt_balanced")
    sw = model_registry.classification_sample_weight(c.model_name, y_train, cw_strategy)
    model.fit(X_train, y_train, sample_weight=sw)
    if needs_proba:
        return positive_class_proba(model, X_val)
    proba = model.predict_proba(X_val)
    full = np.zeros((len(X_val), n_classes))
    for j, cls in enumerate(model.classes_):
        full[:, int(cls)] = proba[:, j]
    return full


def _run_ensemble_experiment(
    task_spec: TaskSpec,
    experiment_config: ExperimentConfig,
    validation_plan: ValidationPlan,
    data_profile: DataProfile,
    train_path: str | Path,
    experiments_dir: str | Path | None = None,
) -> ExperimentResult:
    """Leak-safe CV estimate of soft-voting (averaging) experiment_config.ensemble_of's
    predictions on the SAME folds as any single-model experiment — its metric_value is
    directly comparable to one, so agent/research.py's existing _better()/refit_best() need
    no ensemble-specific code at all; they just see another ExperimentResult."""
    start = time.monotonic()
    configs = experiment_config.ensemble_of or []
    experiment_id = experiment_config.experiment_id or config_hash(
        experiment_config.model_dump(mode="json", exclude={"experiment_id"})
    )
    n_requested = validation_plan.n_splits

    def _result(status: str, **kwargs) -> ExperimentResult:
        res = ExperimentResult(
            status=status,  # type: ignore[arg-type]
            experiment_id=experiment_id,
            metric_name=task_spec.metric_name,
            greater_is_better=task_spec.greater_is_better,
            validation_id=validation_plan.validation_id,
            n_folds_requested=n_requested,
            duration_s=time.monotonic() - start,
            config=experiment_config,
            **kwargs,
        )
        if experiments_dir is not None:
            log_record(res, experiments_dir)
        return res

    if len(configs) < 2:
        return _result("failed", error_message=f"ensemble_of needs at least 2 configs, got {len(configs)}")
    if any(c.ensemble_of for c in configs):
        return _result("failed", error_message="nested ensembles (ensemble_of an ensemble) are not supported")
    bad_vid = [c.validation_id for c in configs if c.validation_id != validation_plan.validation_id]
    if bad_vid:
        return _result("failed", error_message=f"component validation_id mismatch: {bad_vid}")

    try:
        metric_fn = build_task_metric_fn(task_spec)
    except Exception as exc:
        return _result("failed", error_message=f"{type(exc).__name__}: {exc}")
    try:
        check_proba_metric_supported(task_spec)
    except Exception as exc:
        return _result("unsupported", error_message=f"{type(exc).__name__}: {exc}")
    needs_proba = metric_needs_proba(task_spec)

    for c in configs:
        try:
            model_registry.build_model(c.model_name, task_spec.task_type, c.hyperparameters, c.random_seed, c.n_jobs)
        except UnsupportedTaskTypeError as exc:
            return _result("unsupported", error_message=f"{c.model_name}: {exc}")
        except Exception as exc:
            return _result("failed", error_message=f"{c.model_name} construction failed: {type(exc).__name__}: {exc}")
        if model_registry.requires_linear_mode(c.model_name) and c.preprocessing.mode == "native":
            return _result("failed", error_message=f"{c.model_name} needs preprocessing.mode='linear'")
        try:
            parse_steps([s for s in c.feature_steps if s != TARGET_ENCODE_STEP])
        except Exception as exc:
            return _result("failed", error_message=f"{c.model_name} feature_steps: {type(exc).__name__}: {exc}")

    try:
        folds = load_fold_indices(validation_plan, train_path)
        df = pd.read_csv(train_path)
    except Exception as exc:
        return _result("failed", error_message=f"{type(exc).__name__}: {exc}")

    n_classes = len(task_spec.class_labels or [])
    diagnostics: dict = {"n_rows": len(df), "component_models": [c.model_name for c in configs], "fold_warnings": []}

    try:
        if task_spec.task_type == "classification":
            y_all = _encode_labels(df[task_spec.target_column], task_spec.class_labels or [])
        else:
            y_all = df[task_spec.target_column].to_numpy(dtype=float)
    except Exception as exc:
        return _result("failed", error_message=f"target encoding failed: {type(exc).__name__}: {exc}")

    deadline = start + experiment_config.time_budget_s if experiment_config.time_budget_s else None
    fold_metrics: list[float] = []

    try:
        for i, fold in enumerate(folds):
            if deadline is not None and time.monotonic() > deadline:
                diagnostics["fold_warnings"].append(f"stopped after {i}/{n_requested} folds: time_budget_s exceeded")
                break

            train_idx, val_idx = fold.train_idx, fold.val_idx
            if len(train_idx) == 0 or len(val_idx) == 0:
                diagnostics["fold_warnings"].append(f"fold {i} has an empty train or val set; skipped")
                continue
            fold_train_full, fold_val_full = df.iloc[train_idx], df.iloc[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]

            component_preds = [
                _fit_component_fold(task_spec, c, data_profile, fold_train_full, fold_val_full, y_train, needs_proba, n_classes)
                for c in configs
            ]
            avg = np.mean(component_preds, axis=0)
            pred_val = avg if (task_spec.task_type == "regression" or needs_proba) else avg.argmax(axis=1)
            fold_metrics.append(float(metric_fn(y_val, pred_val)))
    except Exception as exc:
        return _result(
            "failed",
            error_message=f"{type(exc).__name__}: {exc}",
            fold_metrics=[float(x) for x in fold_metrics],
            n_folds_completed=len(fold_metrics),
            diagnostics=diagnostics,
        )

    if not fold_metrics:
        return _result("timeout", diagnostics=diagnostics)

    return _result(
        "success",
        metric_value=float(np.mean(fold_metrics)),
        metric_std=float(np.std(fold_metrics)),
        fold_metrics=[float(x) for x in fold_metrics],
        n_folds_completed=len(fold_metrics),
        diagnostics=diagnostics,
    )
