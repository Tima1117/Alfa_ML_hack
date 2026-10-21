"""Model registry.

Scope is deliberately small: one reference baseline, one linear model, and
the tree ensembles that fit the two profiled tasks (both are tabular,
moderate size, with high-cardinality + unseen-at-test-time categoricals,
which is exactly CatBoost/LightGBM/XGBoost/HistGradientBoosting territory).

autogluon / lightautoml / flaml are installed per requirements.txt but are
NOT wrapped here: they run their own internal validation and ensembling,
which would fight with this module's own validation/leakage controls rather
than compose with them. They're a reasonable follow-up, not a v1 default —
see docs/CONTRACTS.md "Next improvements".

Neural nets are out of scope for the same reason the task brief asks for:
no justification to start there when gradient boosting is the well-tested
default for tabular data at this size.

CatBoost/LightGBM/XGBoost are only ever imported inside the functions below
(never at module import time), so importing model_registry never fails even
if one of them isn't installed in a given environment — get_model() raises
a clear, catchable error instead, which training.run_experiment turns into
an ExperimentResult(status="unsupported") rather than crashing the caller.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ml_core.contracts import TaskType
from ml_core.errors import UnsupportedTaskTypeError

NATIVE_MODELS = {"catboost", "lightgbm", "xgboost", "hist_gradient_boosting"}
LINEAR_MODELS = {"baseline", "ridge", "logistic_regression", "random_forest"}
ALL_MODELS = NATIVE_MODELS | LINEAR_MODELS

# ridge/logistic_regression/random_forest need a fully numeric, NaN-free
# matrix — they cannot be fit on native mode's raw NaN + pandas-category
# frames the way catboost/lightgbm/xgboost/hist_gradient_boosting can.
# "baseline" (Dummy*) ignores X entirely, so either mode works for it.
_REQUIRES_LINEAR_MODE = {"ridge", "logistic_regression", "random_forest"}


def suggested_mode(model_name: str) -> str:
    return "native" if model_name in NATIVE_MODELS else "linear"


def requires_linear_mode(model_name: str) -> bool:
    return model_name in _REQUIRES_LINEAR_MODE


def is_available(model_name: str) -> bool:
    if model_name in {"baseline", "ridge", "logistic_regression", "random_forest", "hist_gradient_boosting"}:
        return True  # ships with scikit-learn, always in requirements.txt
    return importlib.util.find_spec(model_name) is not None


def build_model(
    model_name: str,
    task_type: TaskType,
    hyperparameters: dict[str, Any],
    random_seed: int,
    n_jobs: int | None = None,
    cat_feature_names: list[str] | None = None,
):
    """Return an unfitted, sklearn-compatible estimator.

    Raises UnsupportedTaskTypeError (caught by training.run_experiment, never
    propagated to the caller of run_experiment itself) if the model name is
    unknown or its library isn't importable.
    """
    if model_name not in ALL_MODELS:
        raise UnsupportedTaskTypeError(
            f"Unknown model_name {model_name!r}. Known models: {sorted(ALL_MODELS)}"
        )
    if not is_available(model_name):
        raise UnsupportedTaskTypeError(
            f"model_name {model_name!r} is not importable in this environment."
        )

    params = dict(hyperparameters)
    n_jobs = n_jobs if n_jobs is not None else -1

    if model_name == "baseline":
        from sklearn.dummy import DummyClassifier, DummyRegressor

        if task_type == "regression":
            params.setdefault("strategy", "median")
            return DummyRegressor(**params)
        params.setdefault("strategy", "most_frequent")
        return DummyClassifier(**params)

    if model_name == "ridge":
        from sklearn.linear_model import Ridge

        params.setdefault("random_state", random_seed)
        return Ridge(**params)

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        params.setdefault("random_state", random_seed)
        params.setdefault("max_iter", 1000)
        params.setdefault("class_weight", "balanced")
        params.setdefault("n_jobs", n_jobs)
        return LogisticRegression(**params)

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        params.setdefault("random_state", random_seed)
        params.setdefault("n_jobs", n_jobs)
        params.setdefault("n_estimators", 300)
        if task_type == "regression":
            return RandomForestRegressor(**params)
        params.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**params)

    if model_name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        params.setdefault("random_state", random_seed)
        params.setdefault("categorical_features", "from_dtype")
        if task_type == "regression":
            return HistGradientBoostingRegressor(**params)
        params.setdefault("class_weight", "balanced")
        return HistGradientBoostingClassifier(**params)

    if model_name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        params.setdefault("random_seed", random_seed)
        params.setdefault("verbose", False)
        params.setdefault("thread_count", n_jobs)
        params.setdefault("cat_features", cat_feature_names or [])
        if task_type == "regression":
            return CatBoostRegressor(**params)
        params.setdefault("auto_class_weights", "Balanced")
        return CatBoostClassifier(**params)

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        params.setdefault("random_state", random_seed)
        params.setdefault("n_jobs", n_jobs)
        params.setdefault("verbosity", -1)
        if task_type == "regression":
            return LGBMRegressor(**params)
        params.setdefault("class_weight", "balanced")
        return LGBMClassifier(**params)

    if model_name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        params.setdefault("random_state", random_seed)
        params.setdefault("n_jobs", n_jobs)
        params.setdefault("enable_categorical", True)
        params.setdefault("tree_method", "hist")
        if task_type == "regression":
            return XGBRegressor(**params)
        return XGBClassifier(**params)

    raise UnsupportedTaskTypeError(f"model_name {model_name!r} has no builder implemented")


def positive_class_proba(model, X) -> "np.ndarray":
    """P(label == 1) for a fitted binary classifier, used when the task's metric needs a
    probability (roc_auc, log_loss) rather than a hard prediction.

    Looks up the positive class's column in predict_proba's output by its position in
    model.classes_ rather than assuming it's always column 1 — sklearn-compatible
    estimators sort fitted classes, which happens to put binary {0, 1} labels in order,
    but asserting it directly is cheap insurance against a model that doesn't.
    """
    import numpy as np

    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"fitted model's classes_ {classes} has no label '1' to read a positive-class probability from")
    return np.asarray(proba)[:, classes.index(1)]
