"""Out-of-fold mean target encoding for high-cardinality categoricals.

Deliberately NOT part of ml_core/feature_steps.py's FeatureStep framework: that module's
steps receive feature columns only, never the target, specifically so a step (including
LLM-generated py: code, sandboxed but still untrusted) cannot leak the label even if written
carelessly (see its module docstring). Target encoding needs the target by definition, so
it is a separate, hand-reviewed mechanism instead, called directly from training.py's CV loop
and prediction.py's refit — both already have y safely in scope — before/alongside
apply_feature_steps(), never through it.

Leak-safety: no row's own target ever informs its own encoded value.
- The portion being fit on (one CV fold's train rows, or the full train set for the final
  refit) gets an out-of-fold encoding: an inner K-fold split computes each row's encoded
  value from only the OTHER inner folds of that same portion.
- Any other frame (the fold's held-out val rows, or the real test set) gets a plain encoding
  fit on the WHOLE fit-on portion — safe, since those rows' own targets are never touched.

Regression: one encoded column per categorical, <col>__target_enc (mean of the continuous
target). Classification: one column per class, <col>__target_enc_c{k} (mean of the 0/1
one-vs-rest indicator for class k) — the categorical's relationship to the target is rarely
expressible as one number when the classes aren't ordered, so this keeps the per-class signal
instead of collapsing it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ml_core.contracts import TaskType
from ml_core.preprocessing import FeatureColumns

STEP_NAME = "target_encode"


def _oof_single_target(
    cat: pd.Series, y: np.ndarray, other_cats: list[pd.Series], n_inner_folds: int, smoothing: float, seed: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Core OOF mean-encoding for one target vector (a continuous y, or a 0/1 one-vs-rest
    indicator). Smoothing pulls a category's encoding toward the global mean in proportion to
    smoothing / (category's row count + smoothing), so a rare category's encoding doesn't just
    overfit to the handful of rows that happen to define it."""
    n = len(cat)
    global_mean = float(np.mean(y))

    def blended(stats: pd.DataFrame) -> pd.Series:
        return (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)

    full_stats = pd.DataFrame({"cat": cat.to_numpy(), "y": y}).groupby("cat")["y"].agg(["mean", "count"])
    full_enc = blended(full_stats)

    oof = np.full(n, global_mean, dtype=float)
    if n_inner_folds >= 2 and n >= n_inner_folds:
        kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
        for inner_tr, inner_val in kf.split(np.arange(n)):
            inner_stats = pd.DataFrame(
                {"cat": cat.iloc[inner_tr].to_numpy(), "y": y[inner_tr]}
            ).groupby("cat")["y"].agg(["mean", "count"])
            inner_enc = blended(inner_stats)
            oof[inner_val] = cat.iloc[inner_val].map(inner_enc).fillna(global_mean).to_numpy()
    else:  # too few rows for an inner split (e.g. a tiny test fixture) — fall back to the plain encoding
        oof = cat.map(full_enc).fillna(global_mean).to_numpy()

    others_enc = [oc.map(full_enc).fillna(global_mean).to_numpy() for oc in other_cats]
    return oof, others_enc


def out_of_fold_target_encode(
    fit_df: pd.DataFrame,
    y_fit: np.ndarray,
    other_dfs: list[pd.DataFrame],
    columns: list[str],
    task_type: TaskType,
    n_classes: int = 0,
    n_inner_folds: int = 5,
    smoothing: float = 10.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Adds target-encoded column(s) for each name in `columns` that is present in fit_df
    (names already dropped by an earlier feature step are silently skipped, same as
    ml_core.feature_steps.CountEncode). Returns (fit_df_with_new_columns, [other_dfs_with_new_columns]),
    original columns untouched — callers keep using the same (numeric + categorical) column set
    they already had, plus these new numeric ones."""
    present = [c for c in columns if c in fit_df.columns]
    fit_out, others_out = fit_df.copy(), [df.copy() for df in other_dfs]
    if not present:
        return fit_out, others_out

    if task_type == "regression":
        targets = {"": np.asarray(y_fit, dtype=float)}
    else:
        y_int = np.asarray(y_fit, dtype=int)
        targets = {f"_c{k}": (y_int == k).astype(float) for k in range(n_classes)}

    for col in present:
        cat = fit_df[col].astype(str)
        other_cats = [(df[col].astype(str) if col in df.columns else pd.Series([""] * len(df))) for df in other_dfs]
        for suffix, y in targets.items():
            oof, others_enc = _oof_single_target(cat, y, other_cats, n_inner_folds, smoothing, seed)
            fit_out[f"{col}__target_enc{suffix}"] = oof
            for i, enc in enumerate(others_enc):
                others_out[i][f"{col}__target_enc{suffix}"] = enc

    return fit_out, others_out


def maybe_apply_target_encoding(
    feature_steps: list[str],
    fit_df: pd.DataFrame,
    y_fit: np.ndarray,
    other_dfs: list[pd.DataFrame],
    feature_cols: FeatureColumns,
    task_type: TaskType,
    n_classes: int,
    seed: int,
) -> tuple[list[str], pd.DataFrame, list[pd.DataFrame], FeatureColumns]:
    """If STEP_NAME ("target_encode") is in feature_steps, applies out_of_fold_target_encode()
    to every current categorical column and returns the remaining steps (with "target_encode"
    removed, since ml_core.feature_steps.apply_feature_steps doesn't know this name) plus the
    updated frames/FeatureColumns; otherwise a no-op passthrough. Call this once, before
    apply_feature_steps(), at any of this project's fit sites (training.py's single-config and
    ensemble-component fold loops, prediction.py's matching refit paths) — all four do the
    same two calls in the same order, so this is the one place that wiring is written."""
    if STEP_NAME not in feature_steps:
        return feature_steps, fit_df, other_dfs, feature_cols
    remaining = [s for s in feature_steps if s != STEP_NAME]
    new_fit, new_others = out_of_fold_target_encode(
        fit_df, y_fit, other_dfs, feature_cols.categorical, task_type, n_classes, seed=seed
    )
    new_cols = [c for c in new_fit.columns if c not in fit_df.columns]
    updated_cols = FeatureColumns(
        numeric=feature_cols.numeric + new_cols,
        categorical=feature_cols.categorical,
        low_card_categorical=feature_cols.low_card_categorical,
        high_card_categorical=feature_cols.high_card_categorical,
    )
    return remaining, new_fit, new_others, updated_cols
