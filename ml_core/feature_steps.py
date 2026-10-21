"""Feature-engineering steps named by ExperimentConfig.feature_steps.

Steps are applied INSIDE the CV loop, exactly like every other fitted
transform in this package: `fit` only ever sees the training fold, and the
validation fold (or test, in refit_and_predict) is only ever `transform`ed.
Steps receive feature columns only — never the target, never the id — so a
step cannot leak the label even if it is written carelessly.

Supported step names:

- ``"drop:<column>"``         drop one feature column (e.g. a disguised period
                              id the agent decided not to train on).
- ``"missing_indicators"``    add ``<col>__isna`` (0/1) for every column that has
                              missing values in the training fold.
- ``"count_encode"``          add ``<col>__count`` (category frequency in the
                              training fold; unseen -> 0) for categorical columns.
- ``"row_nan_count"``         add ``__row_nan_count``, number of missing features per row.
- ``"py:<path/to/file.py>"``  a generated module defining a *stateless*
                              ``make_features(df: pd.DataFrame) -> pd.DataFrame``.
                              It receives a copy of the feature frame and returns
                              a frame with the same rows (same length, same order);
                              it may add or drop columns. Checking that such code
                              is safe/deterministic is the caller's job (the agent
                              runs it in a sandbox before ever naming it here).

New columns get their kind from their dtype after the step ran (numeric ->
numeric, anything else -> categorical), so the rest of preprocessing treats
them like any other column.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from ml_core.errors import FeatureStepError
from ml_core.preprocessing import FeatureColumns

PY_PREFIX = "py:"
DROP_PREFIX = "drop:"
BUILTIN_STEPS = ("missing_indicators", "count_encode", "row_nan_count")
_MAX_MISSING_INDICATORS = 50


class FeatureStep:
    def fit(self, df: pd.DataFrame) -> "FeatureStep":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError


class DropColumn(FeatureStep):
    def __init__(self, column: str) -> None:
        self.column = column

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[self.column], errors="ignore")


class MissingIndicators(FeatureStep):
    def fit(self, df: pd.DataFrame) -> "MissingIndicators":
        rates = df.isna().mean()
        self.columns_ = rates[rates > 0].sort_values(ascending=False).index[:_MAX_MISSING_INDICATORS].tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns_:
            if col in out.columns:
                out[f"{col}__isna"] = out[col].isna().astype(np.int8)
        return out


class CountEncode(FeatureStep):
    def fit(self, df: pd.DataFrame) -> "CountEncode":
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        self.counts_ = {c: df[c].astype(str).value_counts().to_dict() for c in cat_cols}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, counts in self.counts_.items():
            if col in out.columns:
                out[f"{col}__count"] = out[col].astype(str).map(counts).fillna(0).astype(float)
        return out


class RowNanCount(FeatureStep):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["__row_nan_count"] = df.isna().sum(axis=1).astype(float)
        return out


class PythonFileStep(FeatureStep):
    def __init__(self, path: str) -> None:
        self.path = path
        file = Path(path)
        if not file.is_file():
            raise FeatureStepError(f"feature step file not found: {path}")
        spec = importlib.util.spec_from_file_location(f"_feature_step_{abs(hash(path))}", file)
        if spec is None or spec.loader is None:
            raise FeatureStepError(f"cannot load feature step module: {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise FeatureStepError(f"feature step {path} failed to import: {type(exc).__name__}: {exc}") from exc
        fn = getattr(module, "make_features", None)
        if not callable(fn):
            raise FeatureStepError(f"feature step {path} does not define make_features(df)")
        self._fn = fn

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._fn(df.copy())
        if not isinstance(out, pd.DataFrame):
            raise FeatureStepError(f"make_features in {self.path} returned {type(out).__name__}, not a DataFrame")
        if len(out) != len(df):
            raise FeatureStepError(f"make_features in {self.path} changed the row count ({len(df)} -> {len(out)})")
        out = out.copy()
        out.index = df.index
        return out.replace([np.inf, -np.inf], np.nan)


def parse_steps(names: list[str]) -> list[FeatureStep]:
    """Build fresh (unfitted) step objects; raises FeatureStepError on an unknown name."""
    steps: list[FeatureStep] = []
    for name in names:
        if name.startswith(DROP_PREFIX):
            steps.append(DropColumn(name[len(DROP_PREFIX):]))
        elif name.startswith(PY_PREFIX):
            steps.append(PythonFileStep(name[len(PY_PREFIX):]))
        elif name == "missing_indicators":
            steps.append(MissingIndicators())
        elif name == "count_encode":
            steps.append(CountEncode())
        elif name == "row_nan_count":
            steps.append(RowNanCount())
        else:
            raise FeatureStepError(
                f"unknown feature step {name!r}; known: {list(BUILTIN_STEPS)}, '{DROP_PREFIX}<column>', "
                f"'{PY_PREFIX}<path>'"
            )
    return steps


def apply_feature_steps(
    step_names: list[str],
    fit_df: pd.DataFrame,
    other_dfs: list[pd.DataFrame],
    feature_cols: FeatureColumns,
    high_cardinality_threshold: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame], FeatureColumns]:
    """Fit steps on `fit_df` (training rows only), transform every frame.

    Returns feature-only frames plus the FeatureColumns describing them.
    """
    cols = feature_cols.numeric + feature_cols.categorical
    fit_out = fit_df[cols].copy()
    others_out = [df[cols].copy() for df in other_dfs]
    for step in parse_steps(step_names):
        step.fit(fit_out)
        fit_out = step.transform(fit_out)
        others_out = [step.transform(df) for df in others_out]

    others_out = [df.reindex(columns=fit_out.columns) for df in others_out]

    new_cols = FeatureColumns()
    known_numeric = set(feature_cols.numeric)
    known_high = set(feature_cols.high_card_categorical)
    known_cat = set(feature_cols.categorical)
    for col in fit_out.columns:
        series = fit_out[col]
        if col in known_numeric or (col not in known_cat and pd.api.types.is_numeric_dtype(series)):
            if not pd.api.types.is_numeric_dtype(series):
                raise FeatureStepError(f"column {col!r} was numeric before feature steps but is not any more")
            new_cols.numeric.append(col)
            continue
        new_cols.categorical.append(col)
        is_high = col in known_high if col in known_cat else series.nunique(dropna=True) > high_cardinality_threshold
        (new_cols.high_card_categorical if is_high else new_cols.low_card_categorical).append(col)
    return fit_out, others_out, new_cols
