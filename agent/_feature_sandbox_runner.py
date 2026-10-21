"""Runs one generated make_features() in an isolated interpreter (python -I).

Standalone on purpose: it imports nothing from this repo. Usage:
    python -I _feature_sandbox_runner.py CODE TRAIN_PKL TEST_PKL REPORT_JSON SAMPLE_ROWS MAX_NEW_COLUMNS
Always writes REPORT_JSON with a "problems" list (empty == accepted).
"""

import importlib.util
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd


def _apply(fn, df):
    out = fn(df.copy())
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"make_features returned {type(out).__name__}, expected a pandas DataFrame")
    if len(out) != len(df):
        raise ValueError(f"make_features changed the number of rows ({len(df)} -> {len(out)})")
    if out.columns.duplicated().any():
        raise ValueError(f"duplicate output columns: {out.columns[out.columns.duplicated()].tolist()}")
    return out.reset_index(drop=True).replace([np.inf, -np.inf], np.nan)


def main() -> None:
    code_path, train_path, test_path, report_path, sample_rows, max_new = sys.argv[1:7]
    report = {"problems": []}
    start = time.monotonic()
    try:
        spec = importlib.util.spec_from_file_location("generated_features", code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, "make_features")

        train = pd.read_pickle(train_path)
        test = pd.read_pickle(test_path)
        sample = train.head(int(sample_rows))

        first, second = _apply(fn, sample), _apply(fn, sample)
        try:
            pd.testing.assert_frame_equal(first, second)
        except AssertionError:
            report["problems"].append("make_features is not deterministic (two runs on the same rows differ)")

        out_train = _apply(fn, train)
        out_test = _apply(fn, test)
        if list(out_train.columns) != list(out_test.columns):
            report["problems"].append("make_features produces different columns for train and test")

        new_cols = [c for c in out_train.columns if c not in train.columns]
        dropped = [c for c in train.columns if c not in out_train.columns]
        report["new_columns"], report["dropped_columns"] = [str(c) for c in new_cols], [str(c) for c in dropped]
        if not new_cols and not dropped:
            report["problems"].append("make_features neither adds nor drops any column")
        if len(new_cols) > int(max_new):
            report["problems"].append(f"too many new columns ({len(new_cols)} > {max_new})")
        for col in new_cols:
            s = out_train[col]
            if s.isna().all():
                report["problems"].append(f"new column {col!r} is entirely missing")
            elif s.nunique(dropna=True) <= 1:
                report["problems"].append(f"new column {col!r} is constant")
            elif not pd.api.types.is_numeric_dtype(s) and s.nunique() > 0.5 * len(s):
                report["problems"].append(f"new text column {col!r} is nearly unique per row (id-like)")
        for col in out_train.columns:
            if col in train.columns and pd.api.types.is_numeric_dtype(train[col]) and not pd.api.types.is_numeric_dtype(out_train[col]):
                report["problems"].append(f"existing numeric column {col!r} became non-numeric")
    except Exception as exc:
        tb = traceback.format_exc().strip().splitlines()
        report["problems"].append(f"{type(exc).__name__}: {exc} (at: {tb[-2].strip() if len(tb) > 1 else ''})"[:600])
    report["seconds"] = round(time.monotonic() - start, 2)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False)


if __name__ == "__main__":
    main()
