"""Synthetic tiny datasets, shaped like the real ones, for fast tests.

Both fixtures deliberately reproduce the two real data quirks that caused
actual bugs while building this module: a two-category *string* feature
(building_class-style) and a categorical value present in test but absent
from train (state-style). A fixture that didn't include these would not
have caught either bug.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def regression_dataset(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    n_train, n_test = 300, 100

    def make(n: int, start_id: int) -> pd.DataFrame:
        size = rng.normal(1000, 200, n).clip(100, None)
        quality = rng.integers(0, 2, n)  # numeric 0/1 flag -> should be "binary", feature engineering-free
        kind = rng.choice(["Commercial", "Residential"], n)  # 2-category STRING feature
        noise = rng.normal(0, 5, n)
        target = size * 0.05 + quality * 10 + (kind == "Commercial") * 20 + noise
        df = pd.DataFrame(
            {
                "id": np.arange(start_id, start_id + n),
                "size": size,
                "quality_flag": quality,
                "building_kind": kind,
                "target": target,
            }
        )
        return df

    train = make(n_train, 0)
    test = make(n_test, n_train).drop(columns=["target"])
    # inject missing values (common in the real data) on a numeric column
    train.loc[train.sample(frac=0.1, random_state=1).index, "size"] = np.nan
    test.loc[test.sample(frac=0.1, random_state=2).index, "size"] = np.nan

    dd = pd.DataFrame(
        {
            "feature": ["id", "size", "quality_flag", "building_kind", "target"],
            "role": ["id", "feature", "feature", "feature", "target"],
            "dtype": ["numeric", "numeric", "binary", "category", "numeric"],
            "unit": ["", "", "", "", ""],
            "description_ru": ["", "", "", "", ""],
        }
    )
    sample_sub = pd.DataFrame({"id": test["id"], "target": 0.0})

    paths = {
        "train": tmp_path / "train.csv",
        "test": tmp_path / "test.csv",
        "dict": tmp_path / "data_dictionary.csv",
        "sub": tmp_path / "sample_submission.csv",
    }
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    dd.to_csv(paths["dict"], index=False)
    sample_sub.to_csv(paths["sub"], index=False)
    return paths


@pytest.fixture
def classification_dataset(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(1)
    n_train, n_test = 400, 120

    def make(n: int, start_id: int, categories: list[str]) -> pd.DataFrame:
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        cat = rng.choice(categories, n)
        score = x1 + 0.5 * x2 + (cat == "A").astype(float)
        label = np.where(score > 0.7, "Yes", np.where(score > -0.3, "Maybe", "No"))
        return pd.DataFrame(
            {
                "id": np.arange(start_id, start_id + n),
                "x1": x1,
                "x2": x2,
                "flow_state": cat,
                "label": label,
            }
        )

    train = make(n_train, 0, ["A", "B", "C"])
    # "D" only ever appears in test — mirrors the real state column (CLO/ACC unseen in train)
    test = make(n_test, n_train, ["A", "B", "D"]).drop(columns=["label"])

    dd = pd.DataFrame(
        {
            "feature": ["id", "x1", "x2", "flow_state", "label"],
            "role": ["id", "feature", "feature", "feature", "target"],
            "dtype": ["numeric", "numeric", "numeric", "category", "category"],
            "unit": ["", "", "", "", ""],
            "description_ru": ["", "", "", "", ""],
        }
    )
    sample_sub = pd.DataFrame({"id": test["id"], "label": "No"})

    paths = {
        "train": tmp_path / "train.csv",
        "test": tmp_path / "test.csv",
        "dict": tmp_path / "data_dictionary.csv",
        "sub": tmp_path / "sample_submission.csv",
    }
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    dd.to_csv(paths["dict"], index=False)
    sample_sub.to_csv(paths["sub"], index=False)
    return paths


@pytest.fixture
def binary_classification_dataset(tmp_path: Path) -> dict[str, Path]:
    """Imbalanced binary target (~10% positive) — shaped for roc_auc/log_loss tests.

    Mirrors the real finding from drilling a synthetic "private-like" task: under real
    imbalance, a hard model.predict() collapses to the majority class for every row,
    which is a valid label but a useless probability (auc ~= 0.5) regardless of how good
    the underlying model is — the dataset needs real separable signal to tell a correctly
    probability-scored submission apart from that collapse.
    """
    rng = np.random.default_rng(7)
    n_train, n_test = 600, 200

    def make(n: int, start_id: int, categories: list[str]) -> pd.DataFrame:
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        cat = rng.choice(categories, n)
        z = -2.5 + 1.8 * x1 + 0.6 * x2 + (cat == "A").astype(float) * 0.8
        prob = 1 / (1 + np.exp(-z))
        label = np.where(rng.random(n) < prob, "Yes", "No")
        return pd.DataFrame(
            {"id": np.arange(start_id, start_id + n), "x1": x1, "x2": x2, "segment": cat, "churn": label}
        )

    train = make(n_train, 0, ["A", "B", "C"])
    test = make(n_test, n_train, ["A", "B", "D"]).drop(columns=["churn"])  # "D" unseen at train time

    dd = pd.DataFrame(
        {
            "feature": ["id", "x1", "x2", "segment", "churn"],
            "role": ["id", "feature", "feature", "feature", "target"],
            "dtype": ["numeric", "numeric", "numeric", "category", "category"],
            "unit": ["", "", "", "", ""],
            "description_ru": ["", "", "", "", ""],
        }
    )
    sample_sub = pd.DataFrame({"id": test["id"], "churn": 0.0})

    paths = {
        "train": tmp_path / "train.csv",
        "test": tmp_path / "test.csv",
        "dict": tmp_path / "data_dictionary.csv",
        "sub": tmp_path / "sample_submission.csv",
    }
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    dd.to_csv(paths["dict"], index=False)
    sample_sub.to_csv(paths["sub"], index=False)
    return paths
