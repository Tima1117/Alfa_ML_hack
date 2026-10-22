import numpy as np
import pandas as pd

from ml_core.target_encoding import maybe_apply_target_encoding, out_of_fold_target_encode
from ml_core.preprocessing import FeatureColumns


def test_unique_per_row_category_does_not_leak_its_own_target():
    """The adversarial case: every row is its own category (cardinality == n_rows), so a plain
    (non-OOF) mean encoding would trivially reproduce each row's own target exactly — a perfect,
    massively-overfit "predictor" that is pure leakage. The OOF version must not do that: an
    inner fold never sees a category that exists in exactly one row, so it falls back to the
    global mean for every row instead of silently handing the model the answer."""
    n = 200
    rng = np.random.default_rng(0)
    y = rng.normal(50, 10, n)
    df = pd.DataFrame({"id_like_cat": [f"c{i}" for i in range(n)]})

    encoded, _ = out_of_fold_target_encode(df, y, [], ["id_like_cat"], task_type="regression")
    col = encoded["id_like_cat__target_enc"].to_numpy()

    assert not np.allclose(col, y, atol=1.0)  # must NOT just reproduce each row's own target
    assert np.allclose(col, y.mean(), atol=0.5)  # falls back to the global mean instead


def test_real_signal_is_recovered_for_a_genuinely_informative_category():
    n = 600
    rng = np.random.default_rng(1)
    cat = rng.choice(["A", "B", "C"], n)  # few, well-populated categories
    true_mean = {"A": 10.0, "B": 50.0, "C": 90.0}
    y = np.array([true_mean[c] for c in cat]) + rng.normal(0, 2, n)
    df = pd.DataFrame({"cat": cat})

    encoded, _ = out_of_fold_target_encode(df, y, [], ["cat"], task_type="regression", smoothing=5.0)
    means = encoded.groupby(df["cat"])["cat__target_enc"].mean()
    assert means["A"] < means["B"] < means["C"]
    assert abs(means["A"] - 10.0) < 3.0 and abs(means["C"] - 90.0) < 3.0


def test_other_frames_use_the_full_fit_encoding_including_unseen_categories():
    n = 300
    rng = np.random.default_rng(2)
    cat = rng.choice(["A", "B"], n)
    y = np.where(cat == "A", 0.0, 100.0) + rng.normal(0, 1, n)
    fit_df = pd.DataFrame({"cat": cat})
    other_df = pd.DataFrame({"cat": ["A", "B", "UNSEEN"]})

    _, (other_enc,) = out_of_fold_target_encode(fit_df, y, [other_df], ["cat"], task_type="regression")
    vals = other_enc["cat__target_enc"].to_numpy()
    assert vals[0] < 50 < vals[1]  # A low, B high, matching the fit-time means
    assert abs(vals[2] - y.mean()) < 5.0  # unseen category -> global mean


def test_classification_adds_one_column_per_class():
    n = 300
    rng = np.random.default_rng(3)
    cat = rng.choice(["A", "B", "C"], n)
    y = rng.integers(0, 3, n)
    df = pd.DataFrame({"cat": cat})

    encoded, _ = out_of_fold_target_encode(df, y, [], ["cat"], task_type="classification", n_classes=3)
    for k in range(3):
        col = f"cat__target_enc_c{k}"
        assert col in encoded.columns
        assert encoded[col].between(0, 1).all()


def test_maybe_apply_target_encoding_is_a_no_op_without_the_step_name():
    df = pd.DataFrame({"cat": ["A", "B", "A"]})
    other = pd.DataFrame({"cat": ["A", "B"]})
    cols = FeatureColumns(numeric=[], categorical=["cat"])
    y = np.array([1.0, 2.0, 3.0])

    remaining, fit_out, others_out, new_cols = maybe_apply_target_encoding(
        ["count_encode"], df, y, [other], cols, "regression", 0, seed=0
    )
    assert remaining == ["count_encode"]
    assert fit_out is df and others_out == [other]
    assert new_cols is cols


def test_maybe_apply_target_encoding_strips_the_step_and_adds_numeric_columns():
    df = pd.DataFrame({"cat": ["A", "B", "A", "B"] * 10})
    other = pd.DataFrame({"cat": ["A", "B"]})
    cols = FeatureColumns(numeric=["x"], categorical=["cat"])
    y = np.linspace(0, 1, 40)

    remaining, fit_out, (other_out,), new_cols = maybe_apply_target_encoding(
        ["target_encode", "count_encode"], df, y, [other], cols, "regression", 0, seed=0
    )
    assert remaining == ["count_encode"]  # target_encode consumed, the rest untouched
    assert "cat__target_enc" in fit_out.columns and "cat__target_enc" in other_out.columns
    assert new_cols.numeric == ["x", "cat__target_enc"]
    assert new_cols.categorical == ["cat"]  # original categorical columns are kept, not replaced
