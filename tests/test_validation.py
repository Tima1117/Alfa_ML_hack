import numpy as np
import pandas as pd
import pytest

from ml_core.errors import ValidationMismatchError, ValidationPlanError
from ml_core.profiling import infer_task_spec
from ml_core.validation import create_validation, load_fold_indices


def test_same_seed_gives_identical_folds(regression_dataset, tmp_path):
    paths = regression_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    plan_a = create_validation(task_spec, paths["train"], tmp_path / "val_a", n_splits=4, seed=7)
    plan_b = create_validation(task_spec, paths["train"], tmp_path / "val_b", n_splits=4, seed=7)
    folds_a = load_fold_indices(plan_a, paths["train"])
    folds_b = load_fold_indices(plan_b, paths["train"])
    assert plan_a.validation_id == plan_b.validation_id
    for fa, fb in zip(folds_a, folds_b):
        assert np.array_equal(np.sort(fa.val_idx), np.sort(fb.val_idx))
        assert np.array_equal(np.sort(fa.train_idx), np.sort(fb.train_idx))


def test_folds_are_disjoint_and_cover_all_rows(regression_dataset, tmp_path):
    paths = regression_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    plan = create_validation(task_spec, paths["train"], tmp_path / "val", n_splits=5, seed=1, strategy="kfold")
    folds = load_fold_indices(plan, paths["train"])
    all_val_idx = np.concatenate([f.val_idx for f in folds])
    assert len(all_val_idx) == len(set(all_val_idx.tolist()))  # disjoint
    assert set(all_val_idx.tolist()) == set(range(300))  # covers every row (300 = regression_dataset n_train)
    for f in folds:
        # kfold has no ordering constraint: train is exactly "everything not in val"
        assert set(f.train_idx.tolist()) == set(range(300)) - set(f.val_idx.tolist())


def test_stratified_kfold_caps_n_splits_to_rarest_class(classification_dataset, tmp_path):
    paths = classification_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    # ask for way more splits than the rarest class can support
    plan = create_validation(
        task_spec, paths["train"], tmp_path / "val", n_splits=1000, seed=1, strategy="stratified_kfold"
    )
    assert plan.n_splits < 1000
    assert any("reduced" in w for w in plan.warnings)


def test_reusing_plan_against_changed_train_file_raises(regression_dataset, tmp_path):
    paths = regression_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    plan = create_validation(task_spec, paths["train"], tmp_path / "val", n_splits=4, seed=1)
    # mutate train.csv after the plan was built
    with open(paths["train"], "a", encoding="utf-8") as fh:
        fh.write("99999,1.0,1,Commercial,42.0\n")
    with pytest.raises(ValidationMismatchError):
        load_fold_indices(plan, paths["train"])


def test_time_split_requires_enough_distinct_periods(regression_dataset, tmp_path):
    paths = regression_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    with pytest.raises(ValidationPlanError):
        create_validation(
            task_spec,
            paths["train"],
            tmp_path / "val",
            strategy="time_split",
            time_column="quality_flag",  # only 2 distinct values, asking for 5 splits
            n_splits=5,
        )


@pytest.fixture
def periodic_dataset(tmp_path):
    """6 distinct periods (1..6), 50 rows each — enough for a 3-fold walk-forward time_split."""
    rng = np.random.default_rng(3)
    n_per_period = 50
    period = np.repeat(np.arange(1, 7), n_per_period)
    n = len(period)
    train = pd.DataFrame({"id": np.arange(n), "period": period, "x": rng.normal(size=n), "y": rng.normal(size=n)})
    dd = pd.DataFrame(
        {"feature": ["id", "period", "x", "y"], "role": ["id", "feature", "feature", "target"],
         "dtype": ["numeric"] * 4, "unit": [""] * 4, "description_ru": [""] * 4}
    )
    sub = pd.DataFrame({"id": [], "y": []})
    paths = {"train": tmp_path / "train.csv", "dict": tmp_path / "data_dictionary.csv", "sub": tmp_path / "sample_submission.csv"}
    train.to_csv(paths["train"], index=False)
    dd.to_csv(paths["dict"], index=False)
    sub.to_csv(paths["sub"], index=False)
    return paths, train


def test_time_split_multi_fold_is_walk_forward_no_future_leakage(periodic_dataset, tmp_path):
    paths, train = periodic_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    plan = create_validation(
        task_spec, paths["train"], tmp_path / "val", strategy="time_split", time_column="period", n_splits=3
    )
    assert plan.n_splits == 3
    folds = load_fold_indices(plan, paths["train"])
    period = train["period"].to_numpy()

    val_periods_seen = []
    for fold in folds:
        val_periods = set(period[fold.val_idx].tolist())
        train_periods = set(period[fold.train_idx].tolist())
        assert len(val_periods) == 1  # each fold holds out exactly one period
        (holdout,) = val_periods
        val_periods_seen.append(holdout)
        # the core guarantee: training never sees this fold's holdout period or anything after it
        assert max(train_periods) < holdout
        assert all(p < holdout for p in train_periods)

    # the three holdout periods are the three latest (6, 5, 4), each used exactly once
    assert sorted(val_periods_seen) == [4, 5, 6]
    # expanding window: the fold holding out the latest period (6) must have seen strictly
    # more training rows than the fold holding out an earlier period (4)
    sizes = {holdout: len(fold.train_idx) for holdout, fold in zip(val_periods_seen, folds)}
    assert sizes[6] > sizes[5] > sizes[4]


def test_time_split_single_fold_still_works_as_before(periodic_dataset, tmp_path):
    """n_splits=1 (the only mode supported before this fix) must still behave the same:
    one fold, holding out the single latest period, training on every earlier period."""
    paths, train = periodic_dataset
    task_spec = infer_task_spec(paths["train"], paths["dict"], paths["sub"])
    plan = create_validation(
        task_spec, paths["train"], tmp_path / "val", strategy="time_split", time_column="period", n_splits=1
    )
    folds = load_fold_indices(plan, paths["train"])
    assert len(folds) == 1
    period = train["period"].to_numpy()
    assert set(period[folds[0].val_idx].tolist()) == {6}
    assert set(period[folds[0].train_idx].tolist()) == {1, 2, 3, 4, 5}
