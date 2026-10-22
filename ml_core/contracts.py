"""Data contracts shared between the ML core, the LLM agent and infrastructure.

Every model here is a pydantic BaseModel: JSON-serializable via
``model_dump_json()`` / ``model_validate_json()``, so these objects survive a
process boundary (e.g. the "every tool call is a fresh subprocess" style of
agent loop used in the organizers' baseline.py) as well as being usable
directly in-process. ``extra="forbid"`` is set everywhere so a typo'd field
from a hallucinated agent config fails loudly at construction time instead of
silently being ignored.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskType = Literal["regression", "classification"]
ExperimentStatus = Literal["success", "failed", "timeout", "unsupported"]
ValidationStrategy = Literal["kfold", "stratified_kfold", "time_split", "group_kfold"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Task specification
# --------------------------------------------------------------------------- #


class PredictionRequirements(_Strict):
    """Format rules the final submission file must satisfy."""

    output_columns: list[str]
    allow_nan: bool = False
    allow_inf: bool = False


class TaskSpec(_Strict):
    """What the task is, derived primarily from data_dictionary.csv / sample_submission.csv.

    Deliberately NOT inferred from free-text task.md: the markdown is prose
    aimed at an LLM, not a guaranteed schema. Use profiling.infer_task_spec()
    to build this from the files the platform guarantees, and override fields
    by hand (or ask the LLM agent to confirm them) when something looks off.
    """

    task_type: TaskType
    target_column: str
    id_columns: list[str] = Field(default_factory=lambda: ["id"])
    metric_name: str
    greater_is_better: bool
    class_labels: list[str] | None = None
    prediction_requirements: PredictionRequirements
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _classification_needs_labels(self) -> "TaskSpec":
        if self.task_type == "classification" and not self.class_labels:
            raise ValueError("class_labels is required when task_type == 'classification'")
        return self


# --------------------------------------------------------------------------- #
# Data profile
# --------------------------------------------------------------------------- #


class ColumnProfile(_Strict):
    name: str
    role: Literal["id", "target", "feature"]
    declared_dtype: str | None = None
    inferred_kind: Literal["numeric", "categorical", "binary", "constant", "datetime_like"]
    missing_rate: float
    n_unique: int
    is_high_cardinality: bool = False
    train_test_category_overlap: float | None = None
    possible_time_or_group_column: bool = False
    warnings: list[str] = Field(default_factory=list)


class DataProfile(_Strict):
    n_rows_train: int
    n_rows_test: int | None = None
    columns: list[ColumnProfile]
    duplicated_rows_train: int
    id_is_unique_train: bool
    target_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class ValidationPlan(_Strict):
    validation_id: str
    strategy: ValidationStrategy
    n_splits: int
    seed: int
    train_fingerprint: str
    time_column: str | None = None
    group_column: str | None = None
    fold_indices_path: str
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #


class PreprocessingConfig(_Strict):
    mode: Literal["native", "linear"] = "native"
    numeric_imputer: Literal["median", "mean", "none"] = "median"
    categorical_imputer: Literal["most_frequent", "constant"] = "most_frequent"
    high_cardinality_threshold: int = 20
    scale_numeric: bool = False
    target_transform: Literal["none", "log1p"] = "none"
    drop_constant_columns: bool = True


class ExperimentConfig(_Strict):
    experiment_id: str | None = None
    model_name: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    feature_steps: list[str] = Field(default_factory=list)
    random_seed: int = 42
    validation_id: str
    time_budget_s: float | None = None
    n_jobs: int | None = None
    # When set (2+ entries), this config represents a soft-voted average of these component
    # configs' predictions rather than a single model — model_name/hyperparameters/preprocessing/
    # feature_steps above are then unused placeholders (kept only so the field stays required).
    # run_experiment()/refit_and_predict() branch on this at the very top; every other field and
    # every caller (Executor, agent/research.py's _better()/refit_best()) treats it like any other
    # ExperimentConfig, so no other code needs to know ensembles exist.
    ensemble_of: list[ExperimentConfig] | None = None


class ExperimentResult(_Strict):
    status: ExperimentStatus
    experiment_id: str
    metric_name: str
    greater_is_better: bool
    metric_value: float | None = None
    metric_std: float | None = None
    fold_metrics: list[float] = Field(default_factory=list)
    n_folds_completed: int = 0
    n_folds_requested: int = 0
    validation_id: str
    duration_s: float = 0.0
    config: ExperimentConfig
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    error_message: str | None = None

    @property
    def is_fully_comparable(self) -> bool:
        """True only if every requested fold actually ran (no early-timeout)."""
        return self.status == "success" and self.n_folds_completed == self.n_folds_requested


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


class PredictionResult(_Strict):
    status: Literal["success", "failed"]
    output_path: str
    n_rows: int = 0
    checks_passed: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
