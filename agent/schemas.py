"""Shapes of the JSON the LLM is asked to return.

These are deliberately lenient (extra keys ignored, most fields optional):
a model that adds a stray key should not cost a whole LLM call. Strictness
is applied one step later, when a Hypothesis is turned into an ml_core
ExperimentConfig (extra="forbid" there), and any error from that step is
fed back to the LLM as the hypothesis' outcome.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TaskUnderstanding(_Lenient):
    metric_name: str | None = None
    greater_is_better: bool | None = None
    time_column: str | None = None
    group_column: str | None = None
    columns_to_drop: list[str] = Field(default_factory=list)
    summary: str = ""

    @field_validator("columns_to_drop", mode="before")
    @classmethod
    def _none_to_list(cls, v: Any) -> Any:
        return v or []


class Hypothesis(_Lenient):
    title: str = ""
    rationale: str = ""
    kind: Literal["model", "params", "preprocessing", "features", "other"] = "other"
    base: Literal["best", "scratch"] = "best"
    model_name: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    feature_steps: list[str] | None = None  # replaces the base config's steps when given
    add_feature_steps: list[str] = Field(default_factory=list)  # appended to the base config's steps
    feature_code: str | None = None

    @field_validator("hyperparameters", "preprocessing", mode="before")
    @classmethod
    def _none_to_dict(cls, v: Any) -> Any:
        return v or {}

    @field_validator("add_feature_steps", mode="before")
    @classmethod
    def _none_to_list(cls, v: Any) -> Any:
        return v or []

    @field_validator("kind", mode="before")
    @classmethod
    def _unknown_kind(cls, v: Any) -> Any:
        allowed = {"model", "params", "preprocessing", "features", "other"}
        return v if v in allowed else "other"

    @field_validator("base", mode="before")
    @classmethod
    def _unknown_base(cls, v: Any) -> Any:
        return v if v in ("best", "scratch") else "best"


class HypothesisBatch(_Lenient):
    analysis: str = ""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    stop: bool = False
