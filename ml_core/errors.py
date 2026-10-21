"""Exception types raised by ml_core.

These are reserved for programmer/contract errors (bad TaskSpec, mismatched
validation plan, etc). Runtime failures during an experiment (bad
hyperparameters, a model that fails to converge, an unavailable library) are
NOT raised — they are reported through ExperimentResult.status instead, so a
calling agent never has to wrap every hypothesis test in a try/except.
"""

from __future__ import annotations


class MlCoreError(Exception):
    """Base class for all ml_core exceptions."""


class UnsupportedTaskTypeError(MlCoreError):
    """Raised when a TaskSpec/operation asks for a task_type we don't implement."""


class ValidationPlanError(MlCoreError):
    """Raised when a validation split cannot be built for the given data/strategy."""


class ValidationMismatchError(MlCoreError):
    """Raised when a ValidationPlan is reused against data it wasn't built for."""


class SubmissionFormatError(MlCoreError):
    """Raised by refit_and_predict when the produced predictions fail format checks."""


class UnknownMetricError(MlCoreError):
    """Raised when TaskSpec.metric_name is not in the metric registry."""


class FeatureStepError(MlCoreError):
    """Raised when ExperimentConfig.feature_steps names an unknown/broken step."""
