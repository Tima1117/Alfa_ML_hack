"""How the agent runs ml_core operations.

The research loop never calls ml_core.training / ml_core.prediction
directly; it goes through an Executor. InProcessExecutor runs them directly
(ml_core's time budget is soft: checked between CV folds); SubprocessExecutor
(the agent's default) runs each call in a fresh process and kills it on a hard
timeout. Infrastructure can pass its own Executor without touching the agent. An Executor
must never raise for an ML-level failure: return a failed result instead,
exactly like ml_core does.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ml_core import experiment_log, prediction, training
from ml_core.contracts import DataProfile, ExperimentConfig, ExperimentResult, PredictionResult, TaskSpec, ValidationPlan


@dataclass(frozen=True)
class ExperimentContext:
    task_spec: TaskSpec
    data_profile: DataProfile
    validation_plan: ValidationPlan
    train_path: Path
    test_path: Path
    experiments_dir: Path


class Executor(Protocol):
    def run_experiment(self, ctx: ExperimentContext, config: ExperimentConfig, timeout_s: float | None) -> ExperimentResult:
        """Cross-validate one config. Must append the result to ctx.experiments_dir's log."""

    def refit_and_predict(
        self, ctx: ExperimentContext, config: ExperimentConfig, out_path: Path, timeout_s: float | None
    ) -> PredictionResult:
        """Fit on all of train, write a checked submission atomically to out_path."""


class InProcessExecutor:
    def run_experiment(self, ctx: ExperimentContext, config: ExperimentConfig, timeout_s: float | None) -> ExperimentResult:
        if timeout_s is not None:
            config = config.model_copy(update={"time_budget_s": max(1.0, timeout_s)})
        return training.run_experiment(
            ctx.task_spec, config, ctx.validation_plan, ctx.data_profile, ctx.train_path, ctx.experiments_dir
        )

    def refit_and_predict(
        self, ctx: ExperimentContext, config: ExperimentConfig, out_path: Path, timeout_s: float | None
    ) -> PredictionResult:
        return prediction.refit_and_predict(ctx.task_spec, config, ctx.data_profile, ctx.train_path, ctx.test_path, out_path)


class SubprocessExecutor:
    """Runs every ml_core call in a fresh `python -m agent._worker` process with a hard timeout.

    On timeout the process is killed and a status="timeout" result is logged, so a hanging fit (or
    generated feature code) can never eat the rest of the task's budget. Costs ~2-3s of interpreter
    start-up per call, which is negligible next to model fitting.
    """

    def __init__(self, python: str | None = None, grace_s: float = 5.0) -> None:
        self.python = python or sys.executable
        self.grace_s = grace_s

    def _call(self, request: dict, timeout_s: float | None, work_dir: Path) -> dict | None:
        work_dir.mkdir(parents=True, exist_ok=True)
        fd, req_name = tempfile.mkstemp(dir=work_dir, prefix="req_", suffix=".json")
        os.close(fd)
        req_path = Path(req_name)
        resp_path = req_path.with_name(req_path.name.replace("req_", "resp_"))
        req_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        cmd = [self.python, "-m", "agent._worker", str(req_path), str(resp_path)]
        root = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))}
        proc = subprocess.Popen(cmd, cwd=str(work_dir), env=env, start_new_session=True)
        try:
            proc.wait(timeout=None if timeout_s is None else max(1.0, timeout_s) + self.grace_s)
        except subprocess.TimeoutExpired:
            # os.killpg/SIGKILL only exist on POSIX; start_new_session=True is a no-op on
            # Windows, where proc.kill() (TerminateProcess) is already the right call.
            if hasattr(os, "killpg") and hasattr(signal, "SIGKILL"):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
            else:
                proc.kill()
            proc.wait()
            return None
        finally:
            req_path.unlink(missing_ok=True)
        if not resp_path.is_file():
            return {"crashed": f"worker exited with code {proc.returncode} without a result"}
        data = json.loads(resp_path.read_text(encoding="utf-8"))
        resp_path.unlink(missing_ok=True)
        return data

    def _ctx_payload(self, ctx: ExperimentContext) -> dict:
        return {
            "task_spec": ctx.task_spec.model_dump(mode="json"),
            "data_profile": ctx.data_profile.model_dump(mode="json"),
            "validation_plan": ctx.validation_plan.model_dump(mode="json"),
            "train_path": str(ctx.train_path),
            "test_path": str(ctx.test_path),
            "experiments_dir": str(ctx.experiments_dir),
        }

    def run_experiment(self, ctx: ExperimentContext, config: ExperimentConfig, timeout_s: float | None) -> ExperimentResult:
        if timeout_s is not None:
            config = config.model_copy(update={"time_budget_s": max(1.0, timeout_s)})
        start = time.monotonic()
        request = {"op": "run_experiment", "ctx": self._ctx_payload(ctx), "config": config.model_dump(mode="json")}
        data = self._call(request, timeout_s, ctx.experiments_dir / "_worker")
        if data is not None and "result" in data:
            return ExperimentResult.model_validate(data["result"])
        status = "timeout" if data is None else "failed"
        message = f"killed after the hard timeout of {timeout_s:.0f}s" if data is None else data.get("crashed", "worker failed")
        result = ExperimentResult(
            status=status,
            experiment_id=config.experiment_id or "unknown",
            metric_name=ctx.task_spec.metric_name,
            greater_is_better=ctx.task_spec.greater_is_better,
            validation_id=ctx.validation_plan.validation_id,
            n_folds_requested=ctx.validation_plan.n_splits,
            duration_s=time.monotonic() - start,
            config=config,
            error_message=message,
        )
        experiment_log.record(result, ctx.experiments_dir)
        return result

    def refit_and_predict(
        self, ctx: ExperimentContext, config: ExperimentConfig, out_path: Path, timeout_s: float | None
    ) -> PredictionResult:
        request = {"op": "refit_and_predict", "ctx": self._ctx_payload(ctx), "config": config.model_dump(mode="json"),
                   "out_path": str(out_path)}
        data = self._call(request, timeout_s, ctx.experiments_dir / "_worker")
        if data is not None and "result" in data:
            return PredictionResult.model_validate(data["result"])
        message = f"killed after the hard timeout of {timeout_s:.0f}s" if data is None else data.get("crashed", "worker failed")
        return PredictionResult(status="failed", output_path=str(out_path), error_message=message)
