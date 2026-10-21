"""Worker process for SubprocessExecutor: python -m agent._worker REQUEST.json RESPONSE.json"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from ml_core.contracts import DataProfile, ExperimentConfig, TaskSpec, ValidationPlan
from ml_core.io_utils import atomic_write_text

from agent.executor import ExperimentContext, InProcessExecutor


def main() -> None:
    # resp_path is read defensively, outside the try: SubprocessExecutor already tolerates this
    # process exiting without writing it at all (checks resp_path.is_file(), synthesizes its own
    # "crashed" result — see agent/executor.py), so this was never a submission.csv risk the way
    # the child_main()/sched_setaffinity incident was (that crashed before ANY file, including
    # the fallback submission, could be written — see docs/INFRASTRUCTURE.md). Still, writing a
    # response whenever argv is well-formed (even if everything after that fails) beats silently
    # exiting with nothing, which is the one case genuinely worth guarding against here.
    resp_path = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        if len(sys.argv) <= 2:
            raise ValueError(f"expected REQUEST.json RESPONSE.json, got argv={sys.argv[1:]!r}")
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        c = request["ctx"]
        ctx = ExperimentContext(
            task_spec=TaskSpec.model_validate(c["task_spec"]),
            data_profile=DataProfile.model_validate(c["data_profile"]),
            validation_plan=ValidationPlan.model_validate(c["validation_plan"]),
            train_path=Path(c["train_path"]),
            test_path=Path(c["test_path"]),
            experiments_dir=Path(c["experiments_dir"]),
        )
        config = ExperimentConfig.model_validate(request["config"])
        executor = InProcessExecutor()
        if request["op"] == "run_experiment":
            result = executor.run_experiment(ctx, config, timeout_s=config.time_budget_s)
        elif request["op"] == "refit_and_predict":
            result = executor.refit_and_predict(ctx, config, Path(request["out_path"]), timeout_s=None)
        else:
            raise ValueError(f"unknown op {request['op']!r}")
        response = {"result": result.model_dump(mode="json")}
    except Exception:
        response = {"crashed": traceback.format_exc()[-2000:]}
    if resp_path is not None:
        atomic_write_text(json.dumps(response, ensure_ascii=False), resp_path)


if __name__ == "__main__":
    main()
