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
    req_path, resp_path = sys.argv[1], sys.argv[2]
    request = json.loads(Path(req_path).read_text(encoding="utf-8"))
    try:
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
    atomic_write_text(json.dumps(response, ensure_ascii=False), resp_path)


if __name__ == "__main__":
    main()
