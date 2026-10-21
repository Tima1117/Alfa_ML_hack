"""Run the research agent on one task locally (not the platform entry point — that is solution.py).

    python scripts/run_agent_local.py "Base/Datasets/Регрессия/task.md" --budget-s 900
    python scripts/run_agent_local.py <task.md> --no-llm

LLM settings come from OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL (same as baseline.py);
without them the agent runs its no-LLM playbook. Outputs go to runs/agent/<task dir name>/ by default
(submission.csv, report.md, report.json, agent.log, experiments/) so the dataset folder stays clean.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.research import run_research  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_md")
    parser.add_argument("--budget-s", type=float, default=900.0, help="total wall-clock budget incl. final refit")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    task_md = Path(args.task_md).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "agent" / task_md.parent.name
    report = run_research(
        task_md,
        deadline_ts=time.time() + args.budget_s,
        llm=None if args.no_llm else "env",
        work_dir=out_dir,
        out_path=out_dir / "submission.csv",
    )
    print(f"\nstatus={report.status} best={report.best_metric} submitted={report.submitted_experiment_id}")
    print(f"report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
