# agent — LLM research agent: contract for infrastructure

`agent/` is the "brain" on top of `ml_core`: it understands the task,
plans experiments (hypotheses), runs them through `ml_core` under one fixed,
honest validation split, analyses the results and decides what to try next
(the AutoResearch loop from the case PDF). It writes `submission.csv` only
through `ml_core` (format-checked, atomic).

## Entry point

```python
import time
from agent.research import run_research

report = run_research(
    "path/to/task.md",                 # the platform's task file (argv[1] in baseline.py)
    deadline_ts=time.time() + budget,  # absolute wall-clock deadline for EVERYTHING incl. the final refit
    executor=None,                     # default: SubprocessExecutor (hard timeout per ml_core call)
    llm="env",                         # "env" -> OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL; None -> no LLM
    work_dir=None,                     # default: <task dir>/agent_work
    out_path=None,                     # default: <task dir>/submission.csv
)
```

- **Never raises.** Any unexpected error ends the run with
  `report.status == "error"` and `report.error`; whatever submission was
  written before stays on disk.
- **A valid `submission.csv` exists within seconds**: the first thing the
  agent does is `ml_core.prediction.write_fallback_submission`. After the
  seed experiments it is replaced by a real model, later by better ones
  (checkpoint every 30 min, plus the final refit).
- **Time:** the agent never reads the platform clock, only `deadline_ts`.
  It keeps a reserve for the final refit (≈1.4× one CV fold of the best
  model × 1.5 + 90 s). Pass a deadline a few minutes before the platform's
  3-hour kill (the default without `deadline_ts` is start + 2h45m, or
  `AGENT_TIME_BUDGET_S`).
- Output files in `work_dir`: `report.md` / `report.json` (journal of every
  hypothesis: source, rationale, status, metric, duration), `agent.log`,
  `experiments/experiments.jsonl` (ml_core ledger), `validation/`,
  `features/` (accepted generated feature code + sandbox reports).

## Minimal `solution.py` (infra owns the real one)

```python
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent.research import run_research

START = time.time()
if __name__ == "__main__":
    task_md = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md")
    run_research(task_md, deadline_ts=START + 3 * 3600 - 10 * 60)
```

## Executor (where infra can plug in)

The agent runs every `ml_core` call through an `Executor`
(`agent/executor.py`): `run_experiment(ctx, config, timeout_s)` and
`refit_and_predict(ctx, config, out_path, timeout_s)`, both must return
ml_core result objects and never raise.

- `SubprocessExecutor` (default): `python -m agent._worker` per call, kills
  the process group on timeout, logs `status="timeout"` into the ledger.
  Needs the repo root importable (it sets `PYTHONPATH` itself).
- `InProcessExecutor`: direct calls, soft time budget only (between folds).

Each CV run is capped at `min(time left − reserve, AGENT_MAX_EXPERIMENT_S=1800)`.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | — | LLM endpoint, same as `baseline.py`; any OpenAI-compatible server |
| `AGENT_DISABLE_LLM` | off | `1` = run the no-LLM playbook only |
| `AGENT_LLM_MAX_CALLS` / `AGENT_LLM_MAX_TOKENS` | 40 / 400000 | per-task LLM budget (team limit is daily and shared) |
| `AGENT_LLM_MAX_OUTPUT_TOKENS` | 6144 | max tokens per answer (Qwen reasoning included; an answer cut off while reasoning is retried once without reasoning) |
| `AGENT_LLM_TEMPERATURE` | 0.4 | |
| `AGENT_LLM_TIMEOUT_S` | 180 | per request |
| `AGENT_LLM_REASONING_TOKENS` | 3000 | cap on Qwen reasoning per answer (proxy field `reasoning.max_tokens`; 0 = no cap) |
| `AGENT_LLM_NO_THINK` | off | `1` turns reasoning off (`reasoning.effort=none`): ~3x cheaper, weaker answers |
| `AGENT_TIME_BUDGET_S` | 9900 | used only when `deadline_ts` is not passed |
| `AGENT_MAX_ROUNDS` / `AGENT_PATIENCE` | 60 / 4 | research rounds / LLM rounds without a new best before switching to the playbook |
| `AGENT_LLM_REST_ROUNDS` | 6 | playbook/local-search rounds before the LLM gets another turn (patience 2 on return); no LLM round is started when an answer plus one experiment like the best no longer fits in the time left |
| `AGENT_MAX_EXPERIMENT_S` | 1800 | hard cap for one CV run |
| `AGENT_N_JOBS` | -1 | threads per model fit |
| `AGENT_FEATURE_CODE` | on | `0` disables LLM-written feature code |

## How a run goes

1. **Locate files**: paths mentioned in task.md, else search below its folder.
2. **Profile + fallback submission** (`ml_core.profiling`, `write_fallback_submission`).
3. **Understand** (1 LLM call): metric, time/group columns, columns to drop,
   short assessment. Rules decide first, the LLM only refines with answers
   that pass checks: the metric in task.md's text wins over the LLM; a
   time split is used only when every test value of the column is later
   than every train value (catches `Year_Factor`, ignores `swin`/`dwin`).
4. **Seeds**: LightGBM and CatBoost defaults → refit → real submission.
5. **Research rounds** (≤3 hypotheses per LLM call): each hypothesis is a
   change to the current best config — model, hyperparameters,
   preprocessing, built-in feature steps, or LLM-written
   `make_features(df)` code checked in a sandbox (static allowlist +
   isolated process on the real data: deterministic, same rows, no target,
   bounded columns). Invalid/duplicate/rejected ideas are logged and fed
   back to the LLM. After `AGENT_PATIENCE` rounds without a new best, or
   when the LLM says stop, or when the LLM is unavailable, a generic
   playbook continues (log1p target, regularized/bigger boosting, missing
   indicators, count encoding, XGBoost, deeper CatBoost, dropping a period column),
   and once that is exhausted, a local search (1–2 hyperparameters of the best
   config perturbed per experiment) until the time or `AGENT_MAX_ROUNDS` runs out.
6. **Final**: best by `ml_core.experiment_log.select_best` on the run's
   validation_id → refit on all train rows → `submission.csv` → report.

## Measured so far (no LLM, `--budget-s 600`, this container: 4 CPU)

| task | validation | Tima's best (examples/) | agent playbook |
|---|---|---|---|
| regression, RMSE ↓ | time_split on `Year_Factor` | 46.09 (CatBoost + log1p) | **42.97** (LightGBM regularized + log1p, local search; 43.21 before it) |
| multiclass, Macro F1 ↑ | stratified 3-fold | 0.537 (CatBoost, 5-fold) | **0.580** (LightGBM seed; CatBoost seed 0.529) |
