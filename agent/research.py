"""The AutoResearch loop: hypothesis -> experiment -> evaluation -> next step.

Public entry point for infrastructure:

    from agent.research import run_research
    report = run_research("path/to/task.md", deadline_ts=time.time() + budget_s)

Phases:
  0. locate files, profile data, write a fallback submission immediately;
  1. understand the task: metric + honest validation scheme (rules first, LLM may refine);
  2. seed experiments (strong defaults, no LLM) -> first real submission;
  3. research rounds: the LLM proposes hypotheses from the profile and the
     history, each is run under the same validation; without an LLM (or
     after it runs out) a generic playbook is used instead;
  4. refit the best config on all of train -> final submission, write a report.

run_research() does not raise: any unexpected error ends the run with
status="error" in the report, and the best submission written so far (at
worst the fallback) stays on disk. submission.csv is only ever written by
ml_core (checked, atomic), never by the agent itself.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ml_core import prediction, profiling, validation
from ml_core.contracts import ExperimentResult, PreprocessingConfig, TaskSpec
from ml_core.preprocessing import split_feature_columns

from agent import context, prompts
from agent.executor import ExperimentContext, Executor, SubprocessExecutor
from agent.features import FeatureSandbox
from agent.hypotheses import (
    HypothesisError,
    local_search_hypotheses,
    materialize,
    playbook_hypotheses,
    seed_hypotheses,
    stable_experiment_id,
)
from agent.llm import LLMBadResponse, LLMClient, LLMUnavailable
from agent.schemas import Hypothesis, HypothesisBatch, TaskUnderstanding
from agent.task_files import TaskFiles, locate_task_files, read_task_text
from agent.understand import decide_metric, decide_validation, drop_candidates, time_split_candidates

DEFAULT_TOTAL_BUDGET_S = 3 * 3600 - 15 * 60
_ITERATION_KEYS = ("n_estimators", "iterations", "num_boost_round", "num_iterations", "max_iter")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class ResearchSettings:
    # Not a time budget — the per-round wall-clock check (self.out_of_time / remaining() below)
    # is the only thing that may stop a run early, unconditionally, every round. This cap exists
    # only so a pathological case (every idea instantly skipped, nothing ever costs real time)
    # can't spin forever; a real run exhausts its actual time budget long before 500 rounds.
    # Raised from 60 after a real 3h/qwen3.8-27b run on the Регрессия task used only 78.5 of its
    # ~170 available minutes before hitting the old cap — local-search was still finding small
    # improvements that late (43.98 -> 43.82 RMSE in its last ~10 rounds), so the real budget was
    # sitting idle for no reason.
    max_rounds: int = 500
    hypotheses_per_round: int = 3
    patience_rounds: int = 4  # LLM rounds in a row without a new best before the LLM phase ends
    llm_rest_rounds: int = 6  # playbook/local-search rounds before the LLM gets another turn
    return_patience_rounds: int = 2  # patience on such a return visit
    max_bad_llm_answers: int = 3
    n_jobs: int | None = -1
    seed: int = 42
    refit_safety: float = 1.5
    safety_margin_s: float = 90.0
    checkpoint_every_s: float = 1800.0
    max_experiment_s: float = 1800.0  # hard cap for one cross-validation run
    feature_code: bool = True

    @classmethod
    def from_env(cls) -> "ResearchSettings":
        s = cls()
        s.max_rounds = int(_env_float("AGENT_MAX_ROUNDS", s.max_rounds))
        s.patience_rounds = int(_env_float("AGENT_PATIENCE", s.patience_rounds))
        s.llm_rest_rounds = int(_env_float("AGENT_LLM_REST_ROUNDS", s.llm_rest_rounds))
        s.max_experiment_s = _env_float("AGENT_MAX_EXPERIMENT_S", s.max_experiment_s)
        n_jobs = os.environ.get("AGENT_N_JOBS")
        if n_jobs:
            s.n_jobs = int(n_jobs)
        s.feature_code = os.environ.get("AGENT_FEATURE_CODE", "1") not in ("0", "false", "no")
        return s


@dataclass
class JournalEntry:
    round: int
    source: str
    title: str
    rationale: str = ""
    kind: str = ""
    experiment_id: str | None = None
    status: str = ""
    metric: float | None = None
    std: float | None = None
    duration_s: float = 0.0
    new_best: bool = False
    note: str = ""


@dataclass
class ResearchReport:
    status: str = "running"
    task_md: str = ""
    submission_path: str = ""
    work_dir: str = ""
    task_type: str | None = None
    metric_name: str | None = None
    greater_is_better: bool | None = None
    metric_source: str | None = None
    validation: dict = field(default_factory=dict)
    understanding: str = ""
    best_experiment_id: str | None = None
    best_metric: float | None = None
    best_std: float | None = None
    best_config: dict | None = None
    submitted_experiment_id: str | None = None
    n_experiments: int = 0
    rounds: int = 0
    llm_mode: str = "off"
    llm_usage: dict = field(default_factory=dict)
    journal: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_s: float = 0.0

    def write(self, work_dir: Path) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "report.json").write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        (work_dir / "report.md").write_text(render_markdown(self), encoding="utf-8")


def render_markdown(r: ResearchReport) -> str:
    direction = "higher is better" if r.greater_is_better else "lower is better"
    lines = [
        f"# Agent report: {Path(r.task_md).parent.name or r.task_md}",
        "",
        f"- status: **{r.status}**{f' — {r.error}' if r.error else ''}",
        f"- task: {r.task_type}, metric **{r.metric_name}** ({direction}, from {r.metric_source})",
        f"- validation: {r.validation.get('strategy')} × {r.validation.get('n_splits')} — {r.validation.get('reason', '')}",
        f"- best: `{r.best_experiment_id}` = {r.best_metric} ± {r.best_std}; submitted: `{r.submitted_experiment_id}`",
        f"- experiments: {r.n_experiments}, rounds: {r.rounds}, elapsed: {r.elapsed_s / 60:.1f} min",
        f"- LLM: {r.llm_mode}; usage: {json.dumps({k: v for k, v in r.llm_usage.items() if k != 'log'})}",
        "",
    ]
    if r.understanding:
        lines += ["## Initial assessment", "", r.understanding, ""]
    lines += ["## Journal", "", "| round | source | hypothesis | status | metric | ± | s | new best | note |",
              "|---|---|---|---|---|---|---|---|---|"]
    for j in r.journal:
        metric = f"{j['metric']:.5g}" if j["metric"] is not None else ""
        std = f"{j['std']:.2g}" if j["std"] is not None else ""
        note = j["note"].replace("|", "/").replace("\n", " ")[:160]
        lines.append(
            f"| {j['round']} | {j['source']} | {j['title'].replace('|', '/')} | {j['status']} | {metric} | {std} | "
            f"{j['duration_s']:.0f} | {'✓' if j['new_best'] else ''} | {note} |"
        )
    if r.best_config:
        lines += ["", "## Best config", "", "```json", json.dumps(r.best_config, ensure_ascii=False, indent=2), "```"]
    if r.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in r.notes]
    return "\n".join(lines) + "\n"


class _Researcher:
    def __init__(
        self,
        files: TaskFiles,
        deadline_ts: float,
        executor: Executor,
        llm: LLMClient | None,
        work_dir: Path,
        out_path: Path,
        settings: ResearchSettings,
        log: Callable[[str], None],
    ) -> None:
        self.files = files
        self.deadline_ts = deadline_ts
        self.executor = executor
        self.llm = llm
        self.work_dir = work_dir
        self.out_path = out_path
        self.settings = settings
        self._print = log
        self.started = time.time()
        self.report = ResearchReport(task_md=str(files.task_md), submission_path=str(out_path), work_dir=str(work_dir))
        self.report.llm_mode = f"on ({llm.model})" if llm else "off (not configured)"
        self.best: ExperimentResult | None = None
        self.results: list[ExperimentResult] = []
        self.tried: set[str] = set()
        self.titles: dict[str, str] = {}
        self.notes: list[str] = []
        self.bad_llm_answers = 0
        self.last_refit_ts = 0.0
        self.refit_ratio = 1.4
        self.sandbox: FeatureSandbox | None = None
        self.accepted_py_steps: set[str] = set()
        self.rng = random.Random(settings.seed)
        self.ctx: ExperimentContext | None = None
        self.task_spec: TaskSpec | None = None
        self.out_of_time = False

    # ------------------------------------------------------------------ utils
    def log(self, msg: str) -> None:
        line = f"[agent {time.time() - self.started:7.0f}s | left {self.remaining() / 60:6.1f}m] {msg}"
        try:
            self._print(line)
        except UnicodeEncodeError:
            # A non-UTF-8 console (e.g. the default Windows console codepage) can't encode
            # every character — this must never end the run, especially once msg can contain
            # free-text from the LLM. The file log below still gets the exact text.
            self._print(line.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii"))
        try:
            with open(self.work_dir / "agent.log", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def remaining(self) -> float:
        return self.deadline_ts - time.time()

    def _better(self, a: ExperimentResult, b: ExperimentResult | None) -> bool:
        if b is None or b.metric_value is None:
            return True
        assert a.metric_value is not None
        return a.metric_value > b.metric_value if a.greater_is_better else a.metric_value < b.metric_value

    def _estimate_run_s(self, model_name: str, hyper: dict) -> float:
        def ratio_to(ref: ExperimentResult) -> float:
            ratio = 1.0
            for key in _ITERATION_KEYS:
                new, old = hyper.get(key), ref.config.hyperparameters.get(key)
                if isinstance(new, (int, float)) and isinstance(old, (int, float)) and old > 0:
                    ratio = max(ratio, new / old)
            return ratio

        same = [r for r in self.results if r.config.model_name == model_name]
        finished = [r for r in same if r.n_folds_completed > 0]
        estimate = 0.0
        if finished:
            ref = max(finished, key=lambda r: r.duration_s)
            estimate = ref.duration_s * ratio_to(ref) * ref.n_folds_requested / ref.n_folds_completed
        for r in same:
            if r.status == "timeout":  # a similar run already hit the cap: expect at least that long
                estimate = max(estimate, r.duration_s * ratio_to(r))
        if estimate:
            return estimate
        if self.results:
            return 1.5 * max(r.duration_s for r in self.results)
        return 60.0

    def _estimate_refit_s(self, result: ExperimentResult) -> float:
        # refit time ~ one CV fold scaled by self.refit_ratio, which is re-measured after every real refit
        per_fold = result.duration_s / max(1, result.n_folds_completed)
        return per_fold * self.refit_ratio + 15.0

    def _llm_round_fits(self) -> bool:
        """Is there time for an LLM answer plus one experiment like the current best? Else the call is wasted."""
        if self.best is None or self.llm is None:
            return True
        u = self.llm.usage
        llm_s = u.seconds / u.calls if u.calls else 30.0
        needed = self._estimate_run_s(self.best.config.model_name, self.best.config.hyperparameters) * 1.1 + llm_s
        return self.remaining() - self._reserve_s() >= needed

    def _reserve_s(self) -> float:
        refit = self._estimate_refit_s(self.best) * self.settings.refit_safety if self.best else 0.0
        return refit + self.settings.safety_margin_s

    # ----------------------------------------------------------- phase 0 + 1
    def prepare(self) -> None:
        f = self.files
        spec0 = profiling.infer_task_spec(f.train, f.data_dictionary, f.sample_submission)
        profile = profiling.profile_data(f.train, f.data_dictionary, f.test)
        fb = prediction.write_fallback_submission(spec0, profile, f.train, f.test, self.out_path)
        if fb.status == "success":
            self.report.submitted_experiment_id = "fallback"
            self.log(f"fallback submission written to {self.out_path}")
        else:
            self.log(f"WARNING: fallback submission failed: {fb.error_message}")

        task_text = read_task_text(f.task_md)
        descriptions = context.load_descriptions(f.data_dictionary)
        self.data_brief = context.describe_task(spec0, profile, descriptions)
        tcands = time_split_candidates(profile, f.train, f.test)
        flagged = [c.name for c in profile.columns if c.possible_time_or_group_column]

        understanding: TaskUnderstanding | None = None
        if self.llm:
            try:
                understanding = self.llm.complete_json(
                    prompts.UNDERSTAND_SYSTEM,
                    prompts.understand_user(task_text, self.data_brief, tcands, flagged),
                    TaskUnderstanding,
                    purpose="understand",
                )
                self.report.understanding = understanding.summary
                self.log(f"LLM understanding: {understanding.model_dump_json()[:600]}")
            except (LLMUnavailable, LLMBadResponse) as exc:
                self._llm_problem(exc, "understanding")

        md = decide_metric(task_text, spec0, understanding)
        self.notes += md.notes
        spec = spec0
        if (md.metric_name, md.greater_is_better) != (spec0.metric_name, spec0.greater_is_better) or md.source != "default":
            spec = profiling.infer_task_spec(
                f.train, f.data_dictionary, f.sample_submission,
                metric_name=md.metric_name, greater_is_better=md.greater_is_better,
            )
        self.task_spec = spec
        self.report.task_type, self.report.metric_name = spec.task_type, spec.metric_name
        self.report.greater_is_better, self.report.metric_source = spec.greater_is_better, md.source

        if spec.metric_name != spec0.metric_name:
            # The very first fallback above was written against spec0, whose metric is only a
            # data-driven guess (infer_task_spec doesn't read task.md). For a probability metric
            # (roc_auc, log_loss) that guess is usually a hard-label metric instead, so that
            # fallback is a hard class label, not a probability. If every real experiment then
            # fails or times out before refit (private_drill_logloss/README.md: a too-tight
            # budget skipped the only seed candidate), that wrong-shaped fallback is what ships —
            # same "technically a file, but not a valid submission for this metric" failure as
            # the missing-file sched_setaffinity incident, just one level deeper. Refreshing it
            # here, cheaply, before any time budget is spent on real experiments, closes that gap.
            fb2 = prediction.write_fallback_submission(spec, profile, f.train, f.test, self.out_path)
            if fb2.status == "success":
                self.log(f"fallback submission refreshed for metric={spec.metric_name}")
            else:
                self.log(f"WARNING: fallback refresh failed: {fb2.error_message}")

        vd = decide_validation(spec, profile, tcands, understanding)
        try:
            plan = validation.create_validation(
                spec, f.train, self.work_dir / "validation", strategy=vd.strategy, n_splits=vd.n_splits,
                seed=self.settings.seed, time_column=vd.time_column, group_column=vd.group_column,
            )
        except Exception as exc:
            self.notes.append(f"validation {vd.strategy} failed ({exc}); using the default split")
            vd.strategy, vd.reason = "default", f"fallback after error: {exc}"
            plan = validation.create_validation(spec, f.train, self.work_dir / "validation", seed=self.settings.seed)
        self.notes += plan.warnings
        self.report.validation = {
            "strategy": plan.strategy, "n_splits": plan.n_splits, "time_column": plan.time_column,
            "group_column": plan.group_column, "validation_id": plan.validation_id, "reason": vd.reason,
        }
        self.validation_brief = context.describe_metric(spec) + "\n" + context.describe_validation(plan)
        self.drop_candidates = drop_candidates(profile, vd, understanding)
        self.understanding_summary = understanding.summary if understanding else ""
        self.profile = profile
        self.ctx = ExperimentContext(
            task_spec=spec, data_profile=profile, validation_plan=plan, train_path=f.train, test_path=f.test,
            experiments_dir=self.work_dir / "experiments",
        )
        self.log(
            f"task={spec.task_type} metric={spec.metric_name} ({md.source}) "
            f"validation={plan.strategy}x{plan.n_splits} ({vd.reason})"
        )

    # ------------------------------------------------------------- running
    def _llm_problem(self, exc: Exception, where: str) -> None:
        if isinstance(exc, LLMBadResponse):
            self.bad_llm_answers += 1
            self.log(f"LLM gave an unusable answer during {where}: {str(exc)[:300]}")
            if self.bad_llm_answers >= self.settings.max_bad_llm_answers:
                self._disable_llm(f"{self.bad_llm_answers} unusable answers")
        else:
            self.log(f"LLM unavailable during {where}: {exc}")
            if self.llm and not self.llm.available:
                self._disable_llm(self.llm.disabled_reason or str(exc))

    def _disable_llm(self, reason: str) -> None:
        if self.llm is not None:
            self.report.llm_usage = self._usage()
        self.report.llm_mode = f"disabled mid-run: {reason}"
        self.notes.append(f"LLM disabled: {reason}")
        self.llm = None

    def _usage(self) -> dict:
        if self.llm is None:
            return self.report.llm_usage
        u = self.llm.usage
        return {"calls": u.calls, "failed_calls": u.failed_calls, "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens, "cost": round(u.cost, 4), "seconds": round(u.seconds, 1),
                "log": u.log}

    def _sandbox(self) -> FeatureSandbox:
        if self.sandbox is None:
            cols = split_feature_columns(self.profile.columns, PreprocessingConfig())
            keep = cols.numeric + cols.categorical  # what ml_core hands to feature steps (no id/target/constants)
            self.sandbox = FeatureSandbox.from_csv(self.work_dir / "features", self.files.train, self.files.test, keep)
        return self.sandbox

    def _resolve_generated_steps(self, steps: list[str]) -> list[str]:
        """py: steps may only point at code this run's sandbox accepted; short names from the history map back."""
        accepted = {Path(p[3:]).name: p for p in self.accepted_py_steps}
        out = []
        for step in steps:
            if step.startswith("py:"):
                full = step if step in self.accepted_py_steps else accepted.get(Path(step[3:]).name)
                if full is None:
                    raise HypothesisError(f"unknown generated step {step!r}: submit new code via feature_code")
                step = full
            out.append(step)
        return out

    def run_hypothesis(self, hyp: Hypothesis, source: str, round_no: int) -> JournalEntry:
        assert self.ctx is not None
        entry = JournalEntry(round=round_no, source=source, title=hyp.title or "(untitled)", rationale=hyp.rationale, kind=hyp.kind)
        extra_steps: list[str] = []
        if hyp.feature_code:
            if not self.settings.feature_code:
                entry.status, entry.note = "rejected", "feature_code disabled"
                return self._journal(entry)
            check = self._sandbox().check(hyp.feature_code, hyp.title)
            if not check.ok:
                entry.status, entry.note = "rejected", check.summary()
                self.notes.append(f"«{entry.title}»: {check.summary()[:400]}")
                return self._journal(entry)
            extra_steps.append(check.step or "")
            self.accepted_py_steps.add(check.step or "")
            entry.note = check.summary()

        best_cfg = self.best.config if self.best else None
        try:
            config, notes = materialize(
                hyp, best_cfg, self.ctx.validation_plan.validation_id,
                n_jobs=self.settings.n_jobs, seed=self.settings.seed, extra_feature_steps=extra_steps,
                task_type=self.task_spec.task_type if self.task_spec else None,
            )
            resolved = self._resolve_generated_steps(config.feature_steps)
            if resolved != config.feature_steps:
                config = config.model_copy(update={"feature_steps": resolved})
                config = config.model_copy(update={"experiment_id": stable_experiment_id(config)})
        except HypothesisError as exc:
            entry.status, entry.note = "invalid", str(exc)[:400]
            self.notes.append(f"«{entry.title}» invalid: {str(exc)[:300]}")
            return self._journal(entry)
        if notes:
            entry.note = "; ".join([entry.note, *notes]).strip("; ")
        entry.experiment_id = config.experiment_id
        if config.experiment_id in self.tried:
            entry.status, entry.note = "duplicate", "same config as an earlier experiment, skipped"
            return self._journal(entry)

        estimate = self._estimate_run_s(config.model_name, config.hyperparameters)
        available = self.remaining() - self._reserve_s()
        if available < estimate * 1.1:
            entry.status = "skipped"
            entry.note = f"not enough time (needs ~{estimate:.0f}s, {available:.0f}s available)"
            if available < 60:
                self.out_of_time = True
            return self._journal(entry)

        self.log(f"run «{entry.title}» [{source}] {json.dumps(context.compact_config(config), ensure_ascii=False)[:300]}")
        result = self.executor.run_experiment(self.ctx, config, timeout_s=min(available, self.settings.max_experiment_s))
        self.tried.add(config.experiment_id or "")
        self.titles[result.experiment_id] = entry.title
        self.results.append(result)
        entry.status, entry.duration_s = result.status, result.duration_s
        entry.metric, entry.std = result.metric_value, result.metric_std
        if result.status != "success":
            entry.note = "; ".join(filter(None, [entry.note, (result.error_message or "")[:300]]))
        elif result.is_fully_comparable and self._better(result, self.best):
            entry.new_best = True
            self.best = result
        self.log(
            f"  -> {result.status} {self.task_spec.metric_name if self.task_spec else ''}="
            f"{result.metric_value} ± {result.metric_std} in {result.duration_s:.0f}s{'  NEW BEST' if entry.new_best else ''}"
        )
        return self._journal(entry)

    def _journal(self, entry: JournalEntry) -> JournalEntry:
        self.report.journal.append(asdict(entry))
        if entry.status in ("rejected", "invalid", "duplicate", "skipped"):
            self.log(f"«{entry.title}» {entry.status}: {entry.note[:300]}")
        return entry

    def refit_best(self, reason: str) -> None:
        assert self.ctx is not None
        if self.best is None or self.best.experiment_id == self.report.submitted_experiment_id:
            return
        need = self._estimate_refit_s(self.best)
        if self.remaining() < need + 30:
            self.log(f"no time to refit «{self.titles.get(self.best.experiment_id)}» (~{need:.0f}s); keeping current submission")
            return
        self.log(f"refit best ({reason}): «{self.titles.get(self.best.experiment_id)}» on all train rows, ~{need:.0f}s")
        started = time.time()
        pred = self.executor.refit_and_predict(self.ctx, self.best.config, self.out_path, timeout_s=self.remaining() - 10)
        self.last_refit_ts = time.time()
        if pred.status == "success":
            per_fold = self.best.duration_s / max(1, self.best.n_folds_completed)
            if per_fold > 1:
                self.refit_ratio = max(0.5, min(3.0, (self.last_refit_ts - started) / per_fold))
            self.report.submitted_experiment_id = self.best.experiment_id
            self.log(f"submission updated: {self.out_path} ({pred.n_rows} rows)")
        else:
            self.notes.append(f"refit of {self.best.experiment_id} failed: {pred.error_message}")
            self.log(f"refit failed, previous submission kept: {pred.error_message}")

    # ------------------------------------------------------------- phase 3
    def _llm_hypotheses(self, round_no: int) -> tuple[list[Hypothesis], bool]:
        assert self.llm is not None and self.ctx is not None and self.task_spec is not None
        history = context.describe_history(self.results, self.titles, self.task_spec.greater_is_better)
        best_json = json.dumps(context.compact_config(self.best.config), ensure_ascii=False) if self.best else "none yet"
        user = prompts.research_user(
            data_brief=self.data_brief,
            validation_brief=self.validation_brief,
            understanding=self.understanding_summary,
            best_config_json=best_json,
            history=history,
            notes=self.notes,
            minutes_left=max(0.0, (self.remaining() - self._reserve_s()) / 60),
            n_hypotheses=self.settings.hypotheses_per_round,
        )
        batch = self.llm.complete_json(prompts.RESEARCH_SYSTEM, user, HypothesisBatch, purpose=f"round {round_no}")
        if batch.analysis:
            self.notes.append(f"round {round_no} analysis: {batch.analysis[:500]}")
            self.log(f"LLM analysis: {batch.analysis[:500]}")
        return batch.hypotheses[: self.settings.hypotheses_per_round], batch.stop

    def research(self) -> None:
        assert self.task_spec is not None
        for hyp in seed_hypotheses(self.task_spec, self.profile):
            self.run_hypothesis(hyp, "seed", 0)
            if self.out_of_time:
                break
        self.refit_best("after seed experiments")

        playbook = deque(playbook_hypotheses(self.task_spec, self.profile, self.drop_candidates))
        llm_phase = self.llm is not None
        stale, patience, rest = 0, self.settings.patience_rounds, 0
        n_at_leave = 0

        def leave_llm(reason: str) -> None:
            nonlocal llm_phase, rest, n_at_leave
            self.log(reason)
            llm_phase, rest, n_at_leave = False, self.settings.llm_rest_rounds, len(self.results)

        for round_no in range(1, self.settings.max_rounds + 1):
            if self.out_of_time or self.remaining() - self._reserve_s() < 60:
                self.log("time budget for experiments is used up")
                break
            if not llm_phase and self.llm is not None and rest <= 0 and len(self.results) > n_at_leave:
                self.log(f"back to the LLM: {len(self.results) - n_at_leave} new experiments since it last proposed")
                llm_phase, stale, patience = True, 0, self.settings.return_patience_rounds
            source = "playbook"
            hyps: list[Hypothesis] = []
            if llm_phase and self.llm is not None and not self._llm_round_fits():
                self.log("not enough time left for an LLM round plus an experiment; no more LLM calls")
                llm_phase, rest = False, self.settings.max_rounds
            if llm_phase and self.llm is not None:
                try:
                    hyps, stop = self._llm_hypotheses(round_no)
                    source = "llm"
                    if stop or not hyps:
                        leave_llm("LLM decided to stop researching" if stop else "LLM proposed nothing")
                except (LLMUnavailable, LLMBadResponse) as exc:
                    self._llm_problem(exc, f"round {round_no}")
                    if self.llm is None:
                        llm_phase = False
                    else:
                        continue  # one unusable answer: ask again next round
            if not hyps:
                source = "playbook"
                hyps = [playbook.popleft() for _ in range(min(self.settings.hypotheses_per_round, len(playbook)))]
            if not hyps:
                source = "local-search"
                hyps = local_search_hypotheses(
                    self.best.config if self.best else None, self.settings.hypotheses_per_round, self.rng
                )
            if not hyps:
                self.log("no more hypotheses to try")
                break

            self.report.rounds = round_no
            improved = ran = False
            for hyp in hyps:
                entry = self.run_hypothesis(hyp, source, round_no)
                improved |= entry.new_best
                ran |= entry.status != "skipped"  # skipped for lack of time says nothing about the ideas
                if self.out_of_time:
                    break
            if source == "llm" and ran:
                stale = 0 if improved else stale + 1
                if stale >= patience:
                    leave_llm(f"{stale} LLM rounds without a new best; switching to the generic playbook")
            elif source != "llm":
                rest -= 1
            if self.best and time.time() - self.last_refit_ts > self.settings.checkpoint_every_s:
                self.refit_best("checkpoint")

    # ------------------------------------------------------------- phase 4
    def finish(self) -> None:
        # self.best is tracked over this run's results only: the ledger in a reused work_dir may also
        # hold results of earlier runs (same validation_id), which this run never verified.
        self.refit_best("final")
        r = self.report
        if self.best is not None:
            r.best_experiment_id, r.best_metric, r.best_std = self.best.experiment_id, self.best.metric_value, self.best.metric_std
            r.best_config = context.compact_config(self.best.config)
        r.n_experiments = len(self.results)
        r.llm_usage = self._usage()
        r.notes = self.notes
        r.elapsed_s = time.time() - self.started


def run_research(
    task_md: str | Path,
    deadline_ts: float | None = None,
    *,
    executor: Executor | None = None,
    llm: LLMClient | None | str = "env",
    work_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    settings: ResearchSettings | None = None,
    log: Callable[[str], None] | None = None,
) -> ResearchReport:
    """Solve one task end to end. Never raises; see the module docstring.

    deadline_ts: absolute time.time() by which everything (incl. the final refit) must be done.
                 Default: now + AGENT_TIME_BUDGET_S (or 2h45m).
    llm:         "env" -> LLMClient.from_env(); None -> no LLM; or a ready LLMClient.
    """
    start = time.time()
    if deadline_ts is None:
        deadline_ts = start + _env_float("AGENT_TIME_BUDGET_S", DEFAULT_TOTAL_BUDGET_S)
    log = log or (lambda msg: print(msg, flush=True))
    report = ResearchReport(task_md=str(task_md))
    try:
        files = locate_task_files(task_md)
        # absolute: SubprocessExecutor runs its worker in another cwd
        out = Path(out_path).resolve() if out_path else files.default_submission_path
        work = Path(work_dir).resolve() if work_dir else files.task_dir / "agent_work"
        work.mkdir(parents=True, exist_ok=True)
        client = LLMClient.from_env() if llm == "env" else llm
        researcher = _Researcher(
            files, deadline_ts, executor or SubprocessExecutor(), client if not isinstance(client, str) else None,
            work, out, settings or ResearchSettings.from_env(), log,
        )
        report = researcher.report
        try:
            researcher.prepare()
            researcher.research()
            report.status = "ok"
        except Exception as exc:
            report.status, report.error = "error", f"{type(exc).__name__}: {exc}"
            researcher.log("ERROR " + traceback.format_exc())
        finally:
            try:
                researcher.finish()
            except Exception as exc:
                report.status, report.error = "error", f"finish failed: {type(exc).__name__}: {exc}"
                researcher.log("ERROR in finish " + traceback.format_exc())
            report.write(work)
            researcher.log(f"done: status={report.status}, best={report.best_metric}, submitted={report.submitted_experiment_id}")
    except Exception as exc:
        report.status, report.error = "error", f"{type(exc).__name__}: {exc}"
        log("[agent] ERROR before research could start: " + traceback.format_exc())
    report.elapsed_s = time.time() - start
    return report
