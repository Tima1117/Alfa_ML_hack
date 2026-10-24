"""Platform entry point. Python 3.12; no package installation at runtime."""
import time

START_MONO = time.monotonic()
START_WALL = time.time()

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def number(name, default, minimum, maximum):
    raw = os.environ.get(name, str(default))
    value = float(raw)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def merged_pythonpath(existing, runtime):
    """Prepend `runtime` to the child process's PYTHONPATH — never replace it outright.

    Real incident (organizers, 2026-10-24): this used to be a plain `env["PYTHONPATH"] =
    str(runtime)`, which overwrote whatever PYTHONPATH the platform had already set for this
    process — on their platform, that pointed at a pre-installed .vendor directory with
    openai/httpx/etc. The child process could then no longer find those packages at all, so
    every LLM call failed before it could even construct a client — not a network/credentials
    problem, a straightforwardly missing import. Prepending instead keeps our own runtime
    importable first (so agent/ml_core still win any name collision — see child_main()'s own
    sys.modules eviction) while leaving the platform's own path entries intact for the child to
    inherit, exactly like sys.path.insert() already does for this (parent) process.
    """
    return os.pathsep.join(filter(None, [str(runtime), existing]))


def install_runtime(work):
    # Only our build-generated, digest-checked Python sources are unpacked.
    from bundle_sources import SOURCES, SOURCE_SHA256
    import hashlib
    root = Path(tempfile.mkdtemp(prefix="runtime_", dir=work))
    for name, source in SOURCES.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ValueError("Invalid bundled source path")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != SOURCE_SHA256[name]:
            raise ValueError(f"Bundle integrity check failed: {name}")
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return root


def check_submission(task, output):
    """Read-only final check, including exact test order and allowed labels."""
    from agent.task_files import locate_task_files
    files = locate_task_files(task)
    with files.sample_submission.open(encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle))
    with files.train.open(encoding="utf-8-sig", newline="") as handle:
        train_columns = next(csv.reader(handle))
    with files.test.open(encoding="utf-8-sig", newline="") as handle:
        test_columns = next(csv.reader(handle))
    targets = [c for c in columns if c in train_columns and c not in test_columns]
    if len(targets) != 1:
        raise ValueError("Cannot determine the single submission target")
    target = targets[0]
    ids = [c for c in columns if c != target]
    # Match the core's inferred task type; dtype alone is unreliable.
    with files.data_dictionary.open(encoding="utf-8-sig", newline="") as handle:
        dictionary = list(csv.DictReader(handle))
    dtype = next((r.get("dtype", "") for r in dictionary if r.get("feature") == target), "").lower()
    categorical = any(s in dtype for s in ("categor", "string", "object", "binary", "class", "bool"))
    allowed = set()
    with files.train.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            allowed.add(row[target])
    # Numeric class labels are also possible. Core task specification wins when available.
    spec_path = output.parent / "agent_work" / "infra_task_spec.json"
    needs_proba = False
    if spec_path.is_file():
        spec_data = json.loads(spec_path.read_text())
        categorical = spec_data["task_type"] == "classification"
        needs_proba = bool(spec_data.get("needs_proba"))
    if needs_proba:
        # A probability-scored metric (roc_auc, log_loss): the submission value is a
        # float in [0, 1], not one of train's class labels, even though the task itself
        # is classification.
        categorical = False
    seen = set()
    count = 0
    with files.test.open(encoding="utf-8-sig", newline="") as a, output.open(encoding="utf-8-sig", newline="") as b:
        test, result = csv.DictReader(a), csv.DictReader(b)
        if result.fieldnames != columns:
            raise ValueError("Submission columns differ from sample_submission.csv")
        from itertools import zip_longest
        for expected, actual in zip_longest(test, result):
            if expected is None or actual is None:
                raise ValueError("Submission row count differs from test.csv")
            key = tuple(actual[c] for c in ids)
            if not ids or key != tuple(expected[c] for c in ids) or key in seen:
                raise ValueError("Submission ids are missing, duplicated or out of order")
            seen.add(key)
            value = actual.get(target)
            if value is None or value == "" or None in actual:
                raise ValueError("Missing prediction or malformed CSV row")
            if categorical:
                if value not in allowed:
                    raise ValueError("Unknown class in submission")
            elif needs_proba:
                parsed = float(value)
                if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
                    raise ValueError("Predicted probability is not finite or outside [0, 1]")
            elif not math.isfinite(float(value)):
                raise ValueError("Prediction is not finite")
            count += 1
    if count == 0:
        raise ValueError("Empty submission")
    return count


def child_main(args):
    sys.path.insert(0, str(Path(args.runtime).resolve()))
    # Defensive against a name collision: if the platform's own harness (or anything else that
    # runs in this interpreter before us) already imported something called "agent" or
    # "ml_core" — plausible names for either of those on an AI-agent hackathon platform —
    # Python's import system resolves "import agent.research" from sys.modules' cache FIRST,
    # before ever consulting sys.path, so our own sys.path.insert() above would silently never
    # take effect: our bundled code would sit on disk, genuinely different between submissions,
    # while some unrelated already-cached module actually ran instead — and no local test can
    # ever catch this, since nothing in *our* process pre-imports anything by those names.
    # Evicting any such entries forces every subsequent "import agent"/"import ml_core" (direct
    # or via submodule) to resolve fresh against our prepended sys.path[0]. A no-op if there was
    # no collision; cheap enough to always run, so — same principle as the CPU-pinning guard
    # right below — there is no reason not to run it unconditionally, before anything else.
    for _name in list(sys.modules):
        if _name == "agent" or _name.startswith("agent.") or _name == "ml_core" or _name.startswith("ml_core."):
            del sys.modules[_name]
    try:
        if hasattr(os, "sched_getaffinity"):
            available = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, available[:int(os.environ["AGENT_N_JOBS"])])
    except Exception:
        # CPU pinning is an optimization, not a requirement: a hardened multi-tenant sandbox
        # (seccomp/gVisor — plausible for a platform running arbitrary teams' code) can deny
        # this syscall even though the Python binding exists (hasattr is true, the call itself
        # raises). --cpus at the container level and AGENT_N_JOBS already bound actual usage.
        # Real incident: this raised unguarded here, before run_research() ever ran — so its
        # own first action (write_fallback_submission) never happened either, no submission.csv
        # was ever written, and the platform scored the task 0 for a missing file, not a model
        # failure. Whatever happens in this function must never prevent run_research() from
        # being called; see the next try/except too.
        pass
    # Imports stay in the child: startup/model failures cannot crash the observer.
    from agent.research import run_research
    try:
        # Evidence for the sys.modules-collision hypothesis above, in case logs ever become
        # available: where "agent"/"ml_core" (and this file's own __file__) actually resolved
        # from. If these paths are not under args.runtime, something else shadowed our bundle.
        import agent as _agent_mod
        import ml_core as _ml_core_mod
        (Path(args.work) / "infra_import_origins.json").write_text(
            json.dumps({
                "runtime_dir": str(Path(args.runtime).resolve()),
                "agent___file__": getattr(_agent_mod, "__file__", None),
                "ml_core___file__": getattr(_ml_core_mod, "__file__", None),
                "this_file": str(Path(__file__).resolve()),
            }),
            encoding="utf-8",
        )
    except Exception:
        pass
    try:
        from agent.task_files import locate_task_files
        from ml_core.profiling import infer_task_spec

        files = locate_task_files(args.task)
        spec = infer_task_spec(files.train, files.data_dictionary, files.sample_submission)
        (Path(args.work) / "infra_task_spec.json").write_text(spec.model_dump_json(), encoding="utf-8")
    except Exception:
        # Best-effort preliminary hint for check_submission() (lets it treat the target
        # column correctly if run_research() itself never gets to write the real one below).
        # run_research() redoes this same lookup safely inside its own try/except either way —
        # this redundant early copy existing or not must never gate whether it gets called.
        pass
    report = run_research(args.task, deadline_ts=args.deadline, work_dir=args.work)
    # The metric task.md/the LLM actually settles on (report.metric_name) can differ from the
    # preliminary spec above — e.g. a metric needing probabilities (roc_auc, log_loss) changes
    # what a valid submission value looks like. check_submission() (run by the parent after this
    # process exits) reads this updated file, not the preliminary one. Best-effort: a stub/old
    # report shape or a bundle without ml_core.metrics must not crash a run that already has its
    # submission written — check_submission() falls back to its original dtype-based check.
    task_type = getattr(report, "task_type", None)
    metric_name = getattr(report, "metric_name", None)
    needs_proba = False
    try:
        from ml_core.metrics import get_metric

        if metric_name:
            needs_proba = get_metric(metric_name).needs_proba
    except Exception:
        pass
    try:
        (Path(args.work) / "infra_task_spec.json").write_text(
            json.dumps({"task_type": task_type, "metric_name": metric_name, "needs_proba": needs_proba}),
            encoding="utf-8",
        )
    except OSError:
        pass
    return 0 if getattr(report, "status", None) == "ok" else 1


def discover_processes(proc, known, registry):
    import psutil
    if registry.is_file():
        for line in registry.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                known[int(item["pid"])] = float(item["born"])
            except (ValueError, KeyError, TypeError):
                pass  # a final line can be incomplete while another process writes it
    try:
        for child in psutil.Process(proc.pid).children(recursive=True):
            known[child.pid] = child.create_time()
    except (OSError, psutil.Error):
        pass  # restricted macOS: explicit subprocess registration is used instead
    processes = []
    for pid, born in list(known.items()):
        try:
            process = psutil.Process(pid)
            if process.create_time() == born:
                processes.append(process)
        except (OSError, psutil.Error):
            pass
    return processes


def stop_processes(proc, known, registry):
    """Workers create their own sessions, so killing just one group is insufficient."""
    import psutil
    targets = discover_processes(proc, known, registry)
    for process in targets:
        try:
            process.terminate()
        except (OSError, psutil.Error):
            pass
    _, alive = psutil.wait_procs(targets, timeout=2)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=2)
    proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="task.md")
    parser.add_argument("--infra-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runtime", help=argparse.SUPPRESS)
    parser.add_argument("--work", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.infra_child:
        return child_main(args)
    import psutil
    task = Path(args.task).resolve()
    if not task.is_file():
        raise ValueError(f"Task file does not exist: {task}")
    work = task.parent / "agent_work"
    work.mkdir(parents=True, exist_ok=True)
    total = number("INFRA_TOTAL_SECONDS", 10800, 10, 10800)
    reserve = number("INFRA_SAFETY_SECONDS", min(600, total / 5), 1, total / 2)
    stop_at = START_MONO + total - min(30, total / 20)
    research_deadline = START_WALL + total - reserve
    cores = min(8, os.cpu_count() or 1)
    jobs = int(number("AGENT_N_JOBS", cores, 1, cores))
    memory_limit = number("INFRA_MEMORY_GB", 28, 0.05, 28) * 1024 ** 3
    runtime = install_runtime(work)
    sys.path.insert(0, str(runtime))
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = str(jobs)
    env.update(AGENT_N_JOBS=str(jobs), PYTHONUNBUFFERED="1")
    env["MPLCONFIGDIR"] = str(work / "matplotlib")
    env["XDG_CACHE_HOME"] = str(work / "cache")
    env["PYTHONPATH"] = merged_pythonpath(env.get("PYTHONPATH"), runtime)
    registry = work / (runtime.name + "_processes.jsonl")
    env["INFRA_PROCESS_REGISTRY"] = str(registry)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), str(task), "--infra-child",
               "--runtime", str(runtime), "--work", str(work), "--deadline", str(research_deadline)]
    print(f"[infra] start: {jobs} threads, {total:.0f}s total, {reserve:.0f}s reserve, {memory_limit/1024**3:.1f} GiB RSS cap", flush=True)
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    known = {}
    reason = "completed"
    peak = 0
    with (work / "process.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=task.parent, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        known[proc.pid] = psutil.Process(proc.pid).create_time()
        try:
            while proc.poll() is None:
                try:
                    processes = discover_processes(proc, known, registry)
                    rss = psutil.Process().memory_info().rss
                    for process in processes:
                        try:
                            known[process.pid] = process.create_time()
                            rss += process.memory_info().rss
                        except psutil.Error:
                            pass
                    peak = max(peak, rss)
                except (OSError, psutil.Error):
                    rss = 0
                if interrupted:
                    reason = "signal"
                    break
                if rss > memory_limit:
                    reason = "memory_limit"
                    break
                if time.monotonic() >= stop_at:
                    reason = "deadline"
                    break
                time.sleep(0.2)
        finally:
            # Also clean detached workers if the agent exits unexpectedly.
            stop_processes(proc, known, registry)
    output = task.parent / "submission.csv"
    validation_error = None
    rows = 0
    try:
        rows = check_submission(task, output)
    except Exception as exc:
        validation_error = f"{type(exc).__name__}: {exc}"
    summary = dict(reason=reason, child_exit_code=proc.returncode, elapsed_s=time.monotonic()-START_MONO,
                   peak_tree_rss_bytes=peak, threads=jobs, submission_valid=validation_error is None,
                   submission_rows=rows, validation_error=validation_error)
    (work / "infra_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[infra] {reason}; child={proc.returncode}; valid_submission={validation_error is None}; rows={rows}; log={work / 'process.log'}", flush=True)
    if validation_error:
        print(f"[infra] ERROR: {validation_error}", file=sys.stderr)
    return 0 if validation_error is None else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[infra] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
