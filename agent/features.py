"""Sandbox for LLM-written feature code.

The LLM may propose `def make_features(df) -> df`. Before ml_core ever runs
it (as a `py:<file>` feature step, inside the CV loop), the code must pass:

1. a static check: parses, defines make_features, imports only from an
   allowlist (numpy/pandas/math/…), no file/process/network access, no
   exec/eval/open, no dunder attribute tricks;
2. a dynamic check in a separate, isolated Python process with a timeout,
   on the real train and test feature frames (target and id removed, so the
   code cannot see the label at all): same row count, deterministic
   (two runs on a sample give identical output), same output columns for
   train and test, new columns not empty/constant, bounded column count.

This is a guard against mistakes and hallucinations of a cooperative model,
not a security boundary against a hostile one.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ALLOWED_IMPORTS = {"numpy", "pandas", "math", "re", "datetime", "itertools", "functools", "collections", "statistics"}
FORBIDDEN_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input", "breakpoint",
    "exit", "quit", "help", "memoryview", "setattr", "delattr",
}
FORBIDDEN_ATTRS = {"to_csv", "to_pickle", "to_parquet", "to_excel", "to_json", "to_sql", "read_csv", "read_pickle",
                   "read_parquet", "system", "popen", "load", "save", "savez", "fromfile", "tofile"}
MAX_NEW_COLUMNS = 40
SAMPLE_ROWS = 3000

_RUNNER = Path(__file__).with_name("_feature_sandbox_runner.py")


@dataclass
class FeatureCheck:
    ok: bool
    problems: list[str] = field(default_factory=list)
    new_columns: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    seconds: float = 0.0
    step: str | None = None  # "py:<path>" when ok

    def summary(self) -> str:
        if self.ok:
            return f"feature code OK: +{self.new_columns} -{self.dropped_columns} ({self.seconds:.1f}s)"
        return "feature code rejected: " + "; ".join(self.problems)


def static_check(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    problems: list[str] = []
    has_fn = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "make_features":
            has_fn = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            problems.append(f"call to {node.func.id}() is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS:
                problems.append(f"attribute .{node.attr} is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("global/nonlocal state is not allowed (make_features must be stateless)")
    if not has_fn:
        problems.append("code must define `def make_features(df):` returning a DataFrame")
    return sorted(set(problems))


class FeatureSandbox:
    """Holds the feature-only train/test frames once per task and checks code against them."""

    def __init__(self, work_dir: str | Path, train: pd.DataFrame, test: pd.DataFrame, timeout_s: float = 180.0) -> None:
        self.dir = Path(work_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self._train_path = self.dir / "train_features.pkl"
        self._test_path = self.dir / "test_features.pkl"
        train.to_pickle(self._train_path)
        test.to_pickle(self._test_path)

    @classmethod
    def from_csv(
        cls, work_dir: str | Path, train_csv: str | Path, test_csv: str | Path, feature_columns: list[str], timeout_s: float = 180.0
    ) -> "FeatureSandbox":
        """feature_columns: exactly the columns ml_core passes to feature steps (never id or target)."""
        train = pd.read_csv(train_csv, usecols=feature_columns)[feature_columns]
        test = pd.read_csv(test_csv, usecols=feature_columns)[feature_columns]
        return cls(work_dir, train, test, timeout_s)

    def check(self, code: str, title: str = "") -> FeatureCheck:
        problems = static_check(code)
        if problems:
            return FeatureCheck(ok=False, problems=problems)

        digest = hashlib.sha256(code.encode()).hexdigest()[:12]
        code_path = self.dir / f"feat_{digest}.py"
        header = f"# generated feature step: {title}\n".replace("\r", " ") if title else ""
        code_path.write_text(header + code, encoding="utf-8")
        report_path = self.dir / f"feat_{digest}.check.json"
        cmd = [
            sys.executable, "-I", str(_RUNNER), str(code_path), str(self._train_path), str(self._test_path),
            str(report_path), str(SAMPLE_ROWS), str(MAX_NEW_COLUMNS),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return FeatureCheck(ok=False, problems=[f"feature code took longer than {self.timeout_s:.0f}s on the full data"])
        if not report_path.is_file():
            tail = (proc.stderr or proc.stdout or "")[-800:]
            return FeatureCheck(ok=False, problems=[f"sandbox crashed: {tail}"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        check = FeatureCheck(
            ok=not report["problems"],
            problems=report["problems"],
            new_columns=report.get("new_columns", []),
            dropped_columns=report.get("dropped_columns", []),
            seconds=report.get("seconds", 0.0),
        )
        if check.ok:
            check.step = f"py:{code_path.resolve()}"
        return check
