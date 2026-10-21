"""Find a task's CSV files from its task.md.

task.md lists the data paths relative to its own directory (e.g.
`regression/participants/train.csv`), but the layout is not guaranteed — in
this repo the same files sit right next to task.md. So: try every path that
task.md mentions first, then fall back to searching the task directory tree
by file name, preferring the folder that holds train.csv.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_CSV_PATH_RE = re.compile(r"[\w./\\-]+\.csv")
_ROLES = {
    "train": "train.csv",
    "test": "test.csv",
    "data_dictionary": "data_dictionary.csv",
    "sample_submission": "sample_submission.csv",
}
_MAX_SEARCH_DEPTH = 4


class TaskFilesError(Exception):
    pass


@dataclass(frozen=True)
class TaskFiles:
    task_md: Path
    task_dir: Path
    train: Path
    test: Path
    data_dictionary: Path
    sample_submission: Path

    @property
    def default_submission_path(self) -> Path:
        return self.task_dir / "submission.csv"


def _mentioned_paths(text: str, task_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for raw in _CSV_PATH_RE.findall(text):
        path = (task_dir / raw.replace("\\", "/")).resolve()
        for role, name in _ROLES.items():
            if path.name == name and role not in found and path.is_file():
                found[role] = path
    return found


def _search(task_dir: Path, name: str) -> list[Path]:
    hits = []
    for path in task_dir.rglob(name):
        try:
            depth = len(path.relative_to(task_dir).parts)
        except ValueError:
            continue
        if depth <= _MAX_SEARCH_DEPTH and path.is_file():
            hits.append(path.resolve())
    return sorted(hits, key=lambda p: (len(p.parts), str(p)))


def locate_task_files(task_md: str | Path) -> TaskFiles:
    task_md = Path(task_md).resolve()
    task_dir = task_md.parent
    text = task_md.read_text(encoding="utf-8") if task_md.is_file() else ""
    found = _mentioned_paths(text, task_dir)

    if "train" not in found:
        hits = _search(task_dir, _ROLES["train"])
        if not hits:
            raise TaskFilesError(f"train.csv not found next to or below {task_dir}")
        found["train"] = hits[0]
    home = found["train"].parent

    for role, name in _ROLES.items():
        if role in found:
            continue
        sibling = home / name
        if sibling.is_file():
            found[role] = sibling.resolve()
            continue
        hits = _search(task_dir, name)
        if not hits:
            raise TaskFilesError(f"{name} not found next to or below {task_dir}")
        found[role] = hits[0]

    return TaskFiles(task_md=task_md, task_dir=task_dir, **found)


def read_task_text(task_md: str | Path, max_chars: int = 12000) -> str:
    path = Path(task_md)
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    return text if len(text) <= max_chars else text[:max_chars] + "\n…(truncated)"
