"""Exercise the actual ZIP with clean task files; never uses paid LLM calls."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def copy_platform_files(archive, destination):
    """Emulate the platform: ignore nested paths and non-Python auxiliary files."""
    copied = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or "/" in name or "\\" in name:
                continue
            if not (name.endswith(".py") or name == "requirements.txt"):
                continue
            if name in copied:
                raise ValueError(f"Duplicate root file: {name}")
            (destination / name).write_bytes(z.read(info))
            copied.append(name)
    if "solution.py" not in copied:
        raise ValueError("No root solution.py in submission ZIP")
    return copied


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("task", type=Path)
    p.add_argument("output", type=Path, help="New, empty verification directory")
    p.add_argument("--budget-s", type=int, default=300)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--platform-copy", action="store_true",
                   help="Copy only root .py/requirements.txt into task dir; run exactly python solution.py task.md")
    args = p.parse_args()
    home = args.output.resolve()
    home.mkdir(parents=True, exist_ok=False)
    package, task = home / "package", home / "task"
    package.mkdir()
    task.mkdir()
    if args.platform_copy:
        copied = copy_platform_files(args.archive, task)
        print("Platform copied: " + ", ".join(copied), flush=True)
    else:
        with zipfile.ZipFile(args.archive) as z:
            for name in z.namelist():
                if Path(name).name != name or name not in {"solution.py", "bundle_sources.py", "requirements.txt"}:
                    raise ValueError(f"Unexpected ZIP member: {name}")
            z.extractall(package)
    shutil.copy2(args.task, task / "task.md")
    # Task paths may be nested; preserve the layout but omit old generated outputs.
    for source in args.task.parent.rglob("*.csv"):
        if source.name not in {"train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"}:
            continue
        if "agent_work" in source.relative_to(args.task.parent).parts:
            continue
        dest = task / source.relative_to(args.task.parent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_") and k != "PYTHONPATH"}
    env.update(AGENT_DISABLE_LLM="1", INFRA_TOTAL_SECONDS=str(args.budget_s), AGENT_N_JOBS=str(args.jobs))
    command = ([sys.executable, "solution.py", "task.md"] if args.platform_copy else
               [sys.executable, str(package / "solution.py"), str(task / "task.md")])
    result = subprocess.run(command, cwd=task if args.platform_copy else home,
                            env=env, timeout=args.budget_s + 30)
    report = json.loads((task / "agent_work/infra_report.json").read_text())
    print(json.dumps(report, indent=2))
    if result.returncode or not report["submission_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
