"""Integration checks against a flat archive, with controlled process failures."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import zipfile

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("infra_builder", ROOT / "scripts/build_submission.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture
def packaged(tmp_path):
    archive = builder.build(tmp_path / "solution.zip")
    home = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == ["bundle_sources.py", "requirements.txt", "solution.py"]
        z.extractall(home)
    task = tmp_path / "task with spaces"
    task.mkdir()
    (task / "task.md").write_text("Regression. RMSE. train.csv test.csv data_dictionary.csv sample_submission.csv")
    (task / "train.csv").write_text("id,x,y\n1,2,3\n2,3,4\n")
    (task / "test.csv").write_text("id,x\n3,4\n4,5\n")
    (task / "sample_submission.csv").write_text("id,y\n3,0\n4,0\n")
    (task / "data_dictionary.csv").write_text("feature,role,dtype\nid,id,int\nx,feature,float\ny,target,float\n")
    return home, task


def fake_agent(home, behavior):
    # Override only the bundled payload in this isolated test, not production sources.
    source = """
from pathlib import Path
import os, subprocess, sys, time
from types import SimpleNamespace
def run_research(task, **kwargs):
    home = Path(task).parent
    (home / 'submission.csv').write_text('id,y\\n3,3.5\\n4,3.5\\n')
    mode = BEHAVIOR
    if mode in ('hang', 'signal'):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(90)'], start_new_session=True)
        (home / 'worker_pid').write_text(str(child.pid))
        time.sleep(90)
    if mode == 'spawn':
        pid = os.posix_spawn(sys.executable, [sys.executable, '-c', 'import time; time.sleep(90)'], os.environ)
        (home / 'worker_pid').write_text(str(pid))
        time.sleep(90)
    if mode == 'memory':
        data = bytearray(150 * 1024 * 1024)
        time.sleep(90)
    if mode == 'crash':
        raise RuntimeError('simulated agent failure')
    return SimpleNamespace(status='ok')
""".replace("BEHAVIOR", repr(behavior))
    sources = {
        "agent/__init__.py": "",
        "agent/research.py": source,
        "agent/task_files.py": (ROOT / "agent/task_files.py").read_text(encoding="utf-8"),
        "ml_core/__init__.py": "",
        "ml_core/profiling.py": "from types import SimpleNamespace\ndef infer_task_spec(*args):\n return SimpleNamespace(model_dump_json=lambda: '{\"task_type\":\"regression\"}')\n",
    }
    for path in (ROOT / "infra/runtime").glob("*.py"):
        sources[path.name] = path.read_text(encoding="utf-8")
    import hashlib
    hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()}
    (home / "bundle_sources.py").write_text("SOURCES = " + repr(sources) + "\nSOURCE_SHA256 = " + repr(hashes), encoding="utf-8")


def launch(home, task, **extra):
    env = {**os.environ, "INFRA_TOTAL_SECONDS": "10", "AGENT_N_JOBS": "1", **extra}
    return subprocess.Popen([sys.executable, str(home / "solution.py"), str(task / "task.md")],
                            cwd=home.parent, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(proc, task):
    stdout, stderr = proc.communicate(timeout=20)
    assert proc.returncode == 0, (stdout, stderr)
    summary = json.loads((task / "agent_work/infra_report.json").read_text())
    assert summary["submission_valid"] and summary["submission_rows"] == 2
    return summary


def test_build_reproducible_and_sources_unchanged(tmp_path):
    first = builder.build(tmp_path / "first.zip")
    second = builder.build(tmp_path / "second.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as z:
        scope = {}
        exec(z.read("bundle_sources.py"), scope)
        for name, source in scope["SOURCES"].items():
            path = ROOT / name if "/" in name else ROOT / "infra/runtime" / name
            assert source == path.read_text(encoding="utf-8")


def test_source_snapshot_builds_without_git(tmp_path, monkeypatch):
    snapshot = tmp_path / "source_snapshot"
    for name in ["agent", "ml_core", "infra"]:
        shutil.copytree(ROOT / name, snapshot / name)
    (snapshot / "Base").mkdir()
    shutil.copy2(ROOT / "Base/requirements.txt", snapshot / "Base/requirements.txt")
    monkeypatch.setattr(builder, "ROOT", snapshot)
    archive = builder.build(tmp_path / "snapshot.zip")
    assert archive.is_file()
    manifest = json.loads(archive.with_suffix(".manifest.json").read_text())
    assert manifest["base_revision"] is None


def test_platform_copies_only_root_python_and_requirements(packaged, tmp_path):
    home, task = packaged
    verify_spec = importlib.util.spec_from_file_location("infra_verify", ROOT / "scripts/verify_submission.py")
    verify = importlib.util.module_from_spec(verify_spec)
    verify_spec.loader.exec_module(verify)
    archive = tmp_path / "filtered.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in ["solution.py", "bundle_sources.py", "requirements.txt"]:
            z.writestr(name, (home / name).read_bytes())
        z.writestr("agent/research.py", "raise RuntimeError('must not be extracted')")
        z.writestr("ml_core/contracts.py", "raise RuntimeError('must not be extracted')")
        z.writestr("README.md", "must not be copied")
    copied = verify.copy_platform_files(archive, task)
    assert set(copied) == {"solution.py", "bundle_sources.py", "requirements.txt"}
    assert not (task / "agent").exists() and not (task / "ml_core").exists()
    assert not (task / "README.md").exists()
    fake_agent(task, "ok")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("OPENAI_")}
    env.update(INFRA_TOTAL_SECONDS="10", AGENT_N_JOBS="1", AGENT_DISABLE_LLM="1")
    proc = subprocess.Popen([sys.executable, "solution.py", "task.md"], cwd=task,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert finish(proc, task)["reason"] == "completed"


def test_restored_feature_sandbox_runner_is_found(packaged, tmp_path):
    home, task = packaged
    code = """
from pathlib import Path
import sys
home, task = map(Path, sys.argv[1:])
sys.path.insert(0, str(home))
import solution
work = task / 'sandbox_runtime'
work.mkdir()
runtime = solution.install_runtime(work)
sys.path.insert(0, str(runtime))
from agent.features import FeatureSandbox
sandbox = FeatureSandbox.from_csv(task / 'features', task / 'train.csv', task / 'test.csv', ['x'])
result = sandbox.check('def make_features(df):\\n    df["x_sq"] = df["x"] ** 2\\n    return df\\n', 'square')
assert result.ok, result.problems
assert 'x_sq' in result.new_columns
"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("OPENAI_")}
    result = subprocess.run([sys.executable, "-c", code, str(home), str(task)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("behavior", ["ok", "crash"])
def test_valid_submission_survives_agent_exit(packaged, behavior):
    home, task = packaged
    fake_agent(home, behavior)
    summary = finish(launch(home, task), task)
    assert summary["reason"] == "completed"
    assert summary["child_exit_code"] == (0 if behavior == "ok" else 1)


@pytest.mark.parametrize("behavior", ["hang", "spawn"])
def test_global_timeout_cleans_detached_workers(packaged, behavior):
    home, task = packaged
    fake_agent(home, behavior)
    summary = finish(launch(home, task), task)
    assert summary["reason"] == "deadline"
    pid = int((task / "worker_pid").read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_memory_stop_preserves_submission(packaged):
    home, task = packaged
    fake_agent(home, "memory")
    summary = finish(launch(home, task, INFRA_MEMORY_GB="0.08"), task)
    assert summary["reason"] == "memory_limit"


def test_signal_cleans_workers_and_preserves_submission(packaged):
    home, task = packaged
    fake_agent(home, "signal")
    proc = launch(home, task)
    until = time.monotonic() + 5
    while not (task / "worker_pid").exists():
        assert time.monotonic() < until
        time.sleep(0.05)
    time.sleep(0.3)  # allow observer to register worker identity
    proc.send_signal(signal.SIGTERM)
    assert finish(proc, task)["reason"] == "signal"
    pid = int((task / "worker_pid").read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("csv_text", ["id,y\n4,3\n3,3\n", "id,y\n3,nan\n4,3\n", "id,y\n3,3\n", "id,y\n3,3\n3,4\n"])
def test_final_check_rejects_bad_submission(packaged, csv_text):
    home, task = packaged
    entry_spec = importlib.util.spec_from_file_location("infra_entry", home / "solution.py")
    entry = importlib.util.module_from_spec(entry_spec)
    entry_spec.loader.exec_module(entry)
    (task / "submission.csv").write_text(csv_text)
    with pytest.raises(ValueError):
        entry.check_submission(task / "task.md", task / "submission.csv")
