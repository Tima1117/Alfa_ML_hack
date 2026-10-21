"""End-to-end runs of the research loop on the tiny synthetic datasets (conftest.py)."""

import json
import time

import pandas as pd

from agent.llm import LLMClient, scripted_transport
from agent.research import ResearchSettings, run_research


def _task_md(paths, text):
    task_md = paths["train"].parent / "task.md"
    task_md.write_text(text, encoding="utf-8")
    return task_md


def _settings():
    return ResearchSettings(max_rounds=4, hypotheses_per_round=3, n_jobs=1, safety_margin_s=5)


def test_research_without_llm_writes_a_valid_submission_and_report(regression_dataset, tmp_path, monkeypatch):
    task_md = _task_md(regression_dataset, "# Задача\nМетрика оценки — **MAE**.\n")
    monkeypatch.chdir(tmp_path)  # a relative work_dir must survive the subprocess worker's other cwd
    report = run_research(task_md, time.time() + 300, llm=None, work_dir="work", settings=_settings())

    assert report.status == "ok", report.error
    assert report.metric_name == "mae" and report.metric_source == "task.md"
    sub = pd.read_csv(report.submission_path)
    test = pd.read_csv(regression_dataset["test"])
    assert list(sub.columns) == ["id", "target"] and sub["id"].tolist() == test["id"].tolist()
    assert report.submitted_experiment_id == report.best_experiment_id
    assert {j["source"] for j in report.journal} <= {"seed", "playbook", "local-search"}
    assert {"seed", "playbook"} <= {j["source"] for j in report.journal}
    assert (tmp_path / "work" / "report.md").is_file()


def test_research_with_scripted_llm(classification_dataset, tmp_path):
    task_md = _task_md(classification_dataset, "# Задача\nОсновная метрика — Macro F1.\n")
    feature_code = "import numpy as np\ndef make_features(df):\n    df['x_sum'] = df['x1'] + df['x2']\n    return df\n"
    answers = [
        json.dumps({"metric_name": "accuracy", "summary": "x1+x2 drives the label", "columns_to_drop": ["nope"]}),
        "<think>let me think</think>" + json.dumps({
            "analysis": "trees are fine, try the obvious interaction",
            "hypotheses": [
                {"title": "sum feature", "kind": "features", "feature_code": feature_code},
                {"title": "hallucinated model", "model_name": "super_net"},
                {"title": "bad code", "feature_code": "import os\ndef make_features(df):\n    return df"},
                {"title": "sneaky step", "add_feature_steps": ["py:/etc/hostname"]},
            ],
        }),
        json.dumps({"analysis": "done", "hypotheses": [], "stop": True}),
    ]
    llm = LLMClient(scripted_transport(answers), model="scripted")
    settings = _settings()
    settings.max_rounds = 2  # round 1: LLM ideas; round 2: LLM stops -> one playbook round
    settings.hypotheses_per_round = 4
    report = run_research(task_md, time.time() + 300, llm=llm, work_dir=tmp_path / "work", settings=settings)

    assert report.status == "ok", report.error
    # task.md says macro F1; the LLM's "accuracy" must not override it
    assert report.metric_name == "macro_f1"
    statuses = {j["title"]: j["status"] for j in report.journal}
    assert statuses["sum feature"] == "success"
    assert statuses["hallucinated model"] == "invalid"
    assert statuses["bad code"] == "rejected"
    assert statuses["sneaky step"] == "invalid"  # only sandbox-accepted code may run as a py: step
    assert report.llm_usage["calls"] == 3
    assert report.understanding == "x1+x2 drives the label"
    assert any(j["source"] == "playbook" for j in report.journal)  # after the LLM stopped
    assert pd.read_csv(report.submission_path)["label"].isin(["Yes", "Maybe", "No"]).all()


def test_research_survives_a_broken_llm_and_a_tight_deadline(regression_dataset, tmp_path):
    def broken(messages):
        raise TimeoutError("LLM endpoint unreachable")

    task_md = _task_md(regression_dataset, "RMSE")
    report = run_research(
        task_md, time.time() + 300, llm=LLMClient(broken, max_consecutive_errors=1), work_dir=tmp_path / "w1",
        settings=_settings(),
    )
    assert report.status == "ok" and report.llm_mode.startswith("disabled")
    assert report.best_metric is not None

    # deadline already passed: only the fallback submission, still status ok and a valid file
    report = run_research(task_md, time.time() - 1, llm=None, work_dir=tmp_path / "w2", out_path=tmp_path / "late.csv")
    assert report.submitted_experiment_id == "fallback"
    assert len(pd.read_csv(tmp_path / "late.csv")) == len(pd.read_csv(regression_dataset["test"]))


def test_llm_gets_another_turn_after_the_playbook(regression_dataset, tmp_path):
    task_md = _task_md(regression_dataset, "RMSE")
    stop = json.dumps({"analysis": "nothing to add", "hypotheses": [], "stop": True})
    llm = LLMClient(scripted_transport([json.dumps({"metric_name": "rmse"}), stop, stop]), model="scripted")
    settings = _settings()
    settings.max_rounds, settings.llm_rest_rounds = 2, 1  # round 1: LLM stops -> playbook; round 2: LLM again
    logs: list[str] = []
    report = run_research(task_md, time.time() + 300, llm=llm, work_dir=tmp_path / "work", settings=settings, log=logs.append)

    assert report.status == "ok", report.error
    assert report.llm_usage["calls"] == 3
    assert any("back to the LLM" in line for line in logs)
