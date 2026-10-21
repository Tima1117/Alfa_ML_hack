# Synthetic "private-like" task — a drill, not a real competition dataset

Generated locally (not from the organizers) to stress-test the pipeline against
something structurally unlike the two known public tasks, since the brief warns
private-task metrics "may differ". Same guaranteed file shapes
(`train.csv`/`test.csv`/`data_dictionary.csv`/`sample_submission.csv`/`task.md`),
different domain (loan-default prediction, not buildings or network traffic),
and deliberately probes known risk areas:

- **Metric stated as ROC-AUC** in `task.md` — a probability-scored metric, unlike
  both real public tasks (RMSE, Macro F1), which only ever need hard predictions.
- **Severe class imbalance** (~5% positive) — this is what actually surfaced the
  finding below; a balanced synthetic target wouldn't have.
- A disguised time column (`application_batch`: train has 1-6, test is only 7) and
  an unseen-at-train-time category (`region_code` has `R90`/`R91` only in test) —
  the same shape of risk as `Year_Factor`/`state` in the real public tasks, just
  with different names, to confirm nothing is hardcoded to those specific columns.

## What this drill actually found

Running it (before the fix in the same commit range) showed `ml_core` always
calling `model.predict()` for classification, never `model.predict_proba()`.
Under this fixture's imbalance, that collapsed to predicting the majority class
for every single row — a valid *label*, but a submission with **zero
discriminative power** for ROC-AUC (≈0.5, random), regardless of how good the
underlying model actually was. Confirmed by the submission literally being
100% "No" before the fix, and a real spread of 1500 distinct probabilities
(ROC-AUC ≈ 0.70 on this fixture's own validation) after it. See
`docs/CONTRACTS.md`'s "Probability-scored metrics" section for the fix itself.

## Re-running it

```bash
python scripts/run_agent_local.py private_drill/task.md --budget-s 180 --no-llm
```

Drop `--no-llm` (and set `OPENAI_API_KEY`/`OPENAI_MODEL`/`OPENAI_BASE_URL`) to also
exercise the LLM's own hypothesis generation against a metric/domain it hasn't
seen tuned prompts for. Output goes to `runs/agent/private_drill/` by default —
gitignored, regenerate rather than relying on a stale copy.
