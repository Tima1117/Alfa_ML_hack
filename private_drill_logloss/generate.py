"""Generates this drill's train.csv/test.csv from a known logistic model (seed below is fixed,
so re-running reproduces byte-identical data). Not run by tests or the agent itself — a one-off
used to build the fixture and to recompute the Bayes-optimal ceiling in README.md.

Deliberately different from private_drill/ in the two axes that drill didn't cover:
  - metric: log_loss, not roc_auc (still a probability metric, but greater_is_better=False,
    and ml_core's submission-value check differs from roc_auc's by nothing structural today —
    this drill is partly here to confirm that's actually true end to end, not just on paper).
  - validation: NO time or group column at all (plain i.i.d. rows, random train/test split) —
    forces agent/understand.py's decide_validation() down the stratified_kfold path instead of
    time_split, which private_drill/ never exercised.

Run: python private_drill_logloss/generate.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 4242
rng = np.random.default_rng(SEED)

N_TRAIN = 3500
N_TEST = 1500
N = N_TRAIN + N_TEST


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


# --- features (all i.i.d. across the whole pool; no column distinguishes train from test) ---
tenure_months = rng.integers(1, 73, size=N)
monthly_charges = np.clip(rng.normal(70, 28, size=N), 15, 180)
support_tickets_12m = rng.poisson(1.6, size=N)
usage_score = np.clip(rng.normal(55, 22, size=N), 0, 100)
contract_type = rng.choice(["Monthly", "One-year", "Two-year"], size=N, p=[0.5, 0.3, 0.2])
payment_method = rng.choice(["Card", "Bank transfer", "Mobile wallet"], size=N, p=[0.45, 0.35, 0.2])

contract_effect = np.select(
    [contract_type == "Monthly", contract_type == "One-year", contract_type == "Two-year"],
    [0.9, -0.1, -0.8],
)
payment_effect = np.select(
    [payment_method == "Card", payment_method == "Bank transfer", payment_method == "Mobile wallet"],
    [0.0, -0.15, 0.35],
)

logit = (
    -0.55
    - 1.1 * z(tenure_months)
    + 0.55 * z(monthly_charges)
    + 0.6 * z(support_tickets_12m)
    - 0.5 * z(usage_score)
    + contract_effect
    + payment_effect
)
noise = rng.normal(0, 0.8, size=N)
p_true = sigmoid(logit + noise)
y = rng.binomial(1, p_true)
churned = np.where(y == 1, "Yes", "No")

ids = np.arange(1, N + 1)
order = rng.permutation(N)  # i.i.d. shuffle: no column may ever carry a train/test ordering signal

df = pd.DataFrame({
    "customer_id": ids,
    "tenure_months": tenure_months,
    "monthly_charges": np.round(monthly_charges, 2),
    "support_tickets_12m": support_tickets_12m,
    "usage_score": np.round(usage_score, 1),
    "contract_type": contract_type,
    "payment_method": payment_method,
    "churned": churned,
}).iloc[order].reset_index(drop=True)
df["customer_id"] = ids  # re-sequence after shuffling rows, id itself carries no signal

p_true_shuffled = p_true[order]

train = df.iloc[:N_TRAIN].reset_index(drop=True)
test = df.iloc[N_TRAIN:].reset_index(drop=True).drop(columns=["churned"])
p_test = p_true_shuffled[N_TRAIN:]

train.to_csv("private_drill_logloss/train.csv", index=False)
test.to_csv("private_drill_logloss/test.csv", index=False)
pd.DataFrame({"customer_id": test["customer_id"], "churned": 0.5}).to_csv(
    "private_drill_logloss/sample_submission.csv", index=False
)

# Bayes-optimal ceiling: mean binary entropy of the true per-row probability (same convention
# as private_drill/README.md's ROC-AUC ceiling) — the irreducible noise floor if a model somehow
# knew each row's exact generating p_i, which no feature-only model actually can (p_i includes a
# per-row noise draw with no corresponding feature); a real model's best achievable log_loss is
# necessarily somewhat above this number, never below it.
eps = 1e-12
ceiling = float(np.mean(-(p_test * np.log(p_test + eps) + (1 - p_test) * np.log(1 - p_test + eps))))
print(f"train rows: {len(train)}, test rows: {len(test)}")
print(f"train positive rate: {train['churned'].eq('Yes').mean():.3f}")
print(f"Bayes-optimal log_loss ceiling (test rows): {ceiling:.4f}")
y_test = test_churned_true = y[order][N_TRAIN:]
pd.DataFrame({"customer_id": test["customer_id"], "p_true": p_test, "churned_true": y_test}).to_csv(
    "private_drill_logloss/_hidden_test_labels.csv", index=False
)  # not committed (gitignored below); for scoring the agent's real submission against ground truth
