import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score


def summarize_scores(x):
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ci": np.percentile(x, [2.5, 97.5]),
    }


def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, random_state=42, stratify=True):
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    acc_scores = []
    sens_scores = []
    spec_scores = []
    ba_scores = []

    if stratify:
        idx_pos = np.where(y_true == 1)[0]
        idx_neg = np.where(y_true == 0)[0]

        n_pos = len(idx_pos)
        n_neg = len(idx_neg)

        if n_pos == 0 or n_neg == 0:
            raise ValueError("Stratified bootstrap krever minst én observasjon i hver klasse.")

    for _ in range(n_bootstrap):
        if stratify:
            sample_pos = rng.choice(idx_pos, size=n_pos, replace=True)
            sample_neg = rng.choice(idx_neg, size=n_neg, replace=True)
            indices = np.concatenate([sample_pos, sample_neg])
            rng.shuffle(indices)
        else:
            indices = rng.choice(n, n, replace=True)

        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]

        acc_scores.append(accuracy_score(y_true_sample, y_pred_sample))
        sens_scores.append(recall_score(y_true_sample, y_pred_sample, zero_division=0))
        spec_scores.append(recall_score(y_true_sample, y_pred_sample, pos_label=0, zero_division=0))
        ba_scores.append(balanced_accuracy_score(y_true_sample, y_pred_sample))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_bootstrap": summarize_scores(acc_scores),

        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity_bootstrap": summarize_scores(sens_scores),

        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "specificity_bootstrap": summarize_scores(spec_scores),

        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "balanced_accuracy_bootstrap": summarize_scores(ba_scores),
    }


# 1. Les CSV med GP_fasit
marksheet = pd.read_csv("outputs/filtered_pasients.csv")

# sørg for samme format
marksheet["patient_ID"] = marksheet["patient_ID"].astype(str).str.strip()
marksheet["GP_fasit"] = marksheet["GP_fasit"].astype(str).str.strip().str.upper()

# 2. Les JSONL med modellresultater
rows = []
skipped_empty = 0
with open("outputs/GP_neutral/dataset_with_medgemma_outputs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        patient_id = str(obj["patient_ID"]).strip()
        decision = obj.get("doctors", {}).get("neutral_gp", {}).get("decision")

        if decision is None:
            skipped_empty += 1
            continue

        decision = decision.strip().upper()

        rows.append({
            "patient_ID": patient_id,
            "model_decision": decision
        })

model_results = pd.DataFrame(rows)

# 3. Merge
df = marksheet.merge(
    model_results,
    on="patient_ID",
    how="inner"
)

print("Rows in marksheet:", len(marksheet))
print("Rows in model_results:", len(model_results))
print("Rows after merge:", len(df))

print(df[["patient_ID", "GP_fasit", "model_decision"]].head())

# 4. Map til 0/1
y_true = df["GP_fasit"].map({"YES": 1, "NO": 0})
y_pred = df["model_decision"].map({"YES": 1, "NO": 0})

# fjern eventuelle ugyldige rader
valid = y_true.notna() & y_pred.notna()
df = df[valid].copy()
y_true = y_true[valid].astype(int).values
y_pred = y_pred[valid].astype(int).values

print("Valid rows used:", len(y_true))
print("Positive cases:", np.sum(y_true == 1))
print("Negative cases:", np.sum(y_true == 0))

# 5. Metrics
results = bootstrap_metrics(y_true, y_pred, stratify=True)

print("skipped empty lines in JSONL:", skipped_empty)
print(results)
print(df["model_decision"].value_counts())