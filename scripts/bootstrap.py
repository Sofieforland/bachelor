import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score


def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    acc_scores = []
    sens_scores = []
    spec_scores = []
    ba_scores = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, n, replace=True)

        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]

        acc_scores.append(accuracy_score(y_true_sample, y_pred_sample))
        sens_scores.append(recall_score(y_true_sample, y_pred_sample, zero_division=0))
        spec_scores.append(recall_score(y_true_sample, y_pred_sample, pos_label=0, zero_division=0))
        ba_scores.append(balanced_accuracy_score(y_true_sample, y_pred_sample))

    def ci(x):
        return np.percentile(x, [2.5, 97.5])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "accuracy_ci": ci(acc_scores),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "sensitivity_ci": ci(sens_scores),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "specificity_ci": ci(spec_scores),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "balanced_accuracy_ci": ci(ba_scores),
    }

# 1. Les CSV med GP_fasit
marksheet = pd.read_csv("bachelor/outputs/filtered_pasients.csv")

# sørg for samme format
marksheet["patient_ID"] = marksheet["patient_ID"].astype(str).str.strip()
marksheet["GP_fasit"] = marksheet["GP_fasit"].astype(str).str.strip().str.upper()

# 2. Les JSONL med modellresultater
rows = []
skipped_empty = 0
with open("bachelor/outputs/GP_caution/dataset_with_llama_outputs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        patient_id = str(obj["patient_ID"]).strip()
        decision = obj.get("doctors", {}).get("cautious_gp", {}).get("decision") #

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

# 5. Metrics
results = bootstrap_metrics(y_true, y_pred)
print("skipped empty lines in JSONL:", skipped_empty)
print(results)
