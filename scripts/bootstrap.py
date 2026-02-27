import numpy as np
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score
import pandas as pd

def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000):
    n = len(y_true)
    
    acc_scores = []
    sens_scores = []
    spec_scores = []
    ba_scores = []

    for _ in range(n_bootstrap):
        # trekk indekser med tilbakelegging
        indices = np.random.choice(n, n, replace=True)
        
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        
        # metrics
        acc = accuracy_score(y_true_sample, y_pred_sample)
        sens = recall_score(y_true_sample, y_pred_sample)  # sensitivity (TPR)
        spec = recall_score(y_true_sample, y_pred_sample, pos_label=0)
        ba = balanced_accuracy_score(y_true_sample, y_pred_sample)
        
        acc_scores.append(acc)
        sens_scores.append(sens)
        spec_scores.append(spec)
        ba_scores.append(ba)

    def ci(x):
        return np.percentile(x, [2.5, 97.5])

    return {
        "accuracy_mean": np.mean(acc_scores),
        "accuracy_ci": ci(acc_scores),
        "sensitivity_mean": np.mean(sens_scores),
        "sensitivity_ci": ci(sens_scores),
        "specificity_mean": np.mean(spec_scores),
        "specificity_ci": ci(spec_scores),
        "balanced_accuracy_mean": np.mean(ba_scores),
        "balanced_accuracy_ci": ci(ba_scores),
    }

marksheet = pd.read_csv("code/marksheet.csv") #Mest sannsynlig feil fil, ettersom detter er resultat etter bipsy osv
model_results = pd.read_csv("outputs/dataset_with_qwen_outputs.csv")

# gjør ID-kolonnen lik (marksheet har patient_id)
marksheet = marksheet.rename(columns={"patient_id": "patient_ID"})

# (valgfritt, men lurt) sørg for samme datatype og ingen whitespace
marksheet["patient_ID"] = marksheet["patient_ID"].astype(str).str.strip()
model_results["patient_ID"] = model_results["patient_ID"].astype(str).str.strip()

# MERGE med en DataFrame, ikke liste
df = marksheet.merge(
    model_results[["patient_ID", "Chief_FINAL_DECISION"]],
    on="patient_ID",
    how="inner"
)

print("Merged rows:", len(df))
print(df[["patient_ID", "case_csPCa", "Chief_FINAL_DECISION"]].head())

y_true = df["case_csPCa"].map({"YES": 1, "NO": 0}).values
y_pred = df["Chief_FINAL_DECISION"].map({"YES": 1, "NO": 0}).values

results = bootstrap_metrics(y_true, y_pred)
print(results)