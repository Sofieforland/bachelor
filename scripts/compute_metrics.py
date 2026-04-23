import json
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

"""
python3 scripts/compute_metrics.py \
  --jsonl outputs/Merged_chief/rep_0/llama_with_doctors_and_chief.jsonl \
  --marksheet outputs/Dataset/filtered_pasients.csv \
  --target-col GP_fasit \
  --output outputs/Metrics/Rep_0/llama_metrics.json
"""


DOCTOR_NAMES = [
    "cautious_gp",
    "conservative_gp",
    "neutral_gp",
    "overconfident_gp",
]


def summarize_scores(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None

    if len(x) == 1:
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "std": 0.0,
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "ci": [float(x[0]), float(x[0])],
        }

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ci": np.percentile(x, [2.5, 97.5]).tolist(),
    }


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_binary_label(x):
    if x is None:
        return None
    x = str(x).strip().upper()
    if x == "YES":
        return 1
    if x == "NO":
        return 0
    return None


def get_prediction(row, evaluator: str):
    if evaluator == "chief":
        return row.get("chief", {}).get("final_decision")
    return row.get("doctors", {}).get(evaluator, {}).get("decision")


def get_probability_yes(row, evaluator: str):
    if evaluator == "chief":
        return row.get("chief", {}).get("final_probability_yes")
    return row.get("doctors", {}).get(evaluator, {}).get("p_yes")


def safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def extract_word_count(text):
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", str(text)))


def extract_bullets_or_sections(text):
    """
    En enkel parser for å hente ut nummererte seksjoner/bullets fra raw-tekst.
    Ikke perfekt, men robust nok til analysebruk.
    """
    if not text:
        return []

    text = str(text).strip()
    if not text:
        return []

    parts = re.split(r"\n\s*(?:\d+[\)\.]|•|-)\s*", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def contradiction_flag(decision, probability_yes):
    """
    Marker motsetning når tekstlig/strukturert beslutning og sannsynlighet peker i hver sin retning.
    """
    decision_bin = to_binary_label(decision)
    p = safe_float(probability_yes)

    if decision_bin is None or p is None:
        return None

    if decision_bin == 1 and p < 0.5:
        return True
    if decision_bin == 0 and p >= 0.5:
        return True
    return False


def compute_ece(y_true, p_yes, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    p_yes = np.asarray(p_yes, dtype=float)

    if len(y_true) == 0:
        return None, []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (p_yes >= left) & (p_yes <= right)
        else:
            mask = (p_yes >= left) & (p_yes < right)

        n_bin = int(np.sum(mask))
        if n_bin == 0:
            rows.append({
                "bin": i,
                "left": float(left),
                "right": float(right),
                "count": 0,
                "mean_confidence": None,
                "empirical_accuracy": None,
                "abs_gap": None,
            })
            continue

        mean_conf = float(np.mean(p_yes[mask]))
        emp_acc = float(np.mean(y_true[mask]))
        gap = abs(mean_conf - emp_acc)
        ece += (n_bin / len(y_true)) * gap

        rows.append({
            "bin": i,
            "left": float(left),
            "right": float(right),
            "count": n_bin,
            "mean_confidence": mean_conf,
            "empirical_accuracy": emp_acc,
            "abs_gap": float(gap),
        })

    return float(ece), rows


def calibration_curve_data(y_true, p_yes, n_bins=10):
    _, rows = compute_ece(y_true, p_yes, n_bins=n_bins)
    return rows


def bootstrap_metrics(y_true, y_pred, p_yes=None, n_bootstrap=1000, random_state=42, stratify=True):
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    acc_scores = []
    sens_scores = []
    spec_scores = []
    ba_scores = []
    f1_scores = []
    auc_scores = []
    brier_scores = []

    if stratify:
        idx_pos = np.where(y_true == 1)[0]
        idx_neg = np.where(y_true == 0)[0]

        if len(idx_pos) == 0 or len(idx_neg) == 0:
            raise ValueError("Stratified bootstrap krever minst én observasjon i hver klasse.")
    else:
        idx_pos = idx_neg = None

    for _ in range(n_bootstrap):
        if stratify:
            sample_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
            sample_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
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
        f1_scores.append(f1_score(y_true_sample, y_pred_sample, zero_division=0))

        if p_yes is not None:
            p_sample = p_yes[indices]
            try:
                if len(np.unique(y_true_sample)) == 2:
                    auc_scores.append(roc_auc_score(y_true_sample, p_sample))
            except Exception:
                pass

            try:
                brier_scores.append(brier_score_loss(y_true_sample, p_sample))
            except Exception:
                pass

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    results = {
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),

        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_bootstrap": summarize_scores(acc_scores),

        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity_bootstrap": summarize_scores(sens_scores),

        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "specificity_bootstrap": summarize_scores(spec_scores),

        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "balanced_accuracy_bootstrap": summarize_scores(ba_scores),

        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_bootstrap": summarize_scores(f1_scores),
    }

    if p_yes is not None:
        try:
            if len(np.unique(y_true)) == 2:
                results["auc"] = float(roc_auc_score(y_true, p_yes))
                results["auc_bootstrap"] = summarize_scores(auc_scores) if len(auc_scores) > 0 else None
            else:
                results["auc"] = None
                results["auc_bootstrap"] = None
        except Exception:
            results["auc"] = None
            results["auc_bootstrap"] = None

        try:
            results["brier_score"] = float(brier_score_loss(y_true, p_yes))
            results["brier_score_bootstrap"] = summarize_scores(brier_scores) if len(brier_scores) > 0 else None
        except Exception:
            results["brier_score"] = None
            results["brier_score_bootstrap"] = None

        ece, ece_bins = compute_ece(y_true, p_yes, n_bins=10)
        results["ece"] = ece
        results["calibration_curve"] = ece_bins
    else:
        results["auc"] = None
        results["auc_bootstrap"] = None
        results["brier_score"] = None
        results["brier_score_bootstrap"] = None
        results["ece"] = None
        results["calibration_curve"] = []

    return results


def build_eval_df(jsonl_path: Path, evaluator: str):
    rows = load_jsonl(jsonl_path)

    extracted = []
    skipped_missing_decision = 0
    skipped_missing_probability = 0

    for row in rows:
        patient_id = str(row["patient_ID"]).strip()

        decision = get_prediction(row, evaluator)
        probability_yes = get_probability_yes(row, evaluator)

        if decision is None:
            skipped_missing_decision += 1
            continue

        decision = str(decision).strip().upper()
        probability_yes = safe_float(probability_yes)

        if probability_yes is None:
            skipped_missing_probability += 1

        extracted.append({
            "patient_ID": patient_id,
            "model_decision": decision,
            "model_p_yes": probability_yes,
        })

    model_df = pd.DataFrame(extracted)
    return model_df, skipped_missing_decision, skipped_missing_probability


def compute_follow_rate(rows, chief_name="chief", panelists=DOCTOR_NAMES):
    """
    % ganger judge matcher panelist i når panelistene er uenige.
    """
    results = {}

    for panelist in panelists:
        match_count = 0
        total_disagreement_cases = 0

        for row in rows:
            chief_decision = row.get("chief", {}).get("final_decision")
            chief_bin = to_binary_label(chief_decision)
            if chief_bin is None:
                continue

            panel_decisions = []
            for p in panelists:
                d = row.get("doctors", {}).get(p, {}).get("decision")
                d_bin = to_binary_label(d)
                if d_bin is not None:
                    panel_decisions.append((p, d_bin))

            if len(panel_decisions) < 2:
                continue

            decision_values = {d for _, d in panel_decisions}
            disagreement = len(decision_values) > 1
            if not disagreement:
                continue

            this_panelist = row.get("doctors", {}).get(panelist, {}).get("decision")
            this_panelist_bin = to_binary_label(this_panelist)
            if this_panelist_bin is None:
                continue

            total_disagreement_cases += 1
            if chief_bin == this_panelist_bin:
                match_count += 1

        results[panelist] = {
            "n_disagreement_cases": int(total_disagreement_cases),
            "n_matches_with_chief": int(match_count),
            "follow_rate": (
                float(match_count / total_disagreement_cases)
                if total_disagreement_cases > 0 else None
            ),
        }

    return results


def compute_probability_closeness(rows, panelists=DOCTOR_NAMES):
    """
    p_judge - p_panelist og absolutt differanse.
    """
    results = {}

    for panelist in panelists:
        signed_diffs = []
        abs_diffs = []

        for row in rows:
            p_judge = safe_float(row.get("chief", {}).get("final_probability_yes"))
            p_panel = safe_float(row.get("doctors", {}).get(panelist, {}).get("p_yes"))

            if p_judge is None or p_panel is None:
                continue

            diff = p_judge - p_panel
            signed_diffs.append(diff)
            abs_diffs.append(abs(diff))

        results[panelist] = {
            "signed_diff_summary": summarize_scores(signed_diffs),
            "abs_diff_summary": summarize_scores(abs_diffs),
            "mean_signed_diff": float(np.mean(signed_diffs)) if signed_diffs else None,
            "mean_abs_diff": float(np.mean(abs_diffs)) if abs_diffs else None,
            "n": len(abs_diffs),
        }

    return results


def compute_influence(rows, panelists=DOCTOR_NAMES):
    counts = {p: 0 for p in panelists}
    total_cases_with_list = 0

    for row in rows:
        influenced = row.get("chief", {}).get("which_panelists_influenced_me")
        if not isinstance(influenced, list):
            continue

        total_cases_with_list += 1
        for p in influenced:
            if p in counts:
                counts[p] += 1

    rates = {}
    for p in panelists:
        rates[p] = {
            "count": int(counts[p]),
            "rate": float(counts[p] / total_cases_with_list) if total_cases_with_list > 0 else None,
        }

    return {
        "n_cases_with_influence_list": int(total_cases_with_list),
        "per_panelist": rates,
    }


def compute_contradiction_rate(rows, evaluators):
    results = {}

    for evaluator in evaluators:
        flags = []

        for row in rows:
            if evaluator == "chief":
                decision = row.get("chief", {}).get("final_decision")
                p_yes = row.get("chief", {}).get("final_probability_yes")
            else:
                doctor = row.get("doctors", {}).get(evaluator, {})
                decision = doctor.get("decision")
                p_yes = doctor.get("p_yes")

            flag = contradiction_flag(decision, p_yes)
            if flag is not None:
                flags.append(flag)

        n = len(flags)
        contradiction_count = int(sum(flags)) if n > 0 else 0

        results[evaluator] = {
            "n": n,
            "contradiction_count": contradiction_count,
            "contradiction_rate": float(contradiction_count / n) if n > 0 else None,
        }

    return results


def compute_verbosity(rows, evaluators):
    results = {}

    for evaluator in evaluators:
        raw_word_counts = []
        section_counts = []

        for row in rows:
            if evaluator == "chief":
                raw_text = row.get("chief", {}).get("raw")
            else:
                raw_text = row.get("doctors", {}).get(evaluator, {}).get("raw")

            if not raw_text:
                continue

            raw_word_counts.append(extract_word_count(raw_text))
            section_counts.append(len(extract_bullets_or_sections(raw_text)))

        results[evaluator] = {
            "raw_word_count_summary": summarize_scores(raw_word_counts),
            "section_count_summary": summarize_scores(section_counts),
            "n": len(raw_word_counts),
        }

    return results


def compute_evidence_alignment(rows, evaluators):
    """
    Enkel versjon:
    sjekker om evidence/evidence_cited-lignende felt finnes.
    Hvis de ikke finnes i datastrukturen, får du 0 eller None.
    """
    evidence_field_names = {"evidence", "evidence_cited", "evidence_used", "citations"}
    results = {}

    for evaluator in evaluators:
        total = 0
        has_evidence_field = 0

        for row in rows:
            if evaluator == "chief":
                obj = row.get("chief", {})
            else:
                obj = row.get("doctors", {}).get(evaluator, {})

            if not isinstance(obj, dict):
                continue

            total += 1
            if any(field in obj and obj.get(field) not in [None, "", []] for field in evidence_field_names):
                has_evidence_field += 1

        results[evaluator] = {
            "n": int(total),
            "n_with_evidence_field": int(has_evidence_field),
            "evidence_alignment_rate": (
                float(has_evidence_field / total) if total > 0 else None
            ),
        }

    return results


def compute_classification_for_all(rows, marksheet_df, target_col, n_bootstrap, stratify):
    results = {}

    for evaluator in ["chief"] + DOCTOR_NAMES:
        extracted = []

        for row in rows:
            patient_id = str(row["patient_ID"]).strip()
            decision = get_prediction(row, evaluator)
            p_yes = get_probability_yes(row, evaluator)

            if decision is None:
                continue

            extracted.append({
                "patient_ID": patient_id,
                "model_decision": str(decision).strip().upper(),
                "model_p_yes": safe_float(p_yes),
            })

        model_df = pd.DataFrame(extracted)
        if model_df.empty:
            results[evaluator] = None
            continue

        df = marksheet_df.merge(model_df, on="patient_ID", how="inner")

        y_true = df[target_col].map({"YES": 1, "NO": 0})
        y_pred = df["model_decision"].map({"YES": 1, "NO": 0})

        valid = y_true.notna() & y_pred.notna()
        df = df[valid].copy()
        y_true = y_true[valid].astype(int).values
        y_pred = y_pred[valid].astype(int).values

        if len(y_true) == 0:
            results[evaluator] = None
            continue

        p_yes_series = df["model_p_yes"]
        valid_prob = p_yes_series.notna()

        p_yes = None
        if valid_prob.sum() == len(df):
            p_yes = p_yes_series.astype(float).values

        results[evaluator] = bootstrap_metrics(
            y_true=y_true,
            y_pred=y_pred,
            p_yes=p_yes,
            n_bootstrap=n_bootstrap,
            random_state=42,
            stratify=stratify,
        )

    return results


def main(jsonl_path, marksheet_path, target_col, n_bootstrap, stratify, output_path=None):
    rows = load_jsonl(jsonl_path)

    marksheet = pd.read_csv(marksheet_path)
    marksheet.columns = marksheet.columns.str.strip()

    if "patient_ID" not in marksheet.columns:
        raise ValueError("CSV mangler kolonnen 'patient_ID'")
    if target_col not in marksheet.columns:
        raise ValueError(f"CSV mangler target-kolonnen '{target_col}'")

    marksheet["patient_ID"] = marksheet["patient_ID"].astype(str).str.strip()
    marksheet[target_col] = marksheet[target_col].astype(str).str.strip().str.upper()

    classification_results = compute_classification_for_all(
        rows=rows,
        marksheet_df=marksheet,
        target_col=target_col,
        n_bootstrap=n_bootstrap,
        stratify=stratify,
    )

    follow_rate = compute_follow_rate(rows)
    probability_closeness = compute_probability_closeness(rows)
    influence = compute_influence(rows)
    contradiction_rate = compute_contradiction_rate(rows, ["chief"] + DOCTOR_NAMES)
    verbosity = compute_verbosity(rows, ["chief"] + DOCTOR_NAMES)
    evidence_alignment = compute_evidence_alignment(rows, ["chief"] + DOCTOR_NAMES)

    final_output = {
        "jsonl_path": str(jsonl_path),
        "marksheet_path": str(marksheet_path),
        "target_col": target_col,
        "stratify": stratify,
        "n_bootstrap": n_bootstrap,
        "classification_metrics": classification_results,
        "follow_rate_per_panelist": follow_rate,
        "probability_closeness": probability_closeness,
        "influence": influence,
        "contradiction_rate": contradiction_rate,
        "verbosity": verbosity,
        "evidence_alignment": evidence_alignment,
    }

    print(json.dumps(final_output, indent=2, ensure_ascii=False))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path til merged JSONL-fil")
    parser.add_argument("--marksheet", required=True, help="Path til CSV med fasit")
    parser.add_argument(
        "--target-col",
        default="GP_fasit",
        help="Kolonnen i CSV som inneholder fasit, default=GP_fasit",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Antall bootstrap-runder",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Skru av stratified bootstrap",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Valgfri path til JSON-fil med metrics",
    )

    args = parser.parse_args()

    main(
        jsonl_path=Path(args.jsonl),
        marksheet_path=Path(args.marksheet),
        target_col=args.target_col,
        n_bootstrap=args.n_bootstrap,
        stratify=not args.no_stratify,
        output_path=args.output,
    )