#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 1) PARSING-REGLER (henter YES/NO og sannsynlighet fra output)
# ============================================================

YES_SET = {"YES", "Y", "TRUE", "1"}
NO_SET = {"NO", "N", "FALSE", "0"}

# Finder "YES" eller "NO" i fri tekst
DECISION_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)

# Finder sannsynlighet i litt ulike formater:
# - "P_YES=0.75"
# - "probability: 0.73"
# - "YES (0.73)" / "YES [73%]" osv
PROB_RE = re.compile(
    r"(?:(?:prob(?:ability)?)|(?:p\s*[_\(]?\s*yes\s*[_\)]?))\s*[:=]\s*([0-9]*\.?[0-9]+)\s*%?"
    r"|(?:\bYES\b|\bNO\b)\s*[\(\[]\s*([0-9]*\.?[0-9]+)\s*%?\s*[\)\]]",
    re.IGNORECASE,
)


def normalize_prob(x: Any) -> Optional[float]:
    """
    Normaliserer sannsynlighet til 0..1.
    - Hvis output er 0.75 -> 0.75
    - Hvis output er 75 -> 0.75 (tolkes som prosent)
    """
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None

    if math.isnan(v) or math.isinf(v):
        return None

    if 0.0 <= v <= 1.0:
        return float(v)
    if 1.0 < v <= 100.0:
        return float(v) / 100.0
    return None


def parse_decision_prob_from_obj(obj: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], str]:
    """
    Returnerer:
      - decision_01: 1=YES, 0=NO, None hvis ikke funnet
      - p_yes: sannsynlighet(YES) i 0..1, None hvis ikke funnet
      - raw_text_used: teksten vi faktisk parse'et fra
    """

    # (A) Først: bruk strukturerte felter hvis de finnes
    # Du har f.eks:
    #   - doctors.*.decision
    #   - doctors.*.p_yes
    #   - chief.final_decision
    decision = (
        obj.get("decision")
        or obj.get("Decision")
        or obj.get("final_decision")
        or obj.get("FINAL_DECISION")
    )
    prob = (
        obj.get("p_yes")
        or obj.get("pYES")
        or obj.get("probability")
        or obj.get("Probability")
        or obj.get("prob_yes")
        or obj.get("P_YES")
    )

    decision_01: Optional[int] = None
    if isinstance(decision, str):
        d = decision.strip().upper()
        if d in YES_SET:
            decision_01 = 1
        elif d in NO_SET:
            decision_01 = 0
    elif isinstance(decision, (int, bool)):
        # 0/1 eller True/False
        decision_01 = int(decision)

    prob_01 = normalize_prob(prob)

    raw = obj.get("raw") or obj.get("text") or obj.get("response") or ""
    raw_str = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)

    # (B) Hvis decision/prob mangler: forsøk å finne det i raw-teksten via regex
    if decision_01 is None and raw_str:
        m = DECISION_RE.search(raw_str)
        if m:
            decision_01 = 1 if m.group(1).upper() == "YES" else 0

    if prob_01 is None and raw_str:
        m = PROB_RE.search(raw_str)
        if m:
            g1 = m.group(1)
            g2 = m.group(2)
            prob_01 = normalize_prob(g1 if g1 is not None else g2)

    return decision_01, prob_01, raw_str


def word_count(text: str) -> int:
    """En enkel proxy for 'hvor mye forklaring' modellen skrev."""
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def contradiction(decision_01: Optional[int], p_yes: Optional[float], threshold: float = 0.5) -> Optional[int]:
    """
    'Contradiction' = intern inkonsistens:
      - sier YES, men p_yes < 0.5
      - sier NO, men p_yes >= 0.5
    """
    if decision_01 is None or p_yes is None:
        return None
    if decision_01 == 1 and p_yes < threshold:
        return 1
    if decision_01 == 0 and p_yes >= threshold:
        return 1
    return 0


def safe_get_case_id(rec: Dict[str, Any]) -> str:
    """Hent case-id/patient-id robust."""
    for k in ["case_id", "patient_ID", "patient_id", "id", "ID"]:
        if k in rec:
            return str(rec[k])
    return str(abs(hash(json.dumps(rec, sort_keys=True, ensure_ascii=False))) % 10**12)


# ============================================================
# 2) BYGG "parsed_predictions.csv"
#    (én rad per case per rolle: hver panelist + chief)
# ============================================================

def collect_roles_for_case(rec: Dict[str, Any], chief_key: str) -> Dict[str, Any]:
    """
    Samler alle rolle-output i én dict.

    Din JSONL:
      - panelister ligger i rec["doctors"]
      - chief ligger som rec["chief"] på toppnivå

    Dette gjør at vi kan behandle chief på samme måte som de andre rollene.
    """
    roles: Dict[str, Any] = {}
    doctors = rec.get("doctors", {})
    if isinstance(doctors, dict):
        roles.update(doctors)

    # legg til chief hvis den finnes på toppnivå
    if chief_key in rec and isinstance(rec[chief_key], dict):
        roles[chief_key] = rec[chief_key]

    return roles


def build_parsed_predictions(records: List[Dict[str, Any]], judge_name: str, chief_key: str) -> pd.DataFrame:
    rows = []

    for rec in records:
        case_id = safe_get_case_id(rec)

        # (valgfritt) hvis du senere vil gjøre "evidence alignment":
        input_obj = rec.get("input", None)
        input_keys = set(input_obj.keys()) if isinstance(input_obj, dict) else set()

        # (valgfritt) hvis du har fasit-label (y_true), kan vi regne accuracy osv.
        y_true = rec.get("y_true") or rec.get("label") or rec.get("csPCa") or rec.get("target")
        y_true_01 = None
        if isinstance(y_true, str):
            y = y_true.strip().upper()
            if y in YES_SET:
                y_true_01 = 1
            elif y in NO_SET:
                y_true_01 = 0
        elif isinstance(y_true, (int, bool)) and y_true in [0, 1, True, False]:
            y_true_01 = int(y_true)

        # Samle alle roller: panelister + chief
        roles = collect_roles_for_case(rec, chief_key=chief_key)

        for role, out in roles.items():
            # Normaliser til dict
            if isinstance(out, str):
                out = {"raw": out}
            elif not isinstance(out, dict):
                out = {"raw": json.dumps(out, ensure_ascii=False)}

            dec, p, raw_text = parse_decision_prob_from_obj(out)

            # Evidence/citations (valgfritt felt)
            evidence = out.get("evidence_cited") or out.get("evidence") or out.get("citations")
            if isinstance(evidence, list):
                evidence_list = [str(x) for x in evidence]
            elif isinstance(evidence, str):
                evidence_list = [e.strip() for e in evidence.split(",") if e.strip()]
            else:
                evidence_list = []

            # Evidence alignment (valgfritt):
            # Hvis evidence_list inneholder keys som finnes i rec["input"], scorer vi hvor mange som matcher.
            ev_align = None
            if evidence_list and input_keys:
                hits = sum(1 for e in evidence_list if e in input_keys)
                ev_align = hits / max(1, len(evidence_list))

            rows.append(
                {
                    "case_id": case_id,
                    "role": role,
                    # is_judge = 1 for chief/judge-rollen, 0 for panelister
                    "is_judge": int(role == judge_name),
                    "decision": dec,       # 1=YES, 0=NO, None
                    "p_yes": p,            # 0..1, None
                    "raw": raw_text,
                    "word_count": word_count(raw_text),
                    "contradiction": contradiction(dec, p),
                    "evidence_cited": json.dumps(evidence_list, ensure_ascii=False),
                    "evidence_alignment": ev_align,
                    "y_true": y_true_01,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 3) "pairwise_follow.csv"
#    (måler hvor ofte chief følger hver panelist i uenighet)
# ============================================================

def build_pairwise_follow(df_preds: pd.DataFrame, judge_name: str) -> pd.DataFrame:
    """
    Én rad per (case, panelist).
    Dette brukes til "follow-rate" metrics:

      - disagreement: om panelet var uenig (både YES og NO finnes blant panelister)
      - judge_follows: kun definert når disagreement=1:
            1 hvis chief sin beslutning == panelist sin beslutning
            0 ellers
      - abs_prob_diff: hvor langt unna p_yes er mellom panelist og chief

    Poenget:
      Hvis panelistene er uenige, kan vi se hvem chief "ligner mest på" i beslutning/prob.
    """
    rows = []

    df_j = df_preds[df_preds["role"] == judge_name][["case_id", "decision", "p_yes", "raw"]].rename(
        columns={"decision": "judge_decision", "p_yes": "judge_p_yes", "raw": "judge_raw"}
    )
    df_p = df_preds[df_preds["role"] != judge_name].copy()

    if df_j.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "panelist",
                "disagreement",
                "judge_follows",
                "abs_prob_diff",
                "judge_decision",
                "panelist_decision",
                "judge_p_yes",
                "panelist_p_yes",
                "chief_mentions_panelist",
            ]
        )

    # Disagreement per case: finnes både 0 og 1 blant panelister som har decision
    panel_decisions = (
        df_p.dropna(subset=["decision"])
        .groupby("case_id")["decision"]
        .agg(lambda s: set(int(x) for x in s.tolist()))
    )
    disagreement_map = {cid: int((0 in s) and (1 in s)) for cid, s in panel_decisions.items()}

    # Merge judge info på panelist-rader
    df_m = df_p.merge(df_j, on="case_id", how="left")

    for _, r in df_m.iterrows():
        cid = r["case_id"]
        panelist = r["role"]
        dis = disagreement_map.get(cid, 0)

        panel_dec = r["decision"]
        judge_dec = r["judge_decision"]

        follows = None
        if dis == 1 and pd.notna(panel_dec) and pd.notna(judge_dec):
            follows = int(int(panel_dec) == int(judge_dec))

        abs_diff = None
        if pd.notna(r["p_yes"]) and pd.notna(r["judge_p_yes"]):
            abs_diff = float(abs(float(r["judge_p_yes"]) - float(r["p_yes"])))

        # "Influence proxy": nevner chief panelisten i teksten?
        # (Dette er en enkel indikator på "chief refererer til panelistene".)
        chief_text = str(r.get("judge_raw", "") or "")
        mentions = int(panelist.lower() in chief_text.lower())

        rows.append(
            {
                "case_id": cid,
                "panelist": panelist,
                "disagreement": dis,
                "judge_follows": follows,
                "abs_prob_diff": abs_diff,
                "judge_decision": judge_dec,
                "panelist_decision": panel_dec,
                "judge_p_yes": r["judge_p_yes"],
                "panelist_p_yes": r["p_yes"],
                "panelist_word_count": r.get("word_count", None),
                "panelist_contradiction": r.get("contradiction", None),
                "panelist_evidence_alignment": r.get("evidence_alignment", None),
                "chief_mentions_panelist": mentions,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 4) "metrics_summary.csv"
#    (aggregerer metrics per rolle + panelist-follow metrics)
# ============================================================

def _accuracy_from_decisions(g: pd.DataFrame) -> float:
    """
    Accuracy måler hvor ofte modellen treffer fasit (y_true).
    Krever at y_true finnes i fila (0/1).
    """
    gg = g.dropna(subset=["decision", "y_true"])
    if gg.empty:
        return np.nan
    return float((gg["decision"].astype(int) == gg["y_true"].astype(int)).mean())


def _brier_score(g: pd.DataFrame) -> float:
    """
    Brier score måler sannsynlighet-kvalitet:
      mean((p_yes - y_true)^2)
    Krever både p_yes og y_true.
    Lavere er bedre.
    """
    gg = g.dropna(subset=["p_yes", "y_true"])
    if gg.empty:
        return np.nan
    p = gg["p_yes"].astype(float).to_numpy()
    y = gg["y_true"].astype(float).to_numpy()
    return float(np.mean((p - y) ** 2))


def compute_metrics_summary(df_preds: pd.DataFrame, df_pair: pd.DataFrame, judge_name: str) -> pd.DataFrame:
    """
    Lager én oppsummeringsrad per rolle.

    Inneholder:
      - decision_rate_yes: hvor ofte rollen sier YES
      - mean_p_yes: gj.sn sannsynlighet for YES
      - mean_word_count: gj.sn lengde på svaret
      - contradiction_rate: hvor ofte decision motsier p_yes
      - follow_rate_disagreement: for panelister, hvor ofte chief følger dem når det er uenighet
      - mean_abs_prob_diff_disagreement: hvor nær panelisten er chief i sannsynlighet ved uenighet
      - chief_mentions_rate: hvor ofte chief nevner panelisten (proxy for "influence"/referanser)
      - (valgfritt) accuracy og brier hvis y_true finnes
    """
    summary_rows = []

    # Per-rolle stats fra parsed_predictions
    for role, g in df_preds.groupby("role"):
        summary_rows.append(
            {
                "role": role,
                "is_judge": int(role == judge_name),
                "n_cases": int(g["case_id"].nunique()),
                "decision_rate_yes": float(np.nanmean(g["decision"])) if g["decision"].notna().any() else np.nan,
                "mean_p_yes": float(np.nanmean(g["p_yes"])) if g["p_yes"].notna().any() else np.nan,
                "mean_word_count": float(np.nanmean(g["word_count"])) if g["word_count"].notna().any() else np.nan,
                "contradiction_rate": float(np.nanmean(g["contradiction"])) if g["contradiction"].notna().any() else np.nan,
                "mean_evidence_alignment": float(np.nanmean(g["evidence_alignment"])) if g["evidence_alignment"].notna().any() else np.nan,
                # Label-baserte metrics (kun hvis y_true finnes)
                "accuracy": _accuracy_from_decisions(g),
                "brier": _brier_score(g),
            }
        )

    df_sum = pd.DataFrame(summary_rows)

    # Panelist-follow metrics fra pairwise_follow
    if not df_pair.empty:
        dis = df_pair[df_pair["disagreement"] == 1].copy()

        # follow-rate kun når panelistene faktisk var uenige
        follow = dis.groupby("panelist")["judge_follows"].mean()
        absdiff = dis.groupby("panelist")["abs_prob_diff"].mean()
        dis_n = dis.groupby("panelist")["case_id"].nunique()

        # hvor ofte chief nevner panelisten (uavhengig av uenighet)
        mention_rate = df_pair.groupby("panelist")["chief_mentions_panelist"].mean()

        df_sum["follow_rate_disagreement"] = df_sum["role"].map(follow)
        df_sum["mean_abs_prob_diff_disagreement"] = df_sum["role"].map(absdiff)
        df_sum["n_disagreement_cases"] = df_sum["role"].map(dis_n)
        df_sum["chief_mentions_rate"] = df_sum["role"].map(mention_rate)

    return df_sum


# ============================================================
# 5) IO + MAIN (les JSONL og skriv CSV-output)
# ============================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Leser enten:
      - JSONL: én JSON per linje
      - JSON:  én fil som er en liste: [ {...}, {...} ]
    Returnerer alltid en liste med records.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Hvis fila starter med '[' antar vi at det er en JSON-liste
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise RuntimeError(f"{path} starts with '[' but is not a JSON list.")
        # sikre at hvert element er dict
        out = []
        for i, obj in enumerate(data):
            if isinstance(obj, dict):
                out.append(obj)
            else:
                out.append({"raw": obj})
        return out

    # Ellers: fall back til JSONL (én per linje)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error in {path} at line {line_no}: {e}") from e
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True, help="Path to JSONL (one case per line)")
    ap.add_argument("--out_dir", required=True, help="Output directory for CSVs")
    ap.add_argument("--judge_name", default="chief", help="Role name used in outputs (default: chief)")
    ap.add_argument("--chief_key", default="chief", help="Key used at top-level for chief object (default: chief)")
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(in_path)

    # 1) parsed_predictions.csv: grunnlaget for nesten alle metrics
    df_preds = build_parsed_predictions(records, judge_name=args.judge_name, chief_key=args.chief_key)

    # 2) pairwise_follow.csv: follow-rate når panelistene er uenige
    df_pair = build_pairwise_follow(df_preds, judge_name=args.judge_name)

    # 3) metrics_summary.csv: aggregerte tall per rolle
    df_sum = compute_metrics_summary(df_preds, df_pair, judge_name=args.judge_name)

    preds_path = out_dir / "parsed_predictions.csv"
    pair_path = out_dir / "pairwise_follow.csv"
    sum_path = out_dir / "metrics_summary.csv"

    df_preds.to_csv(preds_path, index=False)
    df_pair.to_csv(pair_path, index=False)
    df_sum.to_csv(sum_path, index=False)

    print("Wrote:")
    print(f"- {preds_path}")
    print(f"- {pair_path}")
    print(f"- {sum_path}")

    # Lite “peek” i terminalen
    if "follow_rate_disagreement" in df_sum.columns:
        cols = ["role", "is_judge", "n_cases", "decision_rate_yes", "mean_p_yes",
                "follow_rate_disagreement", "n_disagreement_cases", "mean_abs_prob_diff_disagreement",
                "accuracy", "brier", "chief_mentions_rate"]
        cols = [c for c in cols if c in df_sum.columns]
        print("\nSummary preview:")
        print(df_sum.sort_values(["is_judge", "role"], ascending=[False, True])[cols].to_string(index=False))


if __name__ == "__main__":
    main()