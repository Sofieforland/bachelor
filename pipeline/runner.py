# pipeline/runner.py
# bachelor/pipeline/runner.py
import json
from pathlib import Path
import pandas as pd

from pipeline.panel import run_panel_on_row
from pipeline.parsing import parse_decision_fields

def run_file(
    model,
    in_path: Path,
    out_jsonl_path: Path,
    n_rows: int | None = None,
    write_csv: bool = False,
    out_csv_path: Path | None = None,
    append_jsonl: bool = False,
    model_name: str | None = None,
    model_id: str | None = None,
):
    df = pd.read_csv(in_path).head(n_rows).copy()

    # overwrite hvis ikke append
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if not append_jsonl:
        with open(out_jsonl_path, "w", encoding="utf-8") as f:
            pass

    if write_csv:
        df["Doctor_Cautious"] = ""
        df["Doctor_Pragmatic"] = ""
        df["Chief_Output"] = ""
        df["Doctor_Cautious_DECISION"] = ""
        df["Doctor_Cautious_P_YES"] = ""
        df["Doctor_Pragmatic_DECISION"] = ""
        df["Doctor_Pragmatic_P_YES"] = ""
        df["Chief_FINAL_DECISION"] = ""
        df["Chief_P_YES"] = ""

    for i, row in df.iterrows():
        gp_note = row["input_text_gp"]
        patient_id = row.get("patient_ID", i)
        label = row.get("label", None)

        opinions, chief_text = run_panel_on_row(model, gp_note)

        cautious_text = opinions["doctor_1_cautious_gp"]
        overconfident_text = opinions["doctor_2_overconfident_gp"]
        concervative_text = opinions("doctor_3_conservative_gp")
        neutral_text = opinions("doctor_4_neutral_gp")

        c_dec, c_p = parse_decision_fields(cautious_text)
        p_dec, p_p = parse_decision_fields(overconfident_text)
        p_con, p_con_p = parse_decision_fields(concervative_text)
        p_neutral, p_neutral_p = parse_decision_fields(neutral_text)

        ch_dec, ch_p = parse_decision_fields(chief_text)

        record = {
            "patient_ID": patient_id,
            "label": None if pd.isna(label) else int(label),
            "model": model_name,
            "model_id": model_id,

            # uten dette feltet vil ikke evidence_alignment funke
            "input": {
                "patient_ID": row.get("patient_ID"),
                "patient_age": row.get("patient_age"),
                "psa": row.get("psa"),
                "prostate_volume": row.get("prostate_volume"),
                "psad": row.get("psad"),
                "center": row.get("center"),
            },

            "gp_note": gp_note,

            "doctors": {
                "cautious_gp": {
                    "decision": c_dec,
                    "p_yes": c_p,
                    "raw": cautious_text
                },
                "overconfident_gp": {
                    "decision": p_dec,
                    "p_yes": p_p,
                    "raw": overconfident_text
                },
                "conservative_gp": {
                    "decision": p_con,
                    "p_yes": p_con_p,
                    "raw": concervative_text
                },
                "neutral_gp": {
                    "decision": p_neutral,
                    "p_yes": p_neutral_p,
                    "raw": neutral_text
                },
            },

            "chief": {
                "raw": chief_text,
                "final_decision": ch_dec,
                "p_yes": ch_p
            },
        }

        with open(out_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("Saved JSON:", out_jsonl_path)
        print(f"Done with patient {patient_id} ({i+1}/{len(df)})")
