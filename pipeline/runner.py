# pipeline/runner.py
# bachelor/pipeline/runner.py
import json
from pathlib import Path
import pandas as pd

from bachelor.pipeline.panel import run_panel_on_row
from bachelor.pipeline.parsing import parse_decision_fields

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

        opinions, chief_text = run_panel_on_row(model, gp_note)

        #cautious_text = opinions["doctor_1_cautious_gp"]
        #pragmatic_text = opinions["doctor_2_pragmatic_gp"]
        conservative_text = opinions["doctor_3_conservative_gp"]
        #neutral_text = opinions["doctor_4_neutral_gp"]

        #c_dec, c_p = parse_decision_fields(cautious_text)
        #p_dec, p_p = parse_decision_fields(pragmatic_text)
        cons_dec, cons_p = parse_decision_fields(conservative_text)
       # neut_dec, neut_p = parse_decision_fields(neutral_text)
        #ch_dec, ch_p = parse_decision_fields(chief_text)

        record = {
            "patient_ID": patient_id,
            "model": model_name,
            #"model_id": model_id,
          # "gp_note": gp_note,
            "doctors": {
                # "cautious_gp": {"decision": c_dec, 
                #                 "p_yes": c_p,
                #                 "raw": cautious_text
                #                 },
                # "pragmatic_gp": {"decision": p_dec, "p_yes": p_p, "raw": pragmatic_text},
                "conservative_gp": {"decision": cons_dec, "p_yes": cons_p, "raw": conservative_text},
              #  "neutral_gp": {"decision": neut_dec, "p_yes": neut_p, "raw": neutral_text},

            },
           # "chief": {"raw": chief_text, "final_decision": ch_dec, "p_yes": ch_p},
        }

        with open(out_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("Saved JSON:", out_jsonl_path)
        print("done with patient", patient_id)
