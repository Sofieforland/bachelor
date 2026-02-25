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
    n_rows: int = 3,
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

        cautious_text = opinions["doctor_1_cautious_gp"]
        pragmatic_text = opinions["doctor_2_pragmatic_gp"]

        c_dec, c_p = parse_decision_fields(cautious_text)
        p_dec, p_p = parse_decision_fields(pragmatic_text)
        ch_dec, ch_p = parse_decision_fields(chief_text)

        # record = {
        #     "patient_ID": patient_id,
        #     "model": model_name,
        #     "model_id": model_id,
        #     "gp_note": gp_note,
        #     "doctors": {
        #         "cautious_gp": {"raw": cautious_text, "decision": c_dec, "p_yes": c_p},
        #         "pragmatic_gp": {"raw": pragmatic_text, "decision": p_dec, "p_yes": p_p},
        #     },
        #     "chief": {"raw": chief_text, "final_decision": ch_dec, "p_yes": ch_p},
        # }

        # with open(out_jsonl_path, "a", encoding="utf-8") as f:
        #     f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ULIKT OG MER LESBART FORMAT, men greit for lav n index
    json_rows = []

    for i, row in df.iterrows():
        gp_note = row["input_text_gp"]
        patient_id = row.get("patient_ID", i)

        opinions, chief = run_panel_on_row(model, gp_note)

        cautious_text = opinions["doctor_1_cautious_gp"]
        pragmatic_text = opinions["doctor_2_pragmatic_gp"]
        chief_text = chief

        df.at[i, "Doctor_Cautious"] = cautious_text
        df.at[i, "Doctor_Pragmatic"] = pragmatic_text
        df.at[i, "Chief_Output"] = chief_text

        c_dec, c_p = parse_decision_fields(cautious_text)
        p_dec, p_p = parse_decision_fields(pragmatic_text)
        ch_dec, ch_p = parse_decision_fields(chief_text)

        df.at[i, "Doctor_Cautious_DECISION"] = c_dec or ""
        df.at[i, "Doctor_Cautious_P_YES"] = "" if c_p is None else c_p
        df.at[i, "Doctor_Pragmatic_DECISION"] = p_dec or ""
        df.at[i, "Doctor_Pragmatic_P_YES"] = "" if p_p is None else p_p
        df.at[i, "Chief_FINAL_DECISION"] = ch_dec or ""
        df.at[i, "Chief_P_YES"] = "" if ch_p is None else ch_p

        json_rows.append(
            {
                "patient_ID": patient_id,
                "gp_note": gp_note,
                "doctors": {
                    "cautious_gp": {
                        "raw": cautious_text,
                        "decision": c_dec,
                        "p_yes": c_p,
                    },
                    "pragmatic_gp": {
                        "raw": pragmatic_text,
                        "decision": p_dec,
                        "p_yes": p_p,
                    },
                },
                "chief": {
                    "raw": chief_text,
                    "final_decision": ch_dec,
                    "p_yes": ch_p,
                },
            }
        )

        print(f"Done patient {patient_id} ({i+1}/{len(df)})")

    # Save JSONl
        with open(out_jsonl_path, "w", encoding="utf-8") as f:
            json.dump(json_rows, f, ensure_ascii=False, indent=2)
        print("Saved JSON:", out_jsonl_path)

#######

        if write_csv:
            df.at[i, "Doctor_Cautious"] = cautious_text
            df.at[i, "Doctor_Pragmatic"] = pragmatic_text
            df.at[i, "Chief_Output"] = chief_text
            df.at[i, "Doctor_Cautious_DECISION"] = c_dec or ""
            df.at[i, "Doctor_Cautious_P_YES"] = "" if c_p is None else c_p
            df.at[i, "Doctor_Pragmatic_DECISION"] = p_dec or ""
            df.at[i, "Doctor_Pragmatic_P_YES"] = "" if p_p is None else p_p
            df.at[i, "Chief_FINAL_DECISION"] = ch_dec or ""
            df.at[i, "Chief_P_YES"] = "" if ch_p is None else ch_p

        print(f"Done patient {patient_id} ({i+1}/{len(df)})")

    print("Saved JSONL:", out_jsonl_path)

    if write_csv:
        if out_csv_path is None:
            raise ValueError("write_csv=True requires out_csv_path")
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv_path, index=False)
        print("Saved CSV:", out_csv_path)