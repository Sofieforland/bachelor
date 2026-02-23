import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

BASE_DIR = BASE_DIR = Path.home() / "Bachelor" / "BachelorProject" / "bachelor" # hardcode- better: Path(__file__).resolve().parent.parent
IN_PATH = BASE_DIR / "outputs" / "dataset_with_notes.csv"
OUT_CSV_PATH = BASE_DIR / "outputs" / "dataset_with_qwen_outputs.csv"
OUT_JSON_PATH = BASE_DIR / "outputs" / "dataset_with_qwen_outputs.jsonl"
N_ROWS = 1  # start small, increase once it works

# ---- Load model + processor ----
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

DOCTORS_GP = {
    "doctor_1_cautious_gp": {
        "system": "You are an experienced General Practitioner. Very cautious, low threshold for further workup. Short and structured."
    },
    "doctor_2_pragmatic_gp": {
        "system": "You are a pragmatic General Practitioner. Follows guidelines, avoids unnecessary tests. Structured answer."
    },
    "chief_physician_decider": {
        "system": "You are the chief physician and lead the MDT. Make the final decision based on inputs. Weigh disagreements and propose a plan."
    },
}

DOCTOR_TASK = """
Read the note and respond in this format:

1) Brief assessment (1-3 sentences)
2) Key findings (bullet list)
3) Uncertainty / what's missing? (bullet list)
4) Recommended next step (one clear recommendation)
5) Finally on a SEPARATE LINE decide if the patient should be escalated: DECISION=<YES/NO> and P_YES=<0-1>

NOTE:
{note}
"""

CHIEF_TASK = """
You are the chief physician. You receive a General Practitioner (GP) patient note and input from 2 doctors. Make the final decision.

GP NOTE:
{gp_note}

DOCTORS' INPUT:
{compiled}

Write in this format:
A) Final decision (one sentence)
B) Plan (bullet list, max 5 bullets)
C) Why (short, 3-6 sentences)
D) If disagreement: how you weighted it (2-4 sentences)
E) Finally on a SEPARATE LINE: FINAL_DECISION=<YES/NO> and P_YES=<0-1>
"""


def build_messages(system_prompt: str, user_text: str):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


@torch.inference_mode()
def run_doctor(system_prompt: str, user_text: str, max_new_tokens: int = 256) -> str:
    messages = build_messages(system_prompt, user_text)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text


def parse_decision_fields(text: str):
    """
    Extracts YES/NO + probability from either:
      DECISION=YES
      FINAL_DECISION=YES
      P_YES=0.75
    With or without angle brackets.
    """

    decision = None
    p_yes = None

    # Match DECISION=YES or FINAL_DECISION=YES
    m = re.search(
        r"\b(?:FINAL_DECISION|DECISION)\s*=\s*<?\s*(YES|NO)\s*>?",
        text,
        re.IGNORECASE,
    )
    if m:
        decision = m.group(1).upper()

    # Match P_YES=0.75
    m2 = re.search(
        r"\bP_YES\s*=\s*<?\s*([01](?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    if m2:
        try:
            p_yes = float(m2.group(1))
            p_yes = max(0.0, min(1.0, p_yes))
        except ValueError:
            p_yes = None

    return decision, p_yes


def run_panel_on_row(gp_note: str):
    opinions = {}

    for key in ["doctor_1_cautious_gp", "doctor_2_pragmatic_gp"]:
        user_text = DOCTOR_TASK.format(note=gp_note)
        opinions[key] = run_doctor(DOCTORS_GP[key]["system"], user_text, max_new_tokens=256)

    compiled = ""
    for k, v in opinions.items():
        compiled += f"--- {k} ---\n{v}\n\n"

    chief_text = CHIEF_TASK.format(gp_note=gp_note, compiled=compiled)
    chief = run_doctor(DOCTORS_GP["chief_physician_decider"]["system"], chief_text, max_new_tokens=384)

    return opinions, chief


def main():
    df = pd.read_csv(IN_PATH).head(N_ROWS).copy()

    # Output columns (raw text)
    df["Doctor_Cautious"] = ""
    df["Doctor_Pragmatic"] = ""
    df["Chief_Output"] = ""

    # Parsed fields for convenience
    df["Doctor_Cautious_DECISION"] = ""
    df["Doctor_Cautious_P_YES"] = ""
    df["Doctor_Pragmatic_DECISION"] = ""
    df["Doctor_Pragmatic_P_YES"] = ""
    df["Chief_FINAL_DECISION"] = ""
    df["Chief_P_YES"] = ""

    json_rows = []

    for i, row in df.iterrows():
        gp_note = row["input_text_gp"]
        patient_id = row.get("patient_ID", i)

        opinions, chief = run_panel_on_row(gp_note)

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

    # Save CSV
    df.to_csv(OUT_CSV_PATH, index=False)
    print("Saved CSV:", OUT_CSV_PATH)

    # Save JSON
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, ensure_ascii=False, indent=2)
    print("Saved JSON:", OUT_JSON_PATH)


if __name__ == "__main__":
    main()