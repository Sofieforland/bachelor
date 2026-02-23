#from schema import validate_prediction  # Til selve ai bruk


from pathlib import Path
import os
import json
import pandas as pd
from openai import OpenAI
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = REPO_ROOT / "outputs" / "dataset_with_notes.csv"
OUT_PATH = REPO_ROOT / "outputs" / "agent_outputs_openai.jsonl"
#CONFIG_PATH = REPO_ROOT / "configs" / "run_config.json"     #Prøver å teste
#SELECTED_CASES_PATH = REPO_ROOT / "configs" / "selected_cases.txt"


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Strict JSON schema for output
DECISION_SCHEMA = {
    "name": "csPCa_decision",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "Decision": {"type": "string", "enum": ["YES", "NO"]},
            "Probability_yes": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "self_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale_bullets": {
                "type": "array",
                "minItems": 3,
                "maxItems": 6,
                "items": {"type": "string"},
            },
            "evidence_cited": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
        },
        "required": [
            "Decision",
            "Probability_yes",
            "self_confidence",
            "rationale_bullets",
            "evidence_cited",
        ],
    },
    "strict": True,
}

# 3) Panel prompts (Sofie = konservativ, Dorte = sensitiv)
SYSTEM_PROMPTS = {
    ("gp", "sofie"): (
        "You are a careful general practitioner (GP). "
        "Be conservative: recommend further testing/referral (YES) only if the provided clinical summary suggests meaningful risk. "
        "Use only evidence present in the note."
    ),
    ("gp", "dorte"): (
        "You are a safety-oriented general practitioner (GP) focused on not missing csPCa. "
        "Be sensitive: if there are concerning signals or uncertainty in key markers, lean YES. "
        "Use only evidence present in the note."
    ),
    ("radiology", "sofie"): (
        "You are a careful radiologist. "
        "Recommend biopsy (YES) only if the provided imaging/clinical summary supports suspicion of csPCa. "
        "Be conservative and avoid over-calling. Use only evidence present."
    ),
    ("radiology", "dorte"): (
        "You are a risk-averse radiologist focused on not missing csPCa. "
        "If there is suspicion or uncertainty with risk factors, lean YES for biopsy recommendation. "
        "Use only evidence present."
    ),
}

# Legger til her, usikker på om de skal brukes

# def load_run_config() -> dict:
#     with open(CONFIG_PATH, "r", encoding="utf-8") as f:
#         return json.load(f)

# def load_selected_case_ids() -> list[str]:
#     if not SELECTED_CASES_PATH.exists():
#         return []
#     ids = []
#     with open(SELECTED_CASES_PATH, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 continue
#             ids.append(line)
#     return ids

def run_agent(role: str, panel: str, input_text: str, model: str = "gpt-5.2"):
    system = SYSTEM_PROMPTS[(role, panel)]
    user_prompt = (
        input_text
        + "\n\nReturn your answer strictly following the JSON schema. "
          "In evidence_cited, list the exact fields you relied on (e.g., 'Age', 'PSA', 'PSAD', 'Imaging summary')."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "csPCa_decision",
                "strict": True,
                "schema": DECISION_SCHEMA["schema"],
            }
        },
    )

    raw = getattr(resp, "output_text", None)
    if raw is None:
        raw = resp.output[0].content[0].text

    return json.loads(raw)


def main():
    df = pd.read_csv(IN_PATH)

    # Start med få rader for å teste (kan øke senere)
    df = df.head(10).copy()

    OUT_PATH.parent.mkdir(exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            case_id = str(row["patient_ID"])
            label = 1 if str(row["case_csPCa"]).strip().upper() == "YES" else 0


            for panel in ["sofie", "dorte"]:
                # GP
                gp_out = run_agent("gp", panel, row["input_text_gp"])
                gp_out.update({"case_ID": case_id, "role": "gp", "panel": panel, "label": label})
                f.write(json.dumps(gp_out, ensure_ascii=False) + "\n")

                time.sleep(25)

                # Radiology
                rad_out = run_agent("radiology", panel, row["input_text_radiology"])
                rad_out.update({"case_ID": case_id, "role": "radiology", "panel": panel, "label": label})
                f.write(json.dumps(rad_out, ensure_ascii=False) + "\n")

                time.sleep(25)

    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
