import gc
import json
from pathlib import Path
import torch

from pipeline.prompts import DOCTORS_GP


def extract_json(text: str) -> dict:
    text = text.strip()

    # fjern ```json ```
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    # finn JSON-del
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]

    return json.loads(text)


def validate_chief(chief_obj: dict, patient_id=None):
    dec = chief_obj.get("final_decision")
    p = chief_obj.get("final_probability_yes")
    panelists = chief_obj.get("which_panelists_influenced_me", [])

    if dec not in {"YES", "NO"}:
        print(f"Invalid chief decision for patient {patient_id}: {dec}")

    if not isinstance(p, (int, float)) or not (0 <= p <= 1):
        print(f"Invalid chief probability for patient {patient_id}: {p}")

    if not isinstance(panelists, list):
        print(f"Invalid panelist list for patient {patient_id}: {panelists}")


def validate_doctors(doctors: dict, patient_id=None):
    for name, d in doctors.items():
        dec = d.get("decision")
        p = d.get("p_yes")

        if dec is None or p is None:
            continue

        if dec == "YES" and p < 0.5:
            print(f"Inconsistency in {name} (patient {patient_id}): YES but p_yes={p}")

        if dec == "NO" and p > 0.5:
            print(f"Inconsistency in {name} (patient {patient_id}): NO but p_yes={p}")


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_patient_ids(path: Path):
    existing_ids = set()

    if not path.exists():
        return existing_ids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                patient_id = obj.get("patient_ID")
                if patient_id is not None:
                    existing_ids.add(patient_id)
            except Exception:
                continue

    return existing_ids


def build_chief_user_prompt(record: dict) -> str:
    doctors = record.get("doctors", {})
    full_doctors = {}

    # oppdater reputations her!
    REPUTATION_CONFIG = {
        "cautious_gp": 0,
        "conservative_gp": 1,
        "neutral_gp": 1,
        "overconfident_gp": 1
    }

    for name, d in doctors.items():
        if name not in REPUTATION_CONFIG:
            raise ValueError(f"Missing reputation for {name}")
        full_doctors[name] = {
            "decision": d.get("decision"),
            "p_yes": d.get("p_yes"),
            "reasoning": d.get("raw", ""),
            "reputation": REPUTATION_CONFIG[name]
        }

    return f"""
Patient information:
{json.dumps(record.get("input", {}), ensure_ascii=False, indent=2)}

Assessments from four GP roles:
{json.dumps(full_doctors, ensure_ascii=False, indent=2)}

Task:
Review the four GP assessments and act as the final decision-maker.
Do not simply count votes.
Weigh the panelists based on reasoning quality and consistency.
""".strip()


def run_chief_file(
    model,
    in_path: Path,
    out_path: Path,
    append_jsonl: bool = False,
    n_rows: int | None = None,
):
    rows = load_jsonl(in_path)[:n_rows] if n_rows is not None else load_jsonl(in_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not append_jsonl:
        with open(out_path, "w", encoding="utf-8") as f:
            pass

    existing_ids = load_existing_patient_ids(out_path)
    print(f"Found {len(existing_ids)} existing patients in {out_path}")

    for i, record in enumerate(rows):
        patient_id = record.get("patient_ID")

        if patient_id in existing_ids:
            print(f"Skipping patient {patient_id} ({i+1}/{len(rows)}) - already exists")
            continue

        try:
            validate_doctors(record.get("doctors", {}), patient_id)
            user_prompt = build_chief_user_prompt(record)

            chief_text = model.generate(
                DOCTORS_GP["chief_physician_decider"]["system"],
                user_prompt,
                max_new_tokens=400
            )

            try:
                chief_obj = extract_json(chief_text)
            except Exception:
                chief_obj = {
                    "final_decision": None,
                    "final_probability_yes": None,
                    "final_rationale": None,
                    "which_panelists_influenced_me": []
                }
                validate_chief(chief_obj, patient_id)

            output_record = {
                "patient_ID": record.get("patient_ID"),
                "label": record.get("label"),
                "model": record.get("model"),
                "chief": {
                    "final_decision": chief_obj.get("final_decision"),
                    "final_probability_yes": chief_obj.get("final_probability_yes"),
                    "final_rationale": chief_obj.get("final_rationale"),
                    "which_panelists_influenced_me": chief_obj.get("which_panelists_influenced_me", []),
                    "raw": chief_text,
                }
            }

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

            print(f"Done with patient {patient_id} ({i+1}/{len(rows)})")

        except Exception as e:
            print(f"ERROR on patient {patient_id} ({i+1}/{len(rows)}): {e}")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Saved JSON:", out_path)