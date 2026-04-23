import json
import re
from pathlib import Path
from typing import Optional, Tuple



"""
python3 scripts/repair_medgemma.py \
    outputs/GP_overconfident/dataset_with_medgemma_outputs.jsonl \
    outputs/GP_overconfident/dataset_with_medgemma_outputs_repaired.jsonl
"""

# Matcher ekte svar som:
# DECISION=YES P_YES=0.85
# DECISION=<YES> P_YES=<0.8>
#
# Matcher IKKE placeholder som:
# DECISION=<YES/NO> P_YES=<0-1>
DECISION_PROB_PATTERN = re.compile(
    r"DECISION\s*=\s*<?\s*(YES|NO)\s*>?\s*"
    r"(?:,|\s)+"
    r"P_YES\s*=\s*<?\s*([01](?:\.\d+)?)\s*>?",
    flags=re.IGNORECASE,
)


def parse_last_valid_decision_prob(text: str) -> Tuple[Optional[str], Optional[float]]:
    if not text or not isinstance(text, str):
        return None, None

    matches = DECISION_PROB_PATTERN.findall(text)
    if not matches:
        return None, None

    decision, p_yes_str = matches[-1]
    try:
        p_yes = float(p_yes_str)
    except ValueError:
        return None, None

    if not (0.0 <= p_yes <= 1.0):
        return None, None

    decision = decision.upper()
    return decision, p_yes


def is_consistent(decision: Optional[str], p_yes: Optional[float]) -> bool:
    if decision is None or p_yes is None:
        return False
    if decision == "YES":
        return p_yes >= 0.5
    if decision == "NO":
        return p_yes < 0.5
    return False


def maybe_fix_agent(agent_dict: dict, role_name: str, patient_id) -> dict:
    """
    Returnerer statistikk for denne agenten.
    Oppdaterer agent_dict in-place hvis vi finner gyldig svar i raw.
    """
    stats = {
        "role": role_name,
        "patient_id": patient_id,
        "had_raw_match": False,
        "was_inconsistent_before": False,
        "was_fixed": False,
        "still_inconsistent_after": False,
    }

    if not isinstance(agent_dict, dict):
        return stats

    raw = agent_dict.get("raw", "")
    parsed_decision, parsed_p_yes = parse_last_valid_decision_prob(raw)

    old_decision = agent_dict.get("final_decision", agent_dict.get("decision"))
    old_p_yes = agent_dict.get("p_yes")

    if old_decision is not None:
        old_decision = str(old_decision).upper()

    stats["was_inconsistent_before"] = not is_consistent(old_decision, old_p_yes)

    if parsed_decision is not None and parsed_p_yes is not None:
        stats["had_raw_match"] = True

        # Finn riktige feltnavn
        if "final_decision" in agent_dict:
            current_decision_key = "final_decision"
        elif "decision" in agent_dict:
            current_decision_key = "decision"
        else:
            current_decision_key = "decision"

        current_p_key = "p_yes"

        needs_update = (
            agent_dict.get(current_decision_key) != parsed_decision
            or agent_dict.get(current_p_key) != parsed_p_yes
        )

        if needs_update:
            agent_dict[current_decision_key] = parsed_decision
            agent_dict[current_p_key] = parsed_p_yes
            stats["was_fixed"] = True

        new_decision = agent_dict.get(current_decision_key)
        new_p_yes = agent_dict.get(current_p_key)
        if new_decision is not None:
            new_decision = str(new_decision).upper()
        stats["still_inconsistent_after"] = not is_consistent(new_decision, new_p_yes)
    else:
        # ingen gyldig match i raw, da kan vi ikke sikkert reparere
        new_decision = old_decision
        new_p_yes = old_p_yes
        stats["still_inconsistent_after"] = not is_consistent(new_decision, new_p_yes)

    return stats


def repair_jsonl(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)

    total_records = 0
    chief_stats = {
        "had_raw_match": 0,
        "inconsistent_before": 0,
        "fixed": 0,
        "still_inconsistent_after": 0,
    }
    doctor_stats = {
        "had_raw_match": 0,
        "inconsistent_before": 0,
        "fixed": 0,
        "still_inconsistent_after": 0,
    }

    examples_unfixable = []
    examples_fixed = []

    with input_file.open("r", encoding="utf-8") as fin, output_file.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            total_records += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Hopper over ugyldig JSON på linje {line_num}: {e}")
                continue

            patient_id = record.get("patient_ID")

            # Fix chief
            if "chief" in record and isinstance(record["chief"], dict):
                s = maybe_fix_agent(record["chief"], "chief", patient_id)
                chief_stats["had_raw_match"] += int(s["had_raw_match"])
                chief_stats["inconsistent_before"] += int(s["was_inconsistent_before"])
                chief_stats["fixed"] += int(s["was_fixed"])
                chief_stats["still_inconsistent_after"] += int(s["still_inconsistent_after"])

                if s["was_fixed"] and len(examples_fixed) < 10:
                    examples_fixed.append((patient_id, "chief"))

                if not s["had_raw_match"] and len(examples_unfixable) < 10:
                    examples_unfixable.append((patient_id, "chief"))

            # Fix doctors
            doctors = record.get("doctors", {})
            if isinstance(doctors, dict):
                for doctor_name, doctor_dict in doctors.items():
                    if not isinstance(doctor_dict, dict):
                        continue

                    s = maybe_fix_agent(doctor_dict, doctor_name, patient_id)
                    doctor_stats["had_raw_match"] += int(s["had_raw_match"])
                    doctor_stats["inconsistent_before"] += int(s["was_inconsistent_before"])
                    doctor_stats["fixed"] += int(s["was_fixed"])
                    doctor_stats["still_inconsistent_after"] += int(s["still_inconsistent_after"])

                    if s["was_fixed"] and len(examples_fixed) < 10:
                        examples_fixed.append((patient_id, doctor_name))

                    if not s["had_raw_match"] and len(examples_unfixable) < 10:
                        examples_unfixable.append((patient_id, doctor_name))

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n=== FERDIG ===")
    print(f"Innfil:  {input_file}")
    print(f"Utfil:   {output_file}")
    print(f"Records: {total_records}")

    print("\n--- Chief ---")
    for k, v in chief_stats.items():
        print(f"{k}: {v}")

    print("\n--- Doctors ---")
    for k, v in doctor_stats.items():
        print(f"{k}: {v}")

    if examples_fixed:
        print("\nEksempler som ble fikset:")
        for patient_id, role in examples_fixed:
            print(f"  patient_ID={patient_id}, role={role}")

    if examples_unfixable:
        print("\nEksempler uten gyldig DECISION/P_YES i raw:")
        for patient_id, role in examples_unfixable:
            print(f"  patient_ID={patient_id}, role={role}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    repair_jsonl(args.input_path, args.output_path)