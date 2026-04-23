import json
import argparse
from pathlib import Path


"""
python3 scripts/merge_chief_with_GP.py \
  --gp outputs/Merged/llama_GPs.jsonl \
  --chief outputs/Reputation_0/chief_outputs_llama.jsonl \
  --output outputs/Merged_chief/rep_0/llama_with_doctors_and_chief.jsonl
"""

def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_gp_and_chief(gp_path: Path, chief_path: Path, output_path: Path):
    gp_rows = load_jsonl(gp_path)
    chief_rows = load_jsonl(chief_path)

    gp_lookup = {}
    chief_lookup = {}

    for row in gp_rows:
        patient_id = int(row["patient_ID"])
        gp_lookup[patient_id] = row

    for row in chief_rows:
        patient_id = int(row["patient_ID"])
        chief_lookup[patient_id] = row

    all_patient_ids = sorted(set(gp_lookup.keys()) | set(chief_lookup.keys()))
    merged_rows = []

    missing_gp = 0
    missing_chief = 0

    for patient_id in all_patient_ids:
        gp_row = gp_lookup.get(patient_id)
        chief_row = chief_lookup.get(patient_id)

        if gp_row is None:
            missing_gp += 1
            print(f"WARNING: patient_ID {patient_id} finnes i chief-filen, men ikke i GP-filen.")
            continue

        if chief_row is None:
            missing_chief += 1
            print(f"WARNING: patient_ID {patient_id} finnes i GP-filen, men ikke i chief-filen.")
            continue

        merged_row = {
            "patient_ID": patient_id,
            "label": gp_row.get("label", chief_row.get("label")),
            "model": chief_row.get("model", gp_row.get("model")),
            "input": gp_row.get("input"),
            "gp_note": gp_row.get("gp_note"),
            "doctors": gp_row.get("doctors", {}),
            "chief": chief_row.get("chief"),
        }

        merged_rows.append(merged_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged {len(merged_rows)} patients -> {output_path}")
    print(f"Missing in GP file: {missing_gp}")
    print(f"Missing in chief file: {missing_chief}")

    expected_doctors = {"cautious_gp", "overconfident_gp", "conservative_gp", "neutral_gp"}
    for row in merged_rows:
        pid = row["patient_ID"]
        doctors = set(row.get("doctors", {}).keys())
        missing = expected_doctors - doctors
        if missing:
            print(f"WARNING: {pid} mangler doctor outputs for {missing}")
        if row.get("chief") is None:
            print(f"WARNING: {pid} mangler chief output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gp", required=True, help="Path to merged GP JSONL file")
    parser.add_argument("--chief", required=True, help="Path to chief JSONL file")
    parser.add_argument("--output", required=True, help="Path to output merged JSONL file")

    args = parser.parse_args()

    merge_gp_and_chief(
        gp_path=Path(args.gp),
        chief_path=Path(args.chief),
        output_path=Path(args.output),
    )
