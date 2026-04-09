import json
from pathlib import Path
import pandas as pd


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe(x):
    if pd.isna(x):
        return None
    return x


def build_csv_lookup(csv_path: Path):
    df = pd.read_csv(csv_path)
    lookup = {}

    for _, row in df.iterrows():
        pid = int(row["patient_ID"])

        lookup[pid] = {
            "label": None if pd.isna(row.get("label")) else int(row.get("label")),
            "input": {
                "patient_ID": pid,
                "patient_age": safe(row.get("patient_age")),
                "psa": safe(row.get("psa")),
                "prostate_volume": safe(row.get("prostate_volume")),
                "psad": safe(row.get("psad")),
                "center": safe(row.get("center")),
            },
            "gp_note": safe(row.get("input_text_gp")),
        }

    return lookup


def merge_doctor_files(input_paths: list[Path], output_path: Path, csv_path: Path):
    merged = {}
    csv_lookup = build_csv_lookup(csv_path)

    for path in input_paths:
        rows = load_jsonl(path)

        for row in rows:
            patient_id = int(row["patient_ID"])

            if patient_id not in merged:
                csv_data = csv_lookup.get(patient_id, {})

                merged[patient_id] = {
                    "patient_ID": patient_id,
                    "label": csv_data.get("label"),
                    "model": row.get("model"),
                    "input": csv_data.get("input"),
                    "gp_note": csv_data.get("gp_note"),
                    "doctors": {},
                }

            # merge doctors
            doctors = row.get("doctors", {})
            for doctor_name, doctor_data in doctors.items():
                merged[patient_id]["doctors"][doctor_name] = doctor_data

    # sanity check
    expected = {"cautious_gp", "overconfident_gp", "conservative_gp", "neutral_gp"}
    for pid, rec in merged.items():
        missing = expected - set(rec["doctors"].keys())
        if missing:
            print(f"WARNING: {pid} mangler {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pid in sorted(merged):
            f.write(json.dumps(merged[pid], ensure_ascii=False) + "\n")

    print(f"Merged {len(merged)} patients -> {output_path}")


if __name__ == "__main__":
    input_paths = [
        Path("outputs/GP_cautious/dataset_with_medgemma_outputs.jsonl"),
        Path("outputs/GP_conservative/dataset_with_medgemma_outputs.jsonl"),
        Path("outputs/GP_neutral/dataset_with_medgemma_outputs.jsonl"),
        Path("outputs/GP_overconfident/dataset_with_medgemma_outputs.jsonl"),
    ]

    output_path = Path("outputs/Merged/medgemma_GPs.jsonl")

    csv_path = Path("outputs/dataset_with_notes.csv")  

    merge_doctor_files(input_paths, output_path, csv_path)