import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def attach_chief(gps_path: Path, chief_path: Path, output_path: Path):
    gp_rows = load_jsonl(gps_path)
    chief_rows = load_jsonl(chief_path)

    gp_map = {int(r["patient_ID"]): r for r in gp_rows}
    chief_map = {int(r["patient_ID"]): r for r in chief_rows}

    common_ids = sorted(set(gp_map) & set(chief_map))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pid in common_ids:
            gp = gp_map[pid]
            ch = chief_map[pid]

            merged = {
                "patient_ID": gp["patient_ID"],
                "label": gp.get("label", ch.get("label")),
                "model": ch.get("model", gp.get("model")),
                "input": gp.get("input"),
                "gp_note": gp.get("gp_note"),
                "doctors": gp.get("doctors", {}),
                "chief": ch.get("chief"),
            }

            f.write(json.dumps(merged, ensure_ascii=False) + "\n")

    print(f"Attached chief for {len(common_ids)} patients -> {output_path}")


if __name__ == "__main__":
    BASE = Path("/home/stud/sofiehf/bachelor/outputs")

#HARDKODET!! må endre modell her
    attach_chief(
        gps_path=BASE / "Merged" / "medgemma_GPs.jsonl",
        chief_path=BASE / "Reputation_1" / "chief_outputs_medgemma.jsonl",
        output_path=BASE / "Reputation_1" / "merged_medgemma.jsonl",
    )