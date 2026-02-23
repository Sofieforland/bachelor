# lager tekst som skal mates inn i AI-agentene
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = REPO_ROOT / "outputs" / "dataset_step1c.csv"
OUT_PATH = REPO_ROOT / "outputs" / "dataset_with_notes.csv"

df = pd.read_csv(IN_PATH)

def fmt(val):
    """Gjør NaN/None -> 'NA' og ellers pen tekst."""
    if pd.isna(val):
        return "NA"
    return str(val)

def normalize_label(val):
    """
    Gjør case_csPCa til 0/1 uansett om input er NO/YES, True/False, 0/1.
    Dette er label (ground truth).
    """
    if pd.isna(val):
        return None

    s = str(val).strip().upper()
    if s in ["YES", "Y", "1", "TRUE"]:
        return 1
    if s in ["NO", "N", "0", "FALSE"]:
        return 0

    # fallback hvis det er tall i tekstform
    try:
        return int(float(val))
    except:
        raise ValueError(f"Ukjent case_csPCa-verdi: {val}")


def build_gp_note(row) -> str:
    patient_id = fmt(row.get("patient_ID"))
    age = fmt(row.get("patient_age"))
    psa = fmt(row.get("psa"))
    volume = fmt(row.get("prostate_volume"))
    psad = fmt(row.get("psad"))
    center = fmt(row.get("center"))

    # PSAD tekstlogikk
    if psad in ["NA", None, ""]:
        psad_text = "PSA density not calculated"
    else:
        psad_text = f"PSA density {psad} ng/mL/mL"

    return (
        f"Patient ID: {patient_id}. "
        f"{age}-year-old male. First recorded visit at {center}. "
        f"PSA {psa} ng/mL. Prostate volume {volume} mL. "
        f"{psad_text}. "
        f"No prior histopathology (not RP). "
        f"No DRE or imaging findings described. "
        f"No additional clinical symptoms documented.\n\n"
        "As a general practitioner, decide whether this patient should be referred "
        "for further testing for clinically significant prostate cancer (csPCa). "
        "Return a YES or NO decision with associated probabilities."
    )


def build_radiology_note(row) -> str:
    return (
        f"Case ID: {fmt(row.get('patient_ID'))}\n"
        f"Age: {fmt(row.get('patient_age'))}\n"
        f"PSA: {fmt(row.get('psa'))} ng/mL\n"
        f"Prostate volume: {fmt(row.get('prostate_volume'))} mL\n"
        f"PSA density (PSAD): {fmt(row.get('psad'))} ng/mL/mL\n"
        f"Center: {fmt(row.get('center'))}\n"
        "Imaging available: T2W axial and ADC.\n"
        "Imaging summary: Not provided (baseline clinical-only radiology note).\n"
        "Task: As a radiologist, decide whether this case warrants biopsy for suspected csPCa.\n"
        "Return a YES/NO decision with probabilities.\n"
    )

# Lag label-kolonne 
df["label"] = df["case_csPCa"].apply(normalize_label)

# Lag notes
df["input_text_gp"] = df.apply(build_gp_note, axis=1)
df["input_text_radiology"] = df.apply(build_radiology_note, axis=1)

# Lagre
df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)

# quick sanity check
print(df[["patient_ID", "case_csPCa", "label"]].head(5).to_string(index=False))
print("\nExample GP note:\n", df["input_text_gp"].iloc[0])
print("\nExample Radiology note:\n", df["input_text_radiology"].iloc[0])
