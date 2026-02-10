# Finn og pek på riktige bilder
from pathlib import Path  
import pandas as pd      


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]  
IMAGES_ROOT = REPO_ROOT / "data" / "images"      

OUT_DIR = REPO_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

print("REPO_ROOT:", REPO_ROOT)
print("IMAGES_ROOT:", IMAGES_ROOT)
# print("CLINICAL_PATH:", CLINICAL_PATH)

t2w_map = {}  # (patient_ID, study_ID) -> t2w_path
adc_map = {}  # (patient_ID, study_ID) -> adc_path

image_files = list(IMAGES_ROOT.rglob("*.mha"))
print("Found .mha files:", len(image_files))

for p in image_files:
    name = p.name.lower()

    # forventet: <patientID>_<studyID>_<modality>.mha
    parts = name.replace(".mha", "").split("_")
    if len(parts) < 3:
        continue

    # MODALITY: bruk siste token for robusthet 
    pid, sid, modality = parts[0], parts[1], parts[-1]
    key = (pid, sid)
    if modality == "t2w":
        t2w_map[key] = str(p)
    elif modality == "adc":
        adc_map[key] = str(p)

# 2) Build manifest from IMAGE KEYS (not from clinical CSV)
both_keys = sorted(set(t2w_map.keys()) & set(adc_map.keys()))

print("\n--- IMAGE KEY COUNTS ---")
print("Unique T2W keys:", len(t2w_map))
print("Unique ADC keys:", len(adc_map))
print("Unique BOTH keys (t2w+adc):", len(both_keys))

df_manifest_both = pd.DataFrame(
    [{
        "patient_ID": pid,
        "study_ID": sid,
        "t2w_path": t2w_map[(pid, sid)],
        "adc_path": adc_map[(pid, sid)],
    } for (pid, sid) in both_keys]
)

out_path = OUT_DIR / "manifest_linked.csv"
df_manifest_both.to_csv(out_path, index=False)
print("\nSaved:", out_path)

print("\nExample rows:")
print(df_manifest_both.head(5))
