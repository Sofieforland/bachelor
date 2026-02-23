from pathlib import Path
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Paths (hardkodet til meg, må endres)

BASE_DIR = Path(__file__).resolve().parent.parent
CLINICAL_PATH = BASE_DIR / "code" / "marksheet.csv"
MANIFEST_PATH = REPO_ROOT / "outputs" / "manifest_linked.csv"
OUT_PATH = REPO_ROOT / "outputs" / "dataset_step1c.csv"

print("CLINICAL_PATH:", CLINICAL_PATH)
print("MANIFEST_PATH:", MANIFEST_PATH)

# Load data
df = pd.read_csv(CLINICAL_PATH)
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"patient_id": "patient_ID", "study_id": "study_ID"})

manifest = pd.read_csv(MANIFEST_PATH)
manifest.columns = [c.strip() for c in manifest.columns]

print("Clinical rows:", len(df))
print("Manifest rows:", len(manifest))

# Keep ONLY the first visit per patient (using mri_date)
df_all = df.copy()
df_all["mri_date"] = pd.to_datetime(df_all["mri_date"], errors="coerce")
df_sorted = df_all.sort_values(["patient_ID", "mri_date", "study_ID"])
df_first = df_sorted.groupby("patient_ID", as_index=False).head(1).copy()

print("\n--- FLOW COUNTS ---")
print("Start (rows in clinical file):", len(df_all))
print("After first-visit:", len(df_first))
print("Excluded (multiple visits):", len(df_all) - len(df_first))


#patient_age available
step0 = df_first
step1 = step0[step0["patient_age"].notna()].copy()
print("\nAfter age filter:", len(step1))
print("Excluded (missing patient_age):", len(step0) - len(step1))

# biomarker rule:
# (PSA & volume) OR (PSAd & (PSA OR volume))
psa_ok = step1["psa"].notna()
psad_ok = step1["psad"].notna()
vol_ok = step1["prostate_volume"].notna()

rule1 = psa_ok & vol_ok
rule2 = psad_ok & (psa_ok | vol_ok)
biomarker_ok = rule1 | rule2

step2 = step1[biomarker_ok].copy()
print("\nAfter biomarker rule:", len(step2))
print("Excluded (fails PSA/PSAd/volume rule):", len(step1) - len(step2))

# Histopath IS NOT RP (and histopath must be available)
hist_raw = step2["histopath_type"]
hist = hist_raw.astype(str).str.strip().str.upper()

# ekstra statistikk for flow/rapport 
is_missing = hist_raw.isna()
is_rp = hist.eq("RP")

print("\nHistopath breakdown:")
print("  - missing histopath:", is_missing.sum())
print("  - histopath == RP:", is_rp.sum())
print("  - missing OR RP:", (is_missing | is_rp).sum())

step3 = step2[hist_raw.notna() & hist.ne("RP")].copy()
print("\nAfter histopath != RP (and not missing):", len(step3))
print("Excluded (histopath missing or RP):", len(step2) - len(step3))



# filtrerte kliniske sett
df_filt = step3
print("\nRows after clinical filters (total):", len(df_filt))

# Merge in image paths by patient_ID + study_ID
df_merged = df_filt.merge(manifest, on=["patient_ID", "study_ID"], how="inner")

# ensure both modalities exist (should be true)
df_merged = df_merged[df_merged["t2w_path"].notna() & df_merged["adc_path"].notna()].copy()

print("\nAfter merging images:", len(df_merged))
print("Excluded (missing images):", len(df_filt) - len(df_merged))


final_cols = [
    "patient_ID",
    "patient_age",
    "psa",
    "psad",
    "prostate_volume",
    "case_csPCa",
    "center",
    "t2w_path",
    "adc_path",
]
df_final = df_merged[final_cols].copy()

# klasse fordeling
print("\nClass distribution (case_csPCa):")
print(df_final["case_csPCa"].value_counts(dropna=False))

# 7) Save final dataset
OUT_PATH.parent.mkdir(exist_ok=True)
df_final.to_csv(OUT_PATH, index=False)

print("\nSaved:", OUT_PATH)
print(df_final.head(5))
