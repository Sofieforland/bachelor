#Velg riktige pasienter og lag endelig datasett
from pathlib import Path          # Path gjør det enkelt/trygt å jobbe med filer og mapper (også på tvers av operativsystem)
import pandas as pd               # pandas brukes for å lese CSV og filtrere/merge tabeller

REPO_ROOT = Path(__file__).resolve().parents[1]  # Finner repo-roten (scripts/ -> repo-rot). Brukes for å lage stier relativt til prosjektet.

# 1) Paths
CLINICAL_PATH = Path(             # Absolutt sti til klinisk fil (marksheet.csv) på din maskin
    "/Users/sofieforland/Desktop/Bachelor/code/picai_labels/clinical_information/marksheet.csv"
)
MANIFEST_PATH = REPO_ROOT / "outputs" / "manifest_linked.csv"  # Sti til manifestet du lagde i forrige script (kobler ID -> t2w/adc paths)
OUT_PATH = REPO_ROOT / "outputs" / "dataset_step1c.csv"        # Sti til endelig filtrert datasett som dette scriptet skal lage

print("CLINICAL_PATH:", CLINICAL_PATH)  # Debug: viser hvilken clinical-fil vi leser
print("MANIFEST_PATH:", MANIFEST_PATH)  # Debug: viser hvilken manifest-fil vi leser

# 2) Load data
df = pd.read_csv(CLINICAL_PATH)         # Leser hele klinisk tabell inn i df
df.columns = [c.strip() for c in df.columns]  # Fjerner ekstra whitespace i kolonnenavn for å unngå KeyError
df = df.rename(columns={"patient_id": "patient_ID", "study_id": "study_ID"})  # Standardiserer ID-kolonnenavn til formatet vi bruker videre

manifest = pd.read_csv(MANIFEST_PATH)   # Leser manifestet (ID + t2w_path + adc_path) inn i manifest-DataFrame
manifest.columns = [c.strip() for c in manifest.columns]  # Rensker whitespace i manifest-kolonnenavn også

print("Clinical rows:", len(df))        # Debug: antall rader i klinisk tabell (før filtrering)
print("Manifest rows:", len(manifest))  # Debug: antall rader i manifestet (burde matche antall (patient_ID, study_ID) i utgangspunktet)

# 3) Keep ONLY the first visit per patient (using mri_date)
df["mri_date"] = pd.to_datetime(df["mri_date"], errors="coerce")  # Konverterer mri_date til ekte dato (feil/rare verdier blir NaT)
df = df.sort_values(["patient_ID", "mri_date", "study_ID"])        # Sorterer slik at "første visit" (tidligste dato) kommer først per pasient
df_first = df.groupby("patient_ID", as_index=False).head(1).copy() # Tar bare første rad per pasient (altså første visit). copy() for trygg videre behandling.
print("Rows after first-visit:", len(df_first))                    # Debug: antall pasienter etter at vi bare beholder første visit

# 4) Apply clinical filters

# A) patient_age available
age_ok = df_first["patient_age"].notna()  # True for rader der patient_age ikke mangler (NaN). Krav: alder må finnes.

# B) (PSA & volume) OR (PSAd & (PSA OR volume))
psa_ok = df_first["psa"].notna()                 # True hvis PSA finnes
psad_ok = df_first["psad"].notna()               # True hvis PSAd finnes
vol_ok = df_first["prostate_volume"].notna()     # True hvis prostate_volume finnes

rule1 = psa_ok & vol_ok                          # Krav-del 1: PSA OG volume finnes
rule2 = psad_ok & (psa_ok | vol_ok)              # Krav-del 2: PSAd finnes OG (PSA ELLER volume finnes)
biomarker_ok = rule1 | rule2                     # Endelig biomarkør-krav: enten rule1 eller rule2 må være oppfylt

# C) Histopath IS NOT RP (kolonnen heter histopath_type hos deg)
hist = df_first["histopath_type"].astype(str).str.strip().str.upper()  # Gjør histopath_type om til tekst, fjerner whitespace, uppercase for robust sammenligning
hist_ok = hist.ne("RP")                                               # True hvis verdien IKKE er "RP" (vi filtrerer bort RP)

df_filt = df_first[age_ok & biomarker_ok & hist_ok].copy()  # Beholder bare rader som oppfyller alle 3 krav (alder, biomarkører, histopath != RP)
print("Rows after clinical filters:", len(df_filt))          # Debug: hvor mange pasienter gjenstår etter klinisk filtrering

# 5) Merge in image paths by patient_ID + study_ID
df_merged = df_filt.merge(                                   # Slår sammen filtrert klinikk med manifestet slik at hver rad får t2w_path + adc_path
    manifest, on=["patient_ID", "study_ID"], how="inner"      # inner = behold bare de som finnes i begge (sikrer at vi bare tar pasienter vi har bilder for)
)

# ensure both modalities exist (should be true for your dataset)
df_merged = df_merged[                                       # Ekstra sikkerhet: behold bare rader som faktisk har begge bildefilstiene
    df_merged["t2w_path"].notna() & df_merged["adc_path"].notna()
].copy()
print("Rows after merging images:", len(df_merged))           # Debug: hvor mange gjenstår etter at vi har sikret at begge modaliteter er med

# 6) Select exactly what the assignment asks for
final_cols = [                                                # Liste over kolonner oppgaven ber dere ta med i endelig datasett
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
df_final = df_merged[final_cols].copy()                        # Lager endelig datasett med bare disse kolonnene (copy() for trygghet)

# 7) Save final dataset
OUT_PATH.parent.mkdir(exist_ok=True)                           # Sikrer at outputs-mappen finnes før vi lagrer (hvis ikke, lag den)
df_final.to_csv(OUT_PATH, index=False)                         # Lagrer endelig filtrert datasett som CSV (index=False = ingen ekstra index-kolonne)

print("Saved:", OUT_PATH)                                      # Printer hvor filen ble lagret
print(df_final.head(5))                                        # Printer de 5 første radene så du ser at alt ser riktig ut
