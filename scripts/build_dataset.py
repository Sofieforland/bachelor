# Finn og pek på riktige bilder
from pathlib import Path  # Importerer Path for å håndtere fil- og mappestier på en trygg/portabel måte (bedre enn "vanlige" strenger)
import pandas as pd       # Importerer pandas for å lese/arbeide med tabeller (CSV) som DataFrames

# -----------------------
# Paths
# -----------------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # Finner "roten" til repoet ditt: __file__ er denne .py-filen, resolve() gir absolutt sti, parents[1] går to nivå opp (scripts/ -> repo-rot)
IMAGES_ROOT = REPO_ROOT / "data" / "images"      # Bygger sti til bildemappen: <repo>/data/images
CLINICAL_PATH = Path(                            # Lager en Path til klinisk CSV-fil (her bruker du en absolutt sti på maskinen din)
    "/Users/sofieforland/Desktop/Bachelor/code/picai_labels/clinical_information/marksheet.csv"
)
OUT_DIR = REPO_ROOT / "outputs"                  # Mappen der vi vil lagre resultater (manifestet)
OUT_DIR.mkdir(exist_ok=True)                     # Lager outputs-mappen hvis den ikke finnes (exist_ok=True = ikke kræsje hvis den finnes)
print("REPO_ROOT:", REPO_ROOT)                   # Printer repo-roten for å verifisere at scriptet peker riktig
print("IMAGES_ROOT:", IMAGES_ROOT)               # Printer hvor scriptet tror bildene ligger (nyttig for debugging)
print("CLINICAL_PATH:", CLINICAL_PATH)           # Printer hvor klinisk fil ligger (nyttig for debugging)

# -----------------------
# 1) Load clinical info
# -----------------------
df = pd.read_csv(CLINICAL_PATH)                  # Leser klinisk CSV inn i en DataFrame (tabell i Python)
print("\nCOLUMNS IN MARKSHEET:")                 # Skriver ut overskrift for kolonneliste (debug)
print(list(df.columns))                          # Printer alle kolonnenavn slik at du ser hva de faktisk heter i filen
print("\nHEAD:")                                 # Skriver ut overskrift for "første rader" (debug)
print(df.head(3))                                # Printer de tre første radene for å sjekke at dataene ser fornuftige ut

df.columns = [c.strip() for c in df.columns]     # Fjerner ekstra mellomrom i kolonnenavn (unngår feil hvis det f.eks. står "patient_id " i filen)

# behold bare ID-kolonnene for kobling (vi filtrerer senere)  # Kommentar: vi tar bare ID-ene nå, fordi koblingen klinikk↔bilder skjer via ID-ene
# behold bare ID-kolonnene for kobling (vi filtrerer senere)  # (Dobbel kommentar – kan stå, men er bare repetisjon)
df_ids = df[["patient_id", "study_id"]].dropna().copy()  # Tar ut bare kolonnene patient_id og study_id, dropper rader som mangler en av dem, og copy() for å jobbe trygt videre
df_ids = df_ids.rename(columns={"patient_id": "patient_ID", "study_id": "study_ID"})  # Gir kolonnene standard-navn som resten av koden forventer (pasient/studie-ID)

df_ids["patient_ID"] = df_ids["patient_ID"].astype(str)  # Konverterer patient_ID til string for å sikre at matching mot filnavn (som tekst) blir riktig
df_ids["study_ID"] = df_ids["study_ID"].astype(str)      # Konverterer study_ID til string av samme grunn (unngår at 1000000 tolkes som int og ikke matcher tekst)
df_ids = df_ids.drop_duplicates()                        # Fjerner duplikate (patient_ID, study_ID)-par slik at hver case bare finnes én gang

print("\nClinical unique (patient_ID, study_ID):", len(df_ids))  # Printer hvor mange unike caser vi har i klinisk tabell (sjekkpunkt)

# -----------------------
# 2) Index image files -> map (patient_ID, study_ID) to paths
# -----------------------
# File pattern we observed:                       # Kommentar: vi har observert hvordan filene er navngitt og bruker det til å parse ID-er og modality
# foldX/<patient_ID>/<patient_ID>_<study_ID>_t2w.mha  # T2W-filen: inneholder patient_ID, study_ID og modality "t2w"
# foldX/<patient_ID>/<patient_ID>_<study_ID>_adc.mha  # ADC-filen: inneholder patient_ID, study_ID og modality "adc"

t2w_map = {}                                      # Dictionary som skal mappe (patient_ID, study_ID) -> filsti til T2W
adc_map = {}                                      # Dictionary som skal mappe (patient_ID, study_ID) -> filsti til ADC

# bare .mha (slik dine ser ut), men du kan utvide hvis nødvendig  # Kommentar: datasetet ditt bruker .mha, derfor leter vi bare etter .mha
image_files = list(IMAGES_ROOT.rglob("*.mha"))     # Finner ALLE .mha-filer under IMAGES_ROOT (rekursivt gjennom fold0–fold4 osv.)
print("Found .mha files:", len(image_files))       # Printer hvor mange bildefiler som ble funnet (sjekkpunkt)

for p in image_files:                              # Går gjennom hver bildefil én og én
    name = p.name.lower()                          # Tar bare filnavnet (uten mapper) og gjør det til lowercase for enklere matching/parsing

    # eksempel: 10268_1000272_t2w.mha              # Kommentar: forventet navneformat <patientID>_<studyID>_<modality>.mha
    parts = name.replace(".mha", "").split("_")    # Fjerner endelsen og splitter på "_" slik at vi får [pid, sid, modality]
    if len(parts) < 3:                             # Hvis filnavnet ikke passer forventet format (for få deler), hopp over
        continue                                   # Hopper til neste fil

    pid, sid, modality = parts[0], parts[1], parts[2]  # Leser ut patient_ID, study_ID og modality fra filnavnet

    key = (pid, sid)                               # Lager nøkkel som tuple: (patient_ID, study_ID) for bruk i dictionary
    if modality == "t2w":                          # Hvis denne filen er en T2W
        t2w_map[key] = str(p)                      # Lagre full filsti (som tekst) i t2w_map for denne casen
    elif modality == "adc":                        # Hvis denne filen er en ADC
        adc_map[key] = str(p)                      # Lagre full filsti (som tekst) i adc_map for denne casen

# -----------------------
# 3) Merge clinical IDs + image paths into manifest
# -----------------------
df_manifest = df_ids.copy()                        # Starter manifestet som en kopi av kliniske ID-par (én rad per case)
df_manifest["t2w_path"] = df_manifest.apply(       # Lager en ny kolonne t2w_path ved å slå opp riktig filsti i t2w_map for hver rad
    lambda r: t2w_map.get((r["patient_ID"], r["study_ID"])), axis=1  # get() returnerer None hvis nøkkelen ikke finnes (tryggere enn direkte indexing)
)
df_manifest["adc_path"] = df_manifest.apply(       # Lager en ny kolonne adc_path ved å slå opp riktig filsti i adc_map for hver rad
    lambda r: adc_map.get((r["patient_ID"], r["study_ID"])), axis=1  # Samme prinsipp: match på (patient_ID, study_ID)
)

print("\nManifest rows:", len(df_manifest))        # Printer antall rader i manifestet (skal tilsvare antall unike cases fra klinikken)
print("Has T2W:", df_manifest["t2w_path"].notna().sum())  # Teller hvor mange rader som faktisk fikk en T2W-sti (ikke NaN/None)
print("Has ADC:", df_manifest["adc_path"].notna().sum())  # Teller hvor mange rader som faktisk fikk en ADC-sti
print("Has both:", (                                # Teller hvor mange rader som har både T2W og ADC
    df_manifest["t2w_path"].notna() & df_manifest["adc_path"].notna()
).sum())

# (valgfritt) behold bare de som har begge           # Kommentar: mange pipelines krever at begge modaliteter finnes for hver pasient/case
df_manifest_both = df_manifest[                    # Filtrerer manifestet
    df_manifest["t2w_path"].notna() & df_manifest["adc_path"].notna()  # Beholder bare rader der begge finnes
].copy()                                           # copy() for å unngå SettingWithCopy-problemer og jobbe trygt videre
print("Keeping only both modalities:", len(df_manifest_both))  # Printer hvor mange cases som gjenstår etter dette kravet

# -----------------------
# 4) Save manifest
# -----------------------
out_path = OUT_DIR / "manifest_linked.csv"         # Setter filnavn og sti for output-manifestet i outputs-mappen
df_manifest_both.to_csv(out_path, index=False)     # Lagrer manifestet som CSV (index=False = ikke lag pandas-indeksen som ekstra kolonne)
print("\nSaved:", out_path)                        # Printer hvor filen ble lagret

print("\nExample rows:")                           # Overskrift for eksempelrader
print(df_manifest_both.head(5))                    # Printer de 5 første radene så du ser at kolonnene ser riktige ut
