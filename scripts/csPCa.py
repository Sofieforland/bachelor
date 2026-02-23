import pandas as pd
import numpy as np

# -----------------------------
# Config
# ----------------------------
CSV_PATH = "../outputs/dataset_step1c.csv"



# Expected columns (your file should have these based on your pipeline)
COL_AGE = "patient_age"
COL_PSA = "psa"
COL_PSAD = "psad"
COL_VOL = "prostate_volume"
COL_LABEL = "case_csPCa"

# -----------------------------
# Helpers
# -----------------------------
def ensure_numeric(df, cols):
    """Convert columns to numeric where possible."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def label_to_binary(x):
    """Map csPCa labels to 0/1 robustly."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        # assume already 0/1
        return int(x)
    s = str(x).strip().lower()
    if s in {"1", "yes", "true", "cspca", "cs_pca", "positive"}:
        return 1
    if s in {"0", "no", "false", "non-cspca", "non_cspca", "negative"}:
        return 0
    # fallback: try numeric
    try:
        return int(float(s))
    except Exception:
        return np.nan

def fmt_mean_std(series):
    series = series.dropna()
    if len(series) == 0:
        return "NA"
    return f"{series.mean():.2f} $\\pm$ {series.std(ddof=1):.2f}"

def fmt_median_iqr(series):
    series = series.dropna()
    if len(series) == 0:
        return "NA"
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    med = series.median()
    return f"{med:.2f} ({q1:.2f}--{q3:.2f})"

def volume_bin(v):
    if pd.isna(v):
        return np.nan
    if v <= 35:
        return "≤ 35 mL"
    elif v < 50:
        return "35--50 mL"
    else:
        return "≥ 50 mL"

def fmt_n_pct(n, total):
    if total == 0:
        return "0 (0.00\\%)"
    pct = 100.0 * n / total
    return f"{n:d} ({pct:.2f}\\%)"

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Basic sanity checks
required = [COL_AGE, COL_PSA, COL_PSAD, COL_VOL, COL_LABEL]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns in CSV: {missing_cols}\nFound columns: {list(df.columns)}")

# Convert numerics
df = ensure_numeric(df, [COL_AGE, COL_PSA, COL_PSAD, COL_VOL])

# Normalize label to 0/1
df[COL_LABEL] = df[COL_LABEL].apply(label_to_binary)

# Drop rows where label missing (should usually be none in your final cohort)
df = df.dropna(subset=[COL_LABEL]).copy()
df[COL_LABEL] = df[COL_LABEL].astype(int)

# Create volume groups
df["volume_group"] = df[COL_VOL].apply(volume_bin)

# Split groups
groups = {
    "Non-csPCa": df[df[COL_LABEL] == 0].copy(),
    "csPCa": df[df[COL_LABEL] == 1].copy(),
}

# Totals per group (for percentages)
totals = {name: len(g) for name, g in groups.items()}

# -----------------------------
# Build stats table (rows)
# -----------------------------
rows = []

# Age
rows.append(("Age (years)",
             fmt_mean_std(groups["Non-csPCa"][COL_AGE]),
             fmt_mean_std(groups["csPCa"][COL_AGE])))

# PSA
rows.append(("PSA (ng/mL)",
             fmt_median_iqr(groups["Non-csPCa"][COL_PSA]),
             fmt_median_iqr(groups["csPCa"][COL_PSA])))

# Prostate volume
rows.append(("Prostate volume (mL)",
             fmt_mean_std(groups["Non-csPCa"][COL_VOL]),
             fmt_mean_std(groups["csPCa"][COL_VOL])))

# Volume bins
for bin_name in ["≤ 35 mL", "35--50 mL", "≥ 50 mL"]:
    n_non = int((groups["Non-csPCa"]["volume_group"] == bin_name).sum())
    n_cs  = int((groups["csPCa"]["volume_group"] == bin_name).sum())
    rows.append((bin_name,
                 fmt_n_pct(n_non, totals["Non-csPCa"]),
                 fmt_n_pct(n_cs, totals["csPCa"])))

# PSAd
rows.append(("PSAd (ng/mL$^2$)",
             fmt_median_iqr(groups["Non-csPCa"][COL_PSAD]),
             fmt_median_iqr(groups["csPCa"][COL_PSAD])))

summary_df = pd.DataFrame(rows, columns=["Characteristic", "Non-csPCa", "csPCa"])

# -----------------------------
# Print nicely
# -----------------------------
print("\nDescriptive statistics (final filtered cohort)\n")
print(f"Total patients: {len(df)} | Non-csPCa: {totals['Non-csPCa']} | csPCa: {totals['csPCa']}\n")
print(summary_df.to_string(index=False))

# -----------------------------
# Produce LaTeX table (copy into Overleaf)
# -----------------------------
latex_lines = []
latex_lines.append(r"\begin{table}[htbp]")
latex_lines.append(r"\centering")
latex_lines.append(r"\caption{Descriptive statistics of the final filtered cohort, stratified by diagnostic label. Values are reported as mean $\pm$ standard deviation or median (interquartile range).}")
latex_lines.append(r"\label{tab:data_characteristics}")
latex_lines.append(r"\begin{tabular}{lcc}")
latex_lines.append(r"\hline")
latex_lines.append(r"Characteristic & Non-csPCa & csPCa \\")
latex_lines.append(r"\hline")

for _, r in summary_df.iterrows():
    # Escape % already handled as \% in strings
    char = str(r["Characteristic"])
    non  = str(r["Non-csPCa"])
    cs   = str(r["csPCa"])
    latex_lines.append(f"{char} & {non} & {cs} \\\\")

latex_lines.append(r"\hline")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\end{table}")

latex_table = "\n".join(latex_lines)

print("\n\n--- LaTeX table (copy/paste into Overleaf) ---\n")
print(latex_table)
