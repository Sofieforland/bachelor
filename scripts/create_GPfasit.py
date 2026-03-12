

import pandas as pd


df = pd.read_csv("bachelor/outputs/filtered_pasients.csv")   # bytt til filnavnet ditt

def gp_decision(age, psa):
    if age <= 59:
        return "YES" if psa >= 3 else "NO"
    elif age <= 69:
        return "YES" if psa >= 4 else "No"
    elif age <= 79:
        return "YES" if psa >= 5 else "NO"
    else:
        return "YES" if psa >= 5 else "NO"

# lag ny kolonne
df["GP_fasit"] = df.apply(lambda x: gp_decision(x["patient_age"], x["psa"]), axis=1)

# lagre ny csv
df.to_csv("bachelor/outputs/filtered_pasients.csv", index=False)

print("GP_fasit column created!")