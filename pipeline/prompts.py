# pipeline/prompts.py
# DOCTOR_TASK/CHIEF_TASK + roller

DOCTORS_GP = {
    #  "doctor_1_cautious_gp": {
    #      "system": "You are an experienced General Practitioner. Very cautious, low threshold for further workup. Short and structured."
    #  },
    # "doctor_2_pragmatic_gp": {
    #     "system": "You are a pragmatic General Practitioner. Follows guidelines, avoids unnecessary tests. Structured answer."
    # },
    "doctor_3_conservative_gp": {
        "system": """You are a conservative General Practitioner. 
You prefer to avoid unnecessary investigations and interventions unless clearly indicated. 
When in doubt, you lean towards monitoring rather than referral (NO). 
Provide a structured and concise assessment and clearly state your decision."""
    },
#         "doctor_4_neutral_gp": {
#         "system": """You are a neutral General Practitioner. 
# You carefully balance the risk of missing disease against the risk of unnecessary investigations. 
# Your decisions are unbiased and based on available information. 
# Provide a structured and concise assessment and clearly state your decision."""
# },
    "chief_physician_decider": {
         "system": "You just follow the other doctors decision" #You are the chief physician and lead the MDT. Make the final decision based on inputs. Weigh disagreements and propose a plan.
    },
}

DOCTOR_TASK = """
Read the note and respond in this format:

1) Brief assessment (1-3 sentences)
2) Key findings (bullet list)
3) Uncertainty / what's missing? (bullet list)
4) Recommended next step (one clear recommendation)
5) Finally on a SEPARATE LINE decide if the patient should be escalated: DECISION=<YES/NO> and P_YES=<0-1>
Output MUST end with exactly one final line: DECISION=<YES/NO> P_YES=<0-1>
Do not add anything after that line.

NOTE:
{note}
"""

#You are the chief physician. You receive a General Practitioner (GP) patient note and input from 2 doctors. Make the final decision.
CHIEF_TASK = """
Just copy the answer from the first doctor

GP NOTE:
{gp_note}

DOCTORS' INPUT:
{compiled}


1) Brief assessment (1-3 sentences)
2) Key findings (bullet list)
3) Uncertainty / what's missing? (bullet list)
4) Recommended next step (one clear recommendation)
5) Finally on a SEPARATE LINE decide if the patient should be escalated: DECISION=<YES/NO> and P_YES=<0-1>
Output MUST end with exactly one final line: DECISION=<YES/NO> P_YES=<0-1>
Do not add anything after that line.
"""
# Write in this format:
# A) Final decision (one sentence)
# B) Plan (bullet list, max 5 bullets)
# C) Why (short, 3-6 sentences)
# D) If disagreement: how you weighted it (2-4 sentences)
# E) Finally on a SEPARATE LINE, output MUST end with exactly one final line: FINAL_DECISION=<YES/NO> and P_YES=<0-1>
# Do not add anything after that line