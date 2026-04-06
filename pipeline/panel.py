# pipeline/panel.py

# Kjører doctor 1
# Kjører doctor 2
# Kjører chief
# Returnerer alt

from .prompts import DOCTORS_GP, DOCTOR_TASK, CHIEF_TASK
from models.base import ModelClient

def run_panel_on_row(model: ModelClient, gp_note: str):
    opinions = {}

    for key in ["doctor_1_cautious_gp", "doctor_2_over_gp", "doctor_3_conservative_gp", "doctor_4_neutral_gp"]: 
        user_text = DOCTOR_TASK.format(note=gp_note)
        opinions[key] = model.generate(DOCTORS_GP[key]["system"], user_text, max_new_tokens=512)

    compiled = ""
    for k, v in opinions.items():
        compiled += f"--- {k} ---\n{v}\n\n"

    chief_prompt = CHIEF_TASK.format(gp_note=gp_note, compiled=compiled)
    chief = model.generate(DOCTORS_GP["chief_physician_decider"]["system"], chief_prompt, max_new_tokens=512)

    return opinions, chief