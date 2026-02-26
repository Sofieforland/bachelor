# parse_decision_fields
# pipeline/parsing.py

#calibrates a decision, p_value and yes/no
import re

def parse_decision_fields(text: str):
    decision = None
    p_yes = None

    # Decision (DECISION=YES eller FINAL_DECISION=YES)
    m = re.search(
        r"\b(?:FINAL_DECISION|DECISION)\s*=\s*<?\s*(YES|NO)\s*>?",
        text,
        re.IGNORECASE,
    )
    if m:
        decision = m.group(1).upper()

    # P_YES (tåler 0, 1, 0.75, 1., 0., med eller uten <>)
    m2 = re.search(
        r"\bP_YES\s*=\s*<?\s*([01](?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if m2:
        try:
            p_yes = float(m2.group(1))
            # Clamp safety
            p_yes = max(0.0, min(1.0, p_yes))
        except ValueError:
            p_yes = None

    return decision, p_yes