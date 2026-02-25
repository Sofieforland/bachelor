# parse_decision_fields
# pipeline/parsing.py

#calibrates a decision, p_value and yes/no
import re

def parse_decision_fields(text: str):
    decision = None
    p_yes = None

    m = re.search(
        r"\b(?:FINAL_DECISION|DECISION)\s*=\s*<?\s*(YES|NO)\s*>?",
        text,
        re.IGNORECASE,
    )
    if m:
        decision = m.group(1).upper()

    m2 = re.search(
        r"\bP_YES\s*=\s*<?\s*([01](?:\.\d+)?)\s*>?",
        text,
        re.IGNORECASE,
    )
    if m2:
        try:
            p_yes = float(m2.group(1))
            p_yes = max(0.0, min(1.0, p_yes))
        except ValueError:
            p_yes = None

    return decision, p_yes