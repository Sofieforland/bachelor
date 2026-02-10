# scripts/agents.py

def system_prompt(role: str, panel: str) -> str:
    """
    role: "gp" eller "radiology"
    panel: "sofie" eller "dorte"
    """
    base = (
        "You are a clinical decision support agent.\n"
        "You MUST output valid JSON only (no extra text).\n"
        "Use only the provided note as evidence.\n"
    )

    # Panel-forskjell (konkret og lett å forklare i rapport/møte)
    if panel == "sofie":
        style = (
            "Panel style (Sofie): conservative. Only say YES when evidence clearly supports it. "
            "If key evidence is missing, lean towards NO with lower probability.\n"
        )
    elif panel == "dorte":
        style = (
            "Panel style (Dorte): risk-averse for missing csPCa. If important data is missing or uncertainty is high, "
            "lean towards YES with moderate probability.\n"
        )
    else:
        raise ValueError("panel must be 'sofie' or 'dorte'")

    if role == "gp":
        task = (
            "Role: General practitioner (GP).\n"
            "Decision means: refer for further testing for csPCa (YES/NO).\n"
        )
    elif role == "radiology":
        task = (
            "Role: Radiologist.\n"
            "Decision means: recommend biopsy for suspected csPCa (YES/NO).\n"
        )
    else:
        raise ValueError("role must be 'gp' or 'radiology'")

    return base + style + task


def user_prompt(note_text: str, case_id: str, role: str, panel: str) -> str:
    return (
        f"case_ID: {case_id}\n"
        f"role: {role}\n"
        f"panel: {panel}\n\n"
        "Clinical note:\n"
        f"{note_text}\n\n"
        "Return JSON with fields:\n"
        "- Decision (YES/NO)\n"
        "- Probability_yes (0-1)\n"
        "- self_confidence (0-1)\n"
        "- rationale_bullets (3-6 bullets)\n"
        "- evidence_cited (list of which fields from the note you used)\n"
    )
