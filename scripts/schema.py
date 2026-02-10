# scripts/schema.py
from dataclasses import dataclass
from typing import List, Dict, Any

ALLOWED_DECISIONS = {"YES", "NO"}

def validate_prediction(obj: Dict[str, Any]) -> None:
    # required keys
    required = [
        "case_ID", "role", "panel", "label",
        "Decision", "Probability_yes", "self_confidence",
        "rationale_bullets", "evidence_cited",
    ]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    if obj["Decision"] not in ALLOWED_DECISIONS:
        raise ValueError(f"Decision must be YES/NO, got {obj['Decision']}")

    py = float(obj["Probability_yes"])
    sc = float(obj["self_confidence"])
    if not (0.0 <= py <= 1.0):
        raise ValueError(f"Probability_yes out of range: {py}")
    if not (0.0 <= sc <= 1.0):
        raise ValueError(f"self_confidence out of range: {sc}")

    rb = obj["rationale_bullets"]
    if not isinstance(rb, list) or not (3 <= len(rb) <= 6) or not all(isinstance(x, str) for x in rb):
        raise ValueError("rationale_bullets must be list[str] with 3-6 items")

    ec = obj["evidence_cited"]
    if not isinstance(ec, list) or len(ec) < 1 or not all(isinstance(x, str) for x in ec):
        raise ValueError("evidence_cited must be list[str] with >=1 items")

    if obj["role"] not in ["gp", "radiology"]:
        raise ValueError("role must be gp or radiology")
    if obj["panel"] not in ["sofie", "dorte"]:
        raise ValueError("panel must be sofie or dorte")
    if obj["label"] not in [0, 1]:
        raise ValueError("label must be 0 or 1")
