from __future__ import annotations

from typing import Tuple
from env.models import Action, Reward


def _contains_any(text: str, keywords: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def _normalized_score(positive: float, negative: float, max_positive: float) -> float:
    raw = (positive - negative) / max_positive if max_positive > 0 else 0.0
    return max(0.0, min(1.0, raw))


# ================= TASK 1 =================

def grade_clause_identification(action: Action) -> float:
    text = action.response.lower()
    pos, neg = 0.0, 0.0

    if "termination" in text:
        pos += 1.0
    elif _contains_any(text, ("notice",)):
        pos += 0.6

    if "without cause" in text:
        pos += 0.2

    return _normalized_score(pos, neg, 1.2)  # ✅ returns float


# ================= TASK 2 =================

def grade_risk_classification(action: Action) -> float:
    text = action.response.lower()
    pos, neg = 0.0, 0.0

    if "high" in text:
        pos += 0.7
    elif "medium" in text:
        pos += 0.4
    elif "low" in text:
        pos += 0.15

    if _contains_any(text, ("unlimited", "no cap")):
        pos += 0.15

    return _normalized_score(pos, neg, 1.05)  # ✅ returns float


# ================= TASK 3 =================

def grade_contract_negotiation(action: Action) -> float:
    text = action.response.lower()
    pos, neg = 0.0, 0.0

    if "cap" in text or "limit" in text:
        pos += 0.4

    if "consequential" in text:
        pos += 0.2

    if "mutual" in text:
        pos += 0.1

    return _normalized_score(pos, neg, 0.9)  # ✅ returns float


GRADERS = {
    "clause_identification": grade_clause_identification,
    "risk_classification": grade_risk_classification,
    "contract_negotiation": grade_contract_negotiation,
}


def grade(action: Action) -> Reward:
    if action.task_id not in GRADERS:
        return Reward(task_id=action.task_id, score=0.0, feedback="invalid task", details={})

    score = GRADERS[action.task_id](action)  # ✅ gets float
    return Reward(
        task_id=action.task_id,
        score=score,
        feedback=f"{action.task_id} grading",
        details={}
    )