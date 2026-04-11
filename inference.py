from __future__ import annotations

from typing import Tuple
from env.models import Action, Reward


def _contains_any(text: str, keywords: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def _normalized_score(pos: float, neg: float, max_pos: float) -> float:
    raw = (pos - neg) / max_pos if max_pos > 0 else 0.0
    return max(0.0, min(1.0, raw))


# ---------- TASK 1 ----------
def grade_clause_identification(action: Action) -> Reward:
    text = action.response.lower()
    pos, neg = 0.0, 0.0

    if "termination" in text:
        pos += 1.0
    elif "notice" in text:
        pos += 0.6

    if "without cause" in text:
        pos += 0.2

    score = _normalized_score(pos, neg, 1.2)

    return Reward(
        task_id="clause_identification",
        score=score,
        feedback="clause grading",
        details={}
    )


# ---------- TASK 2 ----------
def grade_risk_classification(action: Action) -> Reward:
    text = action.response.lower()
    pos, neg = 0.0, 0.0

    if "high" in text:
        pos += 0.7
    elif "medium" in text:
        pos += 0.4
        neg += 0.2
    elif "low" in text:
        pos += 0.15
        neg += 0.35

    if _contains_any(text, ("unlimited", "no cap", "without limitation")):
        pos += 0.15

    if "attorneys" in text:
        pos += 0.05

    score = _normalized_score(pos, neg, 1.05)

    return Reward(
        task_id="risk_classification",
        score=score,
        feedback="risk grading",
        details={}
    )


# ---------- TASK 3 ----------
def grade_contract_negotiation(action: Action) -> Reward:
    text = action.response.lower()
    pos, neg = 0.0, 0.0

    if _contains_any(text, ("cap", "limit")):
        pos += 0.4

    if _contains_any(text, ("consequential", "indirect", "punitive")):
        pos += 0.2

    if "mutual" in text:
        pos += 0.1

    if _contains_any(text, ("direct damages", "direct losses")):
        pos += 0.1

    if _contains_any(text, ("unlimited", "all damages")):
        neg += 0.3

    score = _normalized_score(pos, neg, 0.9)

    return Reward(
        task_id="contract_negotiation",
        score=score,
        feedback="negotiation grading",
        details={}
    )


# 🔥 CRITICAL (DO NOT CHANGE)
GRADERS = {
    "clause_identification": grade_clause_identification,
    "risk_classification": grade_risk_classification,
    "contract_negotiation": grade_contract_negotiation,
}


def grade(action: Action) -> Reward:
    if action.task_id not in GRADERS:
        return Reward(
            task_id=action.task_id,
            score=0.0,
            feedback="invalid",
            details={}
        )

    return GRADERS[action.task_id](action)

if __name__ == "__main__":
    try:
        run_episode()
    except Exception:
        print("[END] success=false steps=0 score=0.00 rewards=")