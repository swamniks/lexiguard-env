from __future__ import annotations

from typing import Tuple
from env.models import Action, Reward


def _contains_any(text: str, keywords: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def _normalized_score(positive: float, negative: float, max_positive: float) -> float:
    raw = (positive - negative) / max_positive if max_positive > 0 else 0.0
    # Clamp between 0.01 and 0.99 as required by validator
    return max(0.01, min(0.99, raw))


class ClauseIdentificationGrader:
    """Grader for clause_identification task"""
    
    def grade(self, action: Action, *args, **kwargs) -> float:
        text = action.response.lower()
        pos, neg = 0.0, 0.0
        if "termination" in text:
            pos += 1.0
        elif _contains_any(text, ("notice",)):
            pos += 0.6
        if "without cause" in text:
            pos += 0.2
        return _normalized_score(pos, neg, 1.2)


class RiskClassificationGrader:
    """Grader for risk_classification task"""
    
    def grade(self, action: Action, *args, **kwargs) -> float:
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
        return _normalized_score(pos, neg, 1.05)


class ContractNegotiationGrader:
    """Grader for contract_negotiation task"""
    
    def grade(self, action: Action, *args, **kwargs) -> float:
        text = action.response.lower()
        pos, neg = 0.0, 0.0
        if "cap" in text or "limit" in text:
            pos += 0.4
        if "consequential" in text:
            pos += 0.2
        if "mutual" in text:
            pos += 0.1
        return _normalized_score(pos, neg, 0.9)


# Keep GRADERS dict for backward compatibility
GRADERS = {
    "clause_identification": ClauseIdentificationGrader(),
    "risk_classification": RiskClassificationGrader(),
    "contract_negotiation": ContractNegotiationGrader(),
}


def grade(action: Action) -> Reward:
    if action.task_id not in GRADERS:
        return Reward(task_id=action.task_id, score=0.01, feedback="invalid task", details={})
    
    grader = GRADERS[action.task_id]
    score = grader.grade(action)
    # Ensure score is never 0.0 or 1.0
    score = max(0.01, min(0.99, score))
    
    return Reward(
        task_id=action.task_id,
        score=score,
        feedback=f"{action.task_id} grading",
        details={}
    )