from __future__ import annotations

from typing import Dict, Tuple

from env.models import Action, Reward
from env.tasks import (
    CLAUSE_IDENTIFICATION,
    CONTRACT_NEGOTIATION,
    RISK_CLASSIFICATION,
    TASK_MAP,
    Task,
)


def _contains_any(text: str, keywords: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def _normalized_score(positive: float, negative: float, max_positive: float) -> float:
    """Convert shaped positives/penalties into [0,1] range."""
    raw = (positive - negative) / max_positive if max_positive > 0 else 0.0
    return max(0.0, min(1.0, raw))


def grade_clause_identification(action: Action) -> Reward:
    text = action.response.lower()
    pos = 0.0
    neg = 0.0
    feedback_parts = []

    if "termination" in text:
        pos += 1.0
        feedback_parts.append("Identified termination clause explicitly.")
    elif _contains_any(text, ("notice", "termination notice")):
        pos += 0.6
        feedback_parts.append("Captured notice concept but not explicit termination label.")
    else:
        feedback_parts.append("Missing termination/notice labeling.")

    if "without cause" in text or "no cause" in text:
        pos += 0.2
        feedback_parts.append("Recognized termination is without cause.")

    if _contains_any(text, ("confidentiality", "payment", "governing law", "ip")):
        neg += 0.3
        feedback_parts.append("Introduced unrelated clause types.")

    score = _normalized_score(pos, neg, max_positive=1.2)
    return Reward(
        task_id=CLAUSE_IDENTIFICATION.task_id,
        score=score,
        feedback="; ".join(feedback_parts),
        details={
            "expected": "termination",
            "components": {"label": pos, "penalty": neg},
        },
    )


def grade_risk_classification(action: Action) -> Reward:
    text = action.response.lower()
    pos = 0.0
    neg = 0.0
    feedback_parts = []

    if "high" in text:
        pos += 0.7
        feedback_parts.append("Correctly labeled as high risk.")
    elif "medium" in text:
        pos += 0.4
        neg += 0.2
        feedback_parts.append("Underestimates risk; unlimited indemnity is higher.")
    elif "low" in text:
        pos += 0.15
        neg += 0.35
        feedback_parts.append("Significant underestimation of risk.")
    else:
        feedback_parts.append("Did not map to low/medium/high.")

    if _contains_any(text, ("unlimited", "no cap", "without limitation")):
        pos += 0.15
        feedback_parts.append("Noted absence of a liability cap.")

    if "attorneys' fees" in text or "legal fees" in text:
        pos += 0.05
        feedback_parts.append("Flagged fee-shifting exposure.")

    if _contains_any(text, ("cap", "limit", "liability cap")):
        pos += 0.05
        feedback_parts.append("Suggested or noted a cap (mitigation).")

    if _contains_any(text, ("low risk", "minimal risk")) and "high" not in text:
        neg += 0.1
        feedback_parts.append("Contradictory low-risk language.")

    score = _normalized_score(pos, neg, max_positive=1.05)
    return Reward(
        task_id=RISK_CLASSIFICATION.task_id,
        score=score,
        feedback="; ".join(feedback_parts),
        details={
            "expected": "high",
            "components": {"positive": round(pos, 2), "penalty": round(neg, 2)},
        },
    )


def grade_contract_negotiation(action: Action) -> Reward:
    text = action.response.lower()
    pos = 0.0
    neg = 0.0
    feedback_parts = []

    cap_keywords = ("cap", "capped", "limit", "maximum", "12 months", "twelve months", "fees")
    if _contains_any(text, cap_keywords):
        pos += 0.4
        feedback_parts.append("Introduced or clarified a liability cap.")

    if _contains_any(text, ("consequential", "indirect", "punitive", "special")):
        pos += 0.2
        feedback_parts.append("Excluded consequential/indirect damages.")

    if "mutual" in text or "both parties" in text:
        pos += 0.1
        feedback_parts.append("Added mutuality to obligations.")

    if _contains_any(text, ("direct damages", "direct losses")):
        pos += 0.1
        feedback_parts.append("Limited liability to direct damages.")

    if _contains_any(text, ("indemnity only", "indemnification only", "confidentiality only")):
        pos += 0.1
        feedback_parts.append("Kept carveouts narrow (indemnity/confidentiality).")

    if _contains_any(text, ("all damages", "any damages", "unlimited")):
        neg += 0.3
        feedback_parts.append("Expanded liability instead of limiting it.")

    if _contains_any(text, ("including data incidents", "including data breaches", "including ip")):
        neg += 0.2
        feedback_parts.append("Broadened carveouts unnecessarily.")

    score = _normalized_score(pos, neg, max_positive=0.9)
    if score == 0.0 and pos == 0.0:
        feedback_parts.append("No concrete protective edits detected.")

    return Reward(
        task_id=CONTRACT_NEGOTIATION.task_id,
        score=score,
        feedback="; ".join(feedback_parts),
        details={
            "cap": _contains_any(text, cap_keywords),
            "excluded_consequential": _contains_any(
                text, ("consequential", "indirect", "punitive", "special")
            ),
            "mutuality": "mutual" in text or "both parties" in text,
            "positive": round(pos, 2),
            "penalty": round(neg, 2),
        },
    )


GRADERS = {
    CLAUSE_IDENTIFICATION.task_id: grade_clause_identification,
    RISK_CLASSIFICATION.task_id: grade_risk_classification,
    CONTRACT_NEGOTIATION.task_id: grade_contract_negotiation,
}


def grade(action: Action) -> Reward:
    task: Task = TASK_MAP[action.task_id]
    grader = GRADERS[task.task_id]
    return grader(action)
