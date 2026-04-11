from __future__ import annotations

from env.models import Action, Reward


def _clip(score: float) -> float:
    return max(0.0, min(1.0, score))


def clause_identification(action: Action) -> Reward:
    text = action.response.lower()
    score = 0.0
    feedback = []

    if "termination" in text:
        score += 0.7
        feedback.append("Found termination label.")
    if "notice" in text:
        score += 0.2
        feedback.append("Mentioned notice requirement.")
    if "without cause" in text or "no cause" in text:
        score += 0.1
        feedback.append("Captured without-cause aspect.")

    score = _clip(score)
    if score == 0:
        feedback.append("Did not identify termination/notice.")

    return Reward(task_id="clause_identification", score=score, feedback="; ".join(feedback), details=None)


def risk_classification(action: Action) -> Reward:
    text = action.response.lower()
    score = 0.0
    feedback = []

    if "high" in text:
        score += 0.6
        feedback.append("Labeled as high risk.")
    elif "medium" in text:
        score += 0.3
        feedback.append("Underestimates risk (should be high).")
    elif "low" in text:
        feedback.append("Marked low; significant underestimation.")

    if "unlimited" in text or "no cap" in text or "without limitation" in text:
        score += 0.2
        feedback.append("Recognized uncapped exposure.")
    if "attorneys' fees" in text or "legal fees" in text:
        score += 0.1
        feedback.append("Noted fee-shifting.")

    score = _clip(score)
    if score == 0:
        feedback.append("Missing risk label.")

    return Reward(task_id="risk_classification", score=score, feedback="; ".join(feedback), details=None)


def contract_negotiation(action: Action) -> Reward:
    text = action.response.lower()
    score = 0.0
    feedback = []

    if any(k in text for k in ("cap", "limit", "maximum", "12 months", "twelve months", "fees")):
        score += 0.35
        feedback.append("Added a liability cap.")
    if any(k in text for k in ("consequential", "indirect", "special", "punitive")):
        score += 0.25
        feedback.append("Excluded consequential damages.")
    if "mutual" in text or "both parties" in text:
        score += 0.15
        feedback.append("Made obligations mutual.")
    if "direct damages" in text:
        score += 0.1
        feedback.append("Limited to direct damages.")
    if any(k in text for k in ("all damages", "any damages", "unlimited")):
        score -= 0.2
        feedback.append("Expanded liability instead of limiting it.")

    score = _clip(score)
    if score == 0:
        feedback.append("No protective redlines found.")

    return Reward(task_id="contract_negotiation", score=score, feedback="; ".join(feedback), details=None)


GRADERS = {
    "clause_identification": clause_identification,
    "risk_classification": risk_classification,
    "contract_negotiation": contract_negotiation,
}


def grade(action: Action) -> Reward:
    return GRADERS[action.task_id](action)

