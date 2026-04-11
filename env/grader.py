from __future__ import annotations
from typing import Tuple, Any


def _contains_any(text: str, keywords: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def _normalized_score(positive: float, negative: float, max_positive: float) -> float:
    raw = (positive - negative) / max_positive if max_positive > 0 else 0.0
    return max(0.0, min(1.0, raw))


def _get_response(action: Any) -> str:
    if hasattr(action, 'response'):
        return str(action.response).lower()
    return ""


def grade_clause_identification(action: Any) -> float:
    text = _get_response(action)
    pos, neg = 0.0, 0.0
    if "termination" in text:
        pos += 1.0
    elif _contains_any(text, ("notice",)):
        pos += 0.6
    if "without cause" in text:
        pos += 0.2
    return _normalized_score(pos, neg, 1.2)


def grade_risk_classification(action: Any) -> float:
    text = _get_response(action)
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


def grade_contract_negotiation(action: Any) -> float:
    text = _get_response(action)
    pos, neg = 0.0, 0.0
    if "cap" in text or "limit" in text:
        pos += 0.4
    if "consequential" in text:
        pos += 0.2
    if "mutual" in text:
        pos += 0.1
    return _normalized_score(pos, neg, 0.9)


def grade_compliance_check(action: Any) -> float:
    text = _get_response(action)
    pos, neg = 0.0, 0.0
    if _contains_any(text, ("non-compliant", "non_compliant", "noncompliant", "not compliant", "violates", "violation")):
        pos += 0.5
    if _contains_any(text, ("gdpr", "consent", "lawful basis", "legal basis")):
        pos += 0.2
    if _contains_any(text, ("retention", "indefinite", "storage limitation")):
        pos += 0.15
    if _contains_any(text, ("third party", "third-party", "transparency", "notice")):
        pos += 0.15
    return _normalized_score(pos, neg, 1.0)


GRADERS = {
    "clause_identification": grade_clause_identification,
    "risk_classification": grade_risk_classification,
    "contract_negotiation": grade_contract_negotiation,
    "compliance_check": grade_compliance_check,
}