from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Task:
    task_id: str
    name: str
    difficulty: str
    prompt: str
    rubric: Dict[str, str]
    description: str


CLAUSE_IDENTIFICATION = Task(
    task_id="clause_identification",
    name="Clause Identification",
    difficulty="easy",
    prompt=(
        "Identify the clause type in the following excerpt:\n"
        "\"This Agreement may be terminated by either party upon thirty (30) days' "
        "written notice to the other party without cause.\""
    ),
    rubric={
        "termination": "Correct label expected.",
        "notice": "Acceptable as near-synonym for termination notice clause.",
    },
    description="Detects whether the model can name the legal clause that governs termination.",
)

RISK_CLASSIFICATION = Task(
    task_id="risk_classification",
    name="Risk Classification",
    difficulty="medium",
    prompt=(
        "Classify the risk level (low/medium/high) of this indemnity clause for the supplier:\n"
        "\"Supplier shall indemnify and hold harmless Customer from any and all claims, "
        "damages, losses, liabilities, including attorneys' fees, arising out of or related "
        "to the products or services provided hereunder, without limitation.\""
    ),
    rubric={"high": "Unlimited indemnity with fee coverage is high risk for supplier."},
    description="Assesses whether the model can recognize risk severity.",
)

CONTRACT_NEGOTIATION = Task(
    task_id="contract_negotiation",
    name="Contract Negotiation",
    difficulty="hard",
    prompt=(
        "Propose a redlined revision to reduce supplier risk for this limitation of liability clause.\n"
        "\"Supplier's total liability under this Agreement shall not exceed the amounts paid by "
        "Customer, except that this limitation shall not apply to indemnity obligations, "
        "confidentiality breaches, or data security incidents.\""
    ),
    rubric={
        "cap": "Introduce a monetary cap (e.g., 12 months fees).",
        "carveouts": "Keep limited carveouts; avoid expanding liability.",
        "mutual": "Prefer making obligations mutual where appropriate.",
        "direct_damages": "Restrict to direct damages; exclude consequential/punitive.",
    },
    description="Tests ability to negotiate and draft safer language with concrete changes.",
)

# ✅ NEW: 4th task — compliance check
COMPLIANCE_CHECK = Task(
    task_id="compliance_check",
    name="Compliance Check",
    difficulty="medium",
    prompt=(
        "Review the following data processing clause for GDPR and EU AI Act compliance issues:\n"
        "\"The Company may collect, store, and process all user data submitted through this platform "
        "indefinitely for any business purpose deemed appropriate by the Company, including sharing "
        "with third-party partners without prior notice to the user.\"\n"
        "Identify whether this clause is compliant or non-compliant and explain the violations."
    ),
    rubric={
        "non_compliant": "Clause lacks lawful basis, consent, purpose limitation, and retention policy.",
        "gdpr": "Should reference GDPR violations — no consent, no purpose limitation.",
        "retention": "Indefinite storage without defined retention period violates data minimisation.",
        "third_party": "Sharing with third parties without notice violates transparency obligations.",
    },
    description="Tests ability to identify GDPR and EU AI Act violations in a data processing clause.",
)


TASKS: List[Task] = [
    CLAUSE_IDENTIFICATION,
    RISK_CLASSIFICATION,
    CONTRACT_NEGOTIATION,
    
]

TASK_MAP: Dict[str, Task] = {t.task_id: t for t in TASKS}