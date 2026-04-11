from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Task:
    task_id: str
    name: str
    difficulty: str
    prompt: str
    description: str


TASKS: List[Task] = [
    Task(
        task_id="clause_identification",
        name="Clause Identification",
        difficulty="easy",
        prompt=(
            "Identify the clause type in this excerpt:\n"
            "\"This Agreement may be terminated by either party upon thirty (30) days' "
            "written notice to the other party without cause.\""
        ),
        description="Detect termination/notice clause.",
    ),
    Task(
        task_id="risk_classification",
        name="Risk Classification",
        difficulty="medium",
        prompt=(
            "Classify the supplier risk level (low/medium/high) of this clause:\n"
            "\"Supplier shall indemnify and hold harmless Customer from any and all claims, "
            "damages, losses, liabilities, including attorneys' fees, arising out of or "
            "related to the products or services provided hereunder, without limitation.\""
        ),
        description="Assess indemnity exposure severity.",
    ),
    Task(
        task_id="contract_negotiation",
        name="Contract Negotiation",
        difficulty="hard",
        prompt=(
            "Propose redlines to reduce supplier risk for this limitation of liability clause:\n"
            "\"Supplier's total liability under this Agreement shall not exceed the amounts "
            "paid by Customer, except that this limitation shall not apply to indemnity "
            "obligations, confidentiality breaches, or data security incidents.\""
        ),
        description="Suggest protective edits with caps/carveouts.",
    ),
]

TASK_MAP: Dict[str, Task] = {task.task_id: task for task in TASKS}

