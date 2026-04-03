from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class Observation(BaseModel):
    """What the agent sees at each step."""

    task_id: str = Field(..., description="Identifier for the current task.")
    prompt: str = Field(..., description="The user-visible instruction or clause to analyze.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra context for the task.")


class Action(BaseModel):
    """Action produced by the policy/LLM."""

    task_id: str = Field(..., description="Identifier must match the current task.")
    response: str = Field(..., description="Free-form model output for this task.")

    @validator("response")
    def non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("response must be non-empty")
        return v.strip()


class Reward(BaseModel):
    """Structured reward with shaping signal."""

    task_id: str = Field(..., description="Identifier for the graded task.")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized reward in [0,1].")
    feedback: str = Field(..., description="Human-readable explanation of the score.")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional structured grading breakdown."
    )

