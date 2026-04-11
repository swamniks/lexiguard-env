from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class Observation(BaseModel):
    """Agent-visible prompt and context."""

    task_id: str = Field(..., description="Current task identifier.")
    prompt: str = Field(..., description="Instruction or clause text.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """Agent response for a task."""

    task_id: str = Field(..., description="Target task identifier.")
    response: str = Field(..., description="Free-form model output.")

    @validator("response")
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("response must be non-empty")
        return v.strip()


class Reward(BaseModel):
    """Normalized reward with explanation."""

    task_id: str = Field(..., description="Graded task id.")
    score: float = Field(..., ge=0.0, le=1.0, description="Reward in [0,1].")
    feedback: str = Field(..., description="Human-readable feedback.")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Extra scoring signals.")

