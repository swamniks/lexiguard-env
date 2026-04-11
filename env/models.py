from __future__ import annotations
from typing import List, Optional
from openenv.core.env_server.types import Action, Observation, State


class LexiGuardAction(Action):
    task_id: str
    response: str


class LexiGuardObservation(Observation):
    text: str
    task_id: str


class LexiGuardState(State):
    task_id: str
    step: int
    max_steps: int
    history: List[str]
    done: bool