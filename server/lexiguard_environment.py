from __future__ import annotations
from typing import Any, List, Optional
from openenv.core.env_server import Environment
from server.models import LexiGuardAction, LexiGuardObservation, LexiGuardState
from env.tasks import TASKS
from env.grader import GRADERS

class LexiGuardEnvironment(Environment[LexiGuardAction, LexiGuardObservation, LexiGuardState]):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._task_index: int = 0
        self._done: bool = False
        self._rewards: List[float] = []
        self._current_task = None

    def reset(self, seed=None, episode_id=None, **kwargs: Any) -> LexiGuardObservation:
        task_id = kwargs.get("task_id") or kwargs.get("task") or TASKS[0].task_id
        found = False
        for i, t in enumerate(TASKS):
            if t.task_id == task_id:
                self._task_index = i
                found = True
                break
        if not found:
            self._task_index = 0
        self._current_task = TASKS[self._task_index]
        self._done = False
        self._rewards = []
        return LexiGuardObservation(
            text=self._current_task.prompt,
            task_id=self._current_task.task_id,
        )

    def step(self, action: LexiGuardAction) -> LexiGuardObservation:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is finished. Call reset().")
        if action.task_id in GRADERS:
            reward = GRADERS[action.task_id](action)
        else:
            reward = 0.0
        self._rewards.append(reward)
        self._task_index += 1
        self._done = self._task_index >= len(TASKS)
        if not self._done:
            self._current_task = TASKS[self._task_index]
            next_text = self._current_task.prompt
            next_task_id = self._current_task.task_id
        else:
            next_text = ""
            next_task_id = action.task_id
        return LexiGuardObservation(
            text=next_text,
            task_id=next_task_id,
            reward=reward,
            done=self._done,
            metadata={
                "step": self._task_index,
                "rewards_so_far": self._rewards,
            }
        )

    @property
    def state(self) -> LexiGuardState:
        return LexiGuardState(
            task_id=self._current_task.task_id if self._current_task else "",
            step=self._task_index,
            max_steps=len(TASKS),
            history=[str(r) for r in self._rewards],
            done=self._done,
        )