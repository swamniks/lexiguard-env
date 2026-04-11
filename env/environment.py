from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from env.grader import grade
from env.models import Action, Observation, Reward
from env.tasks import TASKS, TASK_MAP, Task


class LexiGuardEnv:
    def __init__(self) -> None:
        self._tasks: List[Task] = TASKS
        self._index: int = 0
        self._done: bool = False

    def reset(self) -> Observation:
        self._index = 0
        self._done = False
        return self._observation(self._tasks[self._index])

    def state(self) -> Dict[str, object]:
        return {
            "current_task": None if self._done else self._tasks[self._index].task_id,
            "done": self._done,
            "position": self._index,
        }

    def step(self, action: Action) -> Tuple[Optional[Observation], Reward, bool, Dict[str, object]]:
        if self._done:
            raise RuntimeError("Episode completed. Call reset().")

        reward = grade(action)
        info: Dict[str, object] = {"task_id": action.task_id}

        self._index += 1
        if self._index >= len(self._tasks):
            self._done = True
            next_obs: Optional[Observation] = None
        else:
            next_obs = self._observation(self._tasks[self._index])

        return next_obs, reward, self._done, info

    def _observation(self, task: Task) -> Observation:
        return Observation(task_id=task.task_id, prompt=task.prompt, metadata={"difficulty": task.difficulty})


def make() -> LexiGuardEnv:
    return LexiGuardEnv()

