from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from env.grader import grade
from env.models import Action, Observation, Reward
from env.tasks import TASKS, TASK_MAP, Task


class LexiGuardEnv:
    """
    Minimal OpenEnv-compatible environment for legal contract review simulation.
    """

    def __init__(self) -> None:
        self._tasks: List[Task] = TASKS
        self._current_index: int = 0
        self._history: List[Tuple[Observation, Action, Reward]] = []
        self._done: bool = False

    def reset(self) -> Observation:
        self._current_index = 0
        self._history.clear()
        self._done = False
        return self._build_observation(self._tasks[self._current_index])

    def state(self) -> Dict[str, object]:
        return {
            "current_task": (
                self._tasks[self._current_index].task_id if not self._done else None
            ),
            "step": len(self._history),
            "done": self._done,
            "cumulative_score": sum(r.score for *_o, _a, r in self._history),
        }

    def step(self, action: Action) -> Tuple[Optional[Observation], Reward, bool, Dict[str, object]]:
        if self._done:
            raise RuntimeError("Episode finished; call reset()")

        if action.task_id != self._tasks[self._current_index].task_id:
            raise ValueError("Task mismatch")

        reward = grade(action)

        self._history.append(
            (self._build_observation(TASK_MAP[action.task_id]), action, reward)
        )

        self._current_index += 1

        if self._current_index >= len(self._tasks):
            self._done = True
            next_obs = None
        else:
            next_obs = self._build_observation(self._tasks[self._current_index])

        return next_obs, reward, self._done, {"task_id": action.task_id}

    def _build_observation(self, task: Task) -> Observation:
        return Observation(
            task_id=task.task_id,
            prompt=task.prompt,
            metadata={"difficulty": task.difficulty},
        )


def make() -> LexiGuardEnv:
    return LexiGuardEnv()