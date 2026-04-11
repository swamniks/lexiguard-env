from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from env.grader import grade
from env.models import Action, Observation, Reward
from env.tasks import TASKS, TASK_MAP, Task


class LexiGuardEnv:
    """
    Minimal OpenEnv-compatible environment for legal contract review simulation.

    The environment advances through three deterministic tasks. Each call to
    `step` grades the provided Action and moves to the next task until all are done.
    """

    def __init__(self) -> None:
        self._tasks: List[Task] = TASKS
        self._current_index: int = 0
        self._history: List[Tuple[Observation, Action, Reward]] = []
        self._done: bool = False

    def reset(self) -> Observation:
        """Reset episode and return the initial observation."""
        self._current_index = 0
        self._history.clear()
        self._done = False
        return self._build_observation(self._tasks[self._current_index])

    def state(self) -> Dict[str, object]:
        """Return a lightweight snapshot of progress for logging/debugging."""
        return {
            "current_task": (
                self._tasks[self._current_index].task_id if not self._done else None
            ),
            "step": len(self._history),
            "done": self._done,
            "cumulative_score": sum(r.score for *_o, _a, r in self._history),
        }

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, object]]:
        """
        Grade an action, advance to the next task, and return (obs, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode finished; call reset() to start a new run.")

        if action.task_id != self._tasks[self._current_index].task_id:
            raise ValueError(
                f"Action.task_id={action.task_id} does not match current task "
                f"{self._tasks[self._current_index].task_id}"
            )

        reward = grade(action)
        info: Dict[str, object] = {"task_id": action.task_id}
        self._history.append((self._build_observation(TASK_MAP[action.task_id]), action, reward))

        self._current_index += 1
        if self._current_index >= len(self._tasks):
            self._done = True
            next_obs: Optional[Observation] = None
        else:
            next_obs = self._build_observation(self._tasks[self._current_index])

        return next_obs, reward, self._done, info

    # Helpers
    def _build_observation(self, task: Task) -> Observation:
        return Observation(task_id=task.task_id, prompt=task.prompt, metadata={"difficulty": task.difficulty})


# Convenience: factory for integration with OpenEnv runners
def make() -> LexiGuardEnv:
    return LexiGuardEnv()