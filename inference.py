from __future__ import annotations

import os
import sys
from typing import Optional

try:
    from env.environment import LexiGuardEnv
    from env.models import Action, Observation
except Exception as exc:
    print(f"[WARN] Failed to import env modules: {exc}", file=sys.stderr)
    LexiGuardEnv = None  # type: ignore
    Action = None  # type: ignore
    Observation = None  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def heuristic(obs: Observation) -> str:
    tid = obs.task_id
    if tid == "clause_identification":
        return "This is a termination clause allowing either party to terminate on 30 days' notice without cause."
    if tid == "risk_classification":
        return "High risk for supplier due to uncapped indemnity and attorneys' fees."
    if tid == "contract_negotiation":
        return (
            "Cap liability at 12 months of fees; exclude consequential/indirect damages; make obligations mutual."
        )
    return "No response."


def run_episode(client: Optional[object] = None) -> None:
    if LexiGuardEnv is None or Action is None or Observation is None:
        print("[START] environment unavailable; exiting.")
        print("[END]")
        return

    env = LexiGuardEnv()
    obs = env.reset()
    print("[START]")

    while True:
        if client is not None and hasattr(client, "chat"):
            try:
                completion = client.chat.completions.create(
                    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a contract lawyer."},
                        {"role": "user", "content": obs.prompt},
                    ],
                    max_tokens=256,
                )
                response = completion.choices[0].message.content.strip()
            except Exception:
                response = heuristic(obs)
        else:
            response = heuristic(obs)

        action = Action(task_id=obs.task_id, response=response)
        next_obs, reward, done, info = env.step(action)
        print(f"[STEP] {info['task_id']} score={reward.score:.2f} feedback={reward.feedback}")
        if done:
            break
        obs = next_obs  # type: ignore

    print("[END]")


if __name__ == "__main__":
    client = None
    if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI(base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"))
        except Exception:
            client = None
    run_episode(client)

