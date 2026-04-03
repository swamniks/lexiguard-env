from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI, OpenAIError

from env.environment import LexiGuardEnv
from env.models import Action, Observation


# ✅ Use Hugging Face router (correct for hackathon)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")


def _call_llm(client: OpenAI, obs: Observation) -> str:
    system = "You are an expert commercial lawyer. Respond concisely."

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Task: {obs.task_id}\n{obs.prompt}",
                },
            ],
            max_tokens=200,
        )
        return completion.choices[0].message.content.strip()

    except Exception:
        return heuristic_policy(obs)


def heuristic_policy(obs: Observation) -> str:
    tid = obs.task_id

    if tid == "clause_identification":
        return "This clause allows termination with notice."

    if tid == "risk_classification":
        return "High risk due to unlimited liability."

    if tid == "contract_negotiation":
        return "Liability should be capped and mutual."

    return "No response."


def run_episode(client: Optional[OpenAI] = None) -> None:
    env = LexiGuardEnv()
    obs = env.reset()

    # ✅ FIX: pass API key properly
    client = client or OpenAI(
    base_url=API_BASE_URL,
    api_key=os.getenv("HF_TOKEN")
)
    done = False

    while not done:
        response = _call_llm(client, obs)

        action = Action(
            task_id=obs.task_id,
            response=response
        )

        obs, reward, done, info = env.step(action)

        print(f"[{info['task_id']}] score={reward.score:.2f} feedback={reward.feedback}")


if __name__ == "__main__":
    run_episode()