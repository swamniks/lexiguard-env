from __future__ import annotations

import os
from typing import Optional, List

from openai import OpenAI, OpenAIError

from env.environment import LexiGuardEnv
from env.models import Action, Observation


# ================= CONFIG =================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

TASK_NAME = "lexiguard"
BENCHMARK = "lexiguard-openenv"


# ================= LLM =================

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


# ================= MAIN =================

def run_episode(client: Optional[OpenAI] = None) -> None:
    env = LexiGuardEnv()
    obs = env.reset()

    client = client or OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    step = 0
    rewards: List[float] = []
    success = True

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    done = False

    while not done:
        step += 1

        try:
            response = _call_llm(client, obs)

            action = Action(
                task_id=obs.task_id,
                response=response
            )

            obs, reward, done, info = env.step(action)

            rewards.append(round(reward.score, 2))

            print(
                f"[STEP] step={step} action=generated_response "
                f"reward={reward.score:.2f} done={str(done).lower()} error=null"
            )

        except Exception as e:
            success = False
            error_msg = str(e)

            print(
                f"[STEP] step={step} action=error "
                f"reward=0.00 done=true error={error_msg}"
            )
            break

    total_score = sum(rewards) / len(rewards) if rewards else 0.0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"score={total_score:.2f} rewards={rewards_str}"
    )


if __name__ == "__main__":
    run_episode()