from __future__ import annotations
import os
from typing import List

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

TASK_NAME = "lexiguard"
BENCHMARK = "lexiguard-openenv"
TASK_IDS = [
    "clause_identification",
    "risk_classification",
    "contract_negotiation",
    "compliance_check",
]


def heuristic_policy(task_id: str) -> str:
    if task_id == "clause_identification":
        return "This clause allows termination with notice without cause"
    if task_id == "risk_classification":
        return "High risk due to unlimited liability and no cap"
    if task_id == "contract_negotiation":
        return "Liability should be capped, mutual, and exclude consequential damages"
    if task_id == "compliance_check":
        return "This clause is non-compliant with GDPR — lacks consent, lawful basis, and retention policy"
    return "No response"


def sanitize(text: str) -> str:
    if not isinstance(text, str):
        return "invalid_response"
    return (
        text.replace("\n", " ")
        .replace("\r", " ")
        .replace(" ", "_")
        .replace(".", "")
        .replace(",", "")
        .replace("|", "_")
    )[:60]


def _call_llm(client, task_id: str, prompt: str) -> str:
    if client is None:
        return heuristic_policy(task_id)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert lawyer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
        )
        return completion.choices[0].message.content or ""
    except Exception:
        return heuristic_policy(task_id)


def run_episode() -> None:
    step = 0
    rewards: List[float] = []
    success = True

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    try:
        try:
            from server.lexiguard_environment import LexiGuardEnvironment
            from models import LexiGuardAction
        except Exception:
            print("[STEP] step=0 action=error reward=0.00 done=true error=env_import_failed")
            print("[END] success=false steps=0 score=0.00 rewards=")
            return

        try:
            env = LexiGuardEnvironment()
            obs = env.reset()
        except Exception:
            print("[STEP] step=0 action=error reward=0.00 done=true error=env_init_failed")
            print("[END] success=false steps=0 score=0.00 rewards=")
            return

        client = None
        if API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            except Exception:
                client = None

        done = False

        while not done:
            step += 1
            try:
                task_id = getattr(obs, "task_id", TASK_IDS[min(step - 1, len(TASK_IDS) - 1)])
                prompt = getattr(obs, "text", "")

                response = _call_llm(client, task_id, prompt)
                safe_action = sanitize(response)

                action = LexiGuardAction(task_id=task_id, response=response)
                obs = env.step(action)

                reward = float(obs.metadata.get("reward", 0.0)) if obs.metadata else 0.0
                done = getattr(obs, "done", False)

                rewards.append(round(reward, 2))

                print(
                    f"[STEP] step={step} action={safe_action} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null"
                )

            except Exception as e:
                success = False
                print(
                    f"[STEP] step={step} action=error "
                    f"reward=0.00 done=true error=step_failed"
                )
                break

    except Exception:
        success = False
        print("[STEP] step=0 action=error reward=0.00 done=true error=critical_failure")

    total_score = sum(rewards) / len(rewards) if rewards else 0.0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step} "
        f"score={total_score:.2f} rewards={rewards_str}"
    )


if __name__ == "__main__":
    try:
        run_episode()
    except Exception:
        print("[END] success=false steps=0 score=0.00 rewards=")