from __future__ import annotations

import os
from typing import List


# ================= CONFIG =================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

# Change this to use specific task ID instead of "lexiguard"
TASK_NAME = os.getenv("LEXIGUARD_TASK", "clause_identification")
BENCHMARK = "lexiguard"


# ================= FALLBACK =================

def heuristic_policy(task_id: str) -> str:
    if task_id == "clause_identification":
        return "This clause allows termination with notice"

    if task_id == "risk_classification":
        return "High risk due to unlimited liability"

    if task_id == "contract_negotiation":
        return "Liability should be capped and mutual"

    return "No response"


# ================= SAFE STRING =================

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


# ================= SAFE LLM =================

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


# ================= MAIN =================

def run_episode() -> None:
    step = 0
    rewards: List[float] = []
    success = True

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    try:
        # 🔥 SAFE IMPORTS (CRITICAL FIX)
        try:
            from env.environment import LexiGuardEnv
            from env.models import Action
        except Exception:
            print("[STEP] step=0 action=error reward=0.00 done=true error=env_import_failed")
            print("[END] success=false steps=0 score=0.00 rewards=")
            return

        # 🔥 SAFE ENV INIT - Pass the specific task
        try:
            env = LexiGuardEnv(task=TASK_NAME)
            obs = env.reset()
        except Exception:
            print("[STEP] step=0 action=error reward=0.00 done=true error=env_init_failed")
            print("[END] success=false steps=0 score=0.00 rewards=")
            return

        # 🔥 SAFE CLIENT INIT
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
                task_id = getattr(obs, "task_id", "unknown")
                prompt = getattr(obs, "prompt", "")

                response = _call_llm(client, task_id, prompt)
                safe_action = sanitize(response)

                action = Action(task_id=task_id, response=response)

                obs, reward, done, info = env.step(action)

                rewards.append(round(reward.score, 2))

                print(
                    f"[STEP] step={step} action={safe_action} "
                    f"reward={reward.score:.2f} done={str(done).lower()} error=null"
                )

            except Exception:
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

    # FIXED: Removed the extra print() wrapper
    print(f"[END] success={str(success).lower()} steps={step} score={total_score:.3f} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    try:
        run_episode()
    except Exception:
        print("[END] success=false steps=0 score=0.00 rewards=")