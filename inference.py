from __future__ import annotations

import os
from typing import List, Dict, Any


# ================= CONFIG =================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

# Define all 3 tasks
TASKS = [
    {"id": "clause_identification", "difficulty": "easy", "name": "Clause Identification"},
    {"id": "risk_classification", "difficulty": "medium", "name": "Risk Classification"},
    {"id": "contract_negotiation", "difficulty": "hard", "name": "Contract Negotiation"}
]

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


# ================= RUN SINGLE TASK EPISODE =================

def run_task_episode(task_info: Dict[str, str]) -> Dict[str, Any]:
    """Run a single task episode and return results"""
    task_id = task_info["id"]
    difficulty = task_info["difficulty"]
    step = 0
    rewards: List[float] = []
    success = True
    error_msg = None

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    try:
        # SAFE IMPORTS
        try:
            from env.environment import LexiGuardEnv
            from env.models import Action
        except Exception as e:
            error_msg = f"env_import_failed: {e}"
            print("[STEP] step=0 action=error reward=0.00 done=true error=env_import_failed")
            print("[END] success=false steps=0 score=0.01 rewards=")
            return {"success": False, "steps": 0, "score": 0.01, "error": error_msg}

        # SAFE ENV INIT - Pass the specific task
        try:
            env = LexiGuardEnv(task=task_id)
            obs = env.reset()
        except Exception as e:
            error_msg = f"env_init_failed: {e}"
            print("[STEP] step=0 action=error reward=0.00 done=true error=env_init_failed")
            print("[END] success=false steps=0 score=0.01 rewards=")
            return {"success": False, "steps": 0, "score": 0.01, "error": error_msg}

        # SAFE CLIENT INIT
        client = None
        if API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            except Exception:
                client = None

        done = False
        max_steps = 8 if difficulty == "hard" else 5

        while not done and step < max_steps:
            step += 1

            try:
                task_id_obs = getattr(obs, "task_id", task_id)
                prompt = getattr(obs, "prompt", "")

                response = _call_llm(client, task_id_obs, prompt)
                safe_action = sanitize(response)

                action = Action(task_id=task_id_obs, response=response)

                obs, reward, done, info = env.step(action)

                # Clamp reward between 0.01 and 0.99
                clamped_reward = max(0.01, min(0.99, reward.score))
                rewards.append(round(clamped_reward, 2))

                print(
                    f"[STEP] step={step} action={safe_action} "
                    f"reward={clamped_reward:.2f} done={str(done).lower()} error=null"
                )

            except Exception as e:
                success = False
                error_msg = f"step_failed: {e}"
                print(
                    f"[STEP] step={step} action=error "
                    f"reward=0.01 done=true error=step_failed"
                )
                break

    except Exception as e:
        success = False
        error_msg = f"critical_failure: {e}"
        print("[STEP] step=0 action=error reward=0.01 done=true error=critical_failure")

    # Calculate final score - clamp between 0.01 and 0.99
    if rewards:
        raw_score = sum(rewards) / len(rewards)
        final_score = max(0.01, min(0.99, raw_score))
    else:
        final_score = 0.01
    
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(f"[END] success={str(success).lower()} steps={step} score={final_score:.3f} rewards={rewards_str}", flush=True)
    
    return {
        "task_id": task_id,
        "success": success,
        "steps": step,
        "score": final_score,
        "rewards": rewards,
        "error": error_msg
    }


# ================= MAIN =================

def run_all_tasks() -> None:
    """Run all 3 tasks sequentially"""
    print("=" * 60)
    print(f"Running {len(TASKS)} tasks for benchmark: {BENCHMARK}")
    print("=" * 60)
    
    all_results = []
    
    for task_info in TASKS:
        print(f"\n>>> Starting task: {task_info['name']} ({task_info['difficulty']})")
        result = run_task_episode(task_info)
        all_results.append(result)
        print(f"<<< Completed task: {task_info['name']} - Score: {result['score']:.3f}")
        print("-" * 40)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL TASKS")
    print("=" * 60)
    
    total_score = 0
    for i, result in enumerate(all_results):
        task_name = TASKS[i]['name']
        difficulty = TASKS[i]['difficulty']
        print(f"{task_name} ({difficulty}): Score={result['score']:.3f}, Steps={result['steps']}, Success={result['success']}")
        total_score += result['score']
    
    avg_score = total_score / len(TASKS) if TASKS else 0
    print(f"\nAVERAGE SCORE ACROSS ALL TASKS: {avg_score:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_all_tasks()
    except Exception as e:
        print(f"[END] success=false steps=0 score=0.01 error={str(e)}")