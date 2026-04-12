from fastapi import FastAPI, HTTPException
from env.environment import LexiGuardEnv
from env.models import Action, Observation
from env.grader import GRADERS
from inference import _call_llm
from openai import OpenAI
import os
import uvicorn
from typing import Optional, Dict, Any

app = FastAPI()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

# Don't create global env - will create per reset
env = None
current_obs = None
done = False
scores = []

client = None
if API_KEY:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:
        client = None


@app.get("/")
def home():
    return {"message": "LexiGuard OpenEnv running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {
                "id": "clause_identification",
                "difficulty": "easy",
                "summary": "Identify clause type (termination notice)",
                "has_grader": True
            },
            {
                "id": "risk_classification",
                "difficulty": "medium",
                "summary": "Label supplier risk level of indemnity clause",
                "has_grader": True
            },
            {
                "id": "contract_negotiation",
                "difficulty": "hard",
                "summary": "Redline to reduce supplier liability exposure",
                "has_grader": True
            },
        ]
    }


@app.get("/openenv/tasks")
def openenv_tasks():
    """OpenEnv required endpoint to list available tasks with grader paths"""
    return {
        "tasks": [
            {
                "id": "clause_identification",
                "difficulty": "easy",
                "max_steps": 5,
                "grader": "env.grader:grade_clause_identification"
            },
            {
                "id": "risk_classification",
                "difficulty": "medium",
                "max_steps": 5,
                "grader": "env.grader:grade_risk_classification"
            },
            {
                "id": "contract_negotiation",
                "difficulty": "hard",
                "max_steps": 8,
                "grader": "env.grader:grade_contract_negotiation"
            }
        ]
    }


@app.post("/grader")
def grader(action: Action):
    if action.task_id not in GRADERS:
        return {"score": 0.0, "feedback": "invalid task_id"}
    score = GRADERS[action.task_id](action)
    return {
        "task_id": action.task_id,
        "score": round(score, 4),
        "feedback": f"{action.task_id} graded successfully"
    }


@app.post("/reset")
def reset(task: Optional[str] = None):
    global env, current_obs, done, scores
    # Create new env with specific task if provided
    env = LexiGuardEnv(task=task) if task else LexiGuardEnv()
    current_obs = env.reset()
    done = False
    scores = []
    
    # Return in standard OpenEnv format
    return {
        "observation": {
            "task_id": current_obs.task_id,
            "prompt": current_obs.prompt,
            "metadata": current_obs.metadata if hasattr(current_obs, 'metadata') else {}
        }
    }


@app.post("/step")
def step(action: Optional[Dict[str, Any]] = None):
    global env, current_obs, client, scores, done

    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    if done:
        raise HTTPException(status_code=400, detail="Episode already done. Call /reset to start again.")

    try:
        # If action is provided directly, use it; otherwise generate via LLM
        if action and "response" in action:
            task_id = action.get("task_id", getattr(current_obs, "task_id", "unknown"))
            response = action.get("response", "")
            action_obj = Action(task_id=task_id, response=response)
        else:
            # Generate action using LLM
            task_id = getattr(current_obs, "task_id", "unknown")
            prompt = getattr(current_obs, "prompt", "")
            response = _call_llm(client, task_id, prompt)
            action_obj = Action(task_id=task_id, response=response)

        # Execute step
        next_obs, reward, done, info = env.step(action_obj)
        
        # Update global state
        current_obs = next_obs
        scores.append(reward.score)

        # Return in standard OpenEnv format
        return {
            "observation": {
                "task_id": next_obs.task_id,
                "prompt": next_obs.prompt,
                "metadata": next_obs.metadata if hasattr(next_obs, 'metadata') else {}
            },
            "reward": reward.score,
            "done": done,
            "info": info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    global env, scores, done
    
    if env is None:
        return {
            "state": {},
            "done": True,
            "reward_total": 0.0,
            "reward_average": 0.0,
            "steps": 0
        }
    
    env_state = env.state()
    avg = sum(scores) / len(scores) if scores else 0.0
    
    # Return in standard OpenEnv format
    return {
        "state": env_state,
        "done": done,
        "reward_total": sum(scores),
        "reward_average": round(avg, 2),
        "steps": len(scores)
    }


@app.get("/info")
def info():
    """Additional endpoint that validator might expect"""
    return {
        "name": "lexiguard",
        "version": "1.0.0",
        "description": "Legal contract review simulation",
        "tasks": ["clause_identification", "risk_classification", "contract_negotiation"]
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

    