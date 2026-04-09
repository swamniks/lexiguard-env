from fastapi import FastAPI
from env.environment import LexiGuardEnv
from env.models import Action
from env.grader import GRADERS
from inference import _call_llm
from openai import OpenAI
import os
import uvicorn

app = FastAPI()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

env = LexiGuardEnv()
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


# ✅ ADDED: health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# ✅ ADDED: tasks listing endpoint
@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {"id": "clause_identification", "difficulty": "easy",
             "summary": "Identify clause type (termination notice)"},
            {"id": "risk_classification", "difficulty": "medium",
             "summary": "Label supplier risk level of indemnity clause"},
            {"id": "contract_negotiation", "difficulty": "hard",
             "summary": "Redline to reduce supplier liability exposure"},
        ]
    }


# ✅ ADDED: grader endpoint (this is what the validator checks)
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
def reset():
    global current_obs, done, scores
    current_obs = env.reset()
    done = False
    scores = []
    return {
        "message": "Environment reset",
        "task": current_obs.task_id
    }


@app.post("/step")
def step():
    global env, current_obs, client, scores, done

    if current_obs is None:
        current_obs = env.reset()
        done = False
        scores = []

    if done:
        return {"error": "Episode already done. Call /reset to start again.", "done": True}

    try:
        task_id = getattr(current_obs, "task_id", "unknown")
        prompt = getattr(current_obs, "prompt", "")

        response = _call_llm(client, task_id, prompt)

        action = Action(task_id=task_id, response=response)

        current_obs, reward, done, info = env.step(action)

        scores.append(reward.score)

        return {
            "task_id": task_id,
            "response": response,
            "score": reward.score,
            "done": done
        }

    except Exception as e:
        return {"error": str(e), "done": True}


@app.get("/state")
def state():
    global scores, done
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "done": done,
        "scores": scores,
        "average_score": round(avg, 2)
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()