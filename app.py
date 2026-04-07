from fastapi import FastAPI
from env.environment import LexiGuardEnv
from env.models import Action
from inference import _call_llm
from openai import OpenAI
import os

app = FastAPI()

# ================= CONFIG =================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

# ================= GLOBAL ENV =================
env = LexiGuardEnv()
current_obs = None
done = False
scores = []

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ================= ROUTES =================

@app.get("/")
def home():
    return {"message": "LexiGuard OpenEnv running"}


@app.post("/reset")
def reset():
    global current_obs, done, scores

    current_obs = env.reset()
    done = False
    scores = []

    return {
        "message": "Environment reset",
        "task": current_obs.task_id,
        "difficulty": current_obs.metadata.get("difficulty")
    }


@app.post("/step")
def step():
    global current_obs, done, scores

    if done:
        return {"error": "Episode finished. Call /reset first."}

    try:
        response = _call_llm(client, current_obs)

        action = Action(
            task_id=current_obs.task_id,
            response=response
        )

        current_obs, reward, done, info = env.step(action)

        scores.append(round(reward.score, 2))

        return {
            "task": info["task_id"],
            "reward": reward.score,
            "done": done,
            "scores_so_far": scores
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/state")
def state():
    return {
        "done": done,
        "scores": scores,
        "average_score": round(sum(scores)/len(scores), 2) if scores else 0
    }