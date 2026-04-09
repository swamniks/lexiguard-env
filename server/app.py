from fastapi import FastAPI
from env.environment import LexiGuardEnv
from env.models import Action
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

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


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
        "task": current_obs.task_id
    }


@app.post("/step")
def step():
    global env, current_obs, client

    try:
        # Extract safely
        task_id = getattr(current_obs, "task_id", "unknown")
        prompt = getattr(current_obs, "prompt", "")

        # 🔥 FIX: correct function call
        response = _call_llm(client, task_id, prompt)

        action = Action(
            task_id=task_id,
            response=response
        )

        current_obs, reward, done, info = env.step(action)

        return {
            "task_id": task_id,
            "response": response,
            "score": reward.score,
            "done": done
        }

    except Exception as e:
        return {
            "error": str(e),
            "done": True
        }

@app.get("/state")
def state():
    return {
        "done": done,
        "scores": scores,
        "average_score": round(sum(scores)/len(scores), 2) if scores else 0
    }


# ✅ REQUIRED BY OPENENV
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()