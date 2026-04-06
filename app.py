from fastapi import FastAPI
from env.environment import LexiGuardEnv
from env.models import Action
from inference import _call_llm
from openai import OpenAI
import os

app = FastAPI()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")


def run_full_episode():
    env = LexiGuardEnv()
    obs = env.reset()

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    scores = []

    done = False
    while not done:
        response = _call_llm(client, obs)

        action = Action(task_id=obs.task_id, response=response)

        obs, reward, done, info = env.step(action)

        scores.append(round(reward.score, 2))

    return scores


@app.get("/")
def home():
    return {"message": "LexiGuard OpenEnv running"}


@app.get("/state")
def get_state():
    scores = run_full_episode()

    return {
        "tasks": [
            "clause_identification",
            "risk_classification",
            "contract_negotiation"
        ],
        "scores": scores,
        "average_score": round(sum(scores) / len(scores), 2)
    }