from fastapi import FastAPI
from env.environment import LexiGuardEnv

app = FastAPI()

env = LexiGuardEnv()
obs = env.reset()


@app.get("/state")
def get_state():
    return env.state()


@app.get("/")
def home():
    return {"message": "LexiGuard OpenEnv running"}