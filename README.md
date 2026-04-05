---
title: LexiGuard OpenEnv
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "0.0.1"
app_file: inference.py
pinned: false
---

# ⚖️ LexiGuard OpenEnv

Legal-contract review environment with deterministic graders and reward shaping. Provides three tasks (easy/medium/hard) that exercise clause spotting, risk assessment, and negotiation.

## Tasks
- **Clause Identification (easy)** – identify a termination clause from text. Reward gives partial credit for notice-related wording.
- **Risk Classification (medium)** – label indemnity clause risk as low/medium/high. Reward boosts recognition of uncapped exposure.
- **Contract Negotiation (hard)** – propose redlines to reduce supplier liability. Reward increments for adding a monetary cap, excluding consequential damages, and adding mutuality; penalizes widening liability.

## Environment API
Implemented in `env/environment.py`:
- `reset() -> Observation` resets episode and returns first task.
- `step(action: Action) -> (Observation|None, Reward, done, info)` grades the action and advances.
- `state() -> dict` reports progress and cumulative score.

Data models live in `env/models.py` using Pydantic (`Observation`, `Action`, `Reward`). Tasks are defined in `env/tasks.py`. Graders with shaping are in `env/grader.py`.

## Running locally
```bash
pip install -r requirements.txt
python inference.py
```
By default this uses a heuristic fallback so it runs without keys. To use OpenAI:
```bash
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini   # or your model
export API_BASE_URL=https://api.openai.com/v1
python inference.py
```

## Docker
```bash
docker build -t lexiguard .
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY lexiguard
```

## openenv.yaml
`openenv.yaml` exposes the environment entrypoint `env.environment:make` with reward range [0,1] and shaping enabled for orchestration tools.

## Notes
- Deterministic graders ensure reproducible scores between 0.0 and 1.0.
- Reward shaping provides partial credit rather than binary pass/fail.
- `inference.py` demonstrates an end-to-end episode with OpenAI client plus an offline heuristic policy.
# ⚖️ LexiGuard OpenEnv

LexiGuard is a real-world OpenEnv environment for **legal contract review and risk analysis**, simulating tasks performed by legal analysts.

It provides deterministic graders and reward shaping across three progressively difficult tasks.

---

## 🧠 Motivation

Legal contract review is critical but time-intensive and error-prone.  
LexiGuard models this workflow to evaluate AI agents on:
- clause understanding
- risk reasoning
- contract improvement

---

## 🎯 Tasks

### 🟢 Clause Identification (Easy)
Identify a termination clause from contract text.  
Reward provides partial credit for recognizing notice-based language.

---

### 🟡 Risk Classification (Medium)
Classify indemnity clause risk (low / medium / high).  
Reward emphasizes recognition of **uncapped liability exposure**.

---

### 🔴 Contract Negotiation (Hard)
Propose redlines to reduce supplier liability.  
Reward considers:
- introducing a liability cap  
- excluding consequential damages  
- adding mutuality  
- sufficient detail in revision  

Penalties applied for expanding liability or unsafe clauses.

---

## ⚙️ Environment API

Implemented in `env/environment.py`:

- `reset() -> Observation`  
- `step(action: Action) -> (Observation|None, Reward, done, info)`  
- `state() -> dict`

Data models (`Observation`, `Action`, `Reward`) are defined in `env/models.py` using Pydantic.

---

## 🎁 Reward Design

- Scores are normalized between **0.0 and 1.0**
- Multi-criteria grading ensures **partial rewards**
- Deterministic logic ensures reproducibility
- Penalties discourage unsafe or irrelevant outputs

---

## 📊 Baseline Results

| Task | Score |
|------|------|
| Clause Identification | ~0.83 |
| Risk Classification | ~0.86 |
| Contract Negotiation | ~0.56 |

These results reflect increasing task difficulty and realistic performance variation.

---

## 🧪 Running Locally

```bash
pip install -r requirements.txt
python inference.py