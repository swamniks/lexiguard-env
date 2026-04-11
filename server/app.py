from __future__ import annotations
import sys
import os

# Add /app/env to path so 'models' can be found
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from openenv.core.env_server.http_server import create_app
from models import LexiGuardAction, LexiGuardObservation
from server.lexiguard_environment import LexiGuardEnvironment

app = create_app(
    LexiGuardEnvironment,
    LexiGuardAction,
    LexiGuardObservation,
    env_name="lexiguard",
    max_concurrent_envs=10,
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()