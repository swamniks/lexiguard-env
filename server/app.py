from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    raise ImportError("openenv-core is required. Install with: pip install openenv-core")

try:
    from ..models import LexiGuardAction, LexiGuardObservation
    from .lexiguard_environment import LexiGuardEnvironment
except (ImportError, ModuleNotFoundError):
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