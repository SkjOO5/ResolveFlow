"""
server/app.py

OpenEnv server entry point — required by openenv validate for multi-mode deployment.

The main() function is the [project.scripts] entry point:
    serve = "server.app:main"

It is also called directly via:
    python server/app.py
    CMD ["python", "server/app.py"]        ← Dockerfile
"""

import os
import sys
import uvicorn

# Ensure project root is always importable regardless of working directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def create_app():
    """Factory function that creates and returns the FastAPI application."""
    from app import app  # root app.py FastAPI instance
    return app


def main():
    """
    Main entry point for the OpenEnv ResolveFlow server.

    This function is required by openenv validate — it must:
      - Exist in server/app.py
      - Be callable
      - Be reachable via the [project.scripts] serve entry point

    Called by:
      - openenv validate (existence + callable check)
      - [project.scripts]:  serve = "server.app:main"
      - Direct execution:   python server/app.py
      - Docker CMD:         CMD ["python", "server/app.py"]
    """
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    log_level = os.environ.get("LOG_LEVEL", "info")
    reload = os.environ.get("RELOAD", "false").lower() == "true"

    print(f"[ResolveFlow] Starting OpenSupportEnv on {host}:{port}")
    print(f"[ResolveFlow] Log level: {log_level}")

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


if __name__ == "__main__":
    main()
