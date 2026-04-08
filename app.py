from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import traceback
import os

from envs.models import Action
from envs.environment import OpenSupportEnv
from envs.tasks import ALL_TASKS

app = FastAPI(title="ResolveFlow API", description="OpenSupportEnv backend")
env = OpenSupportEnv()


# ── Request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_001_easy"


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/reset")
def reset_env(request: ResetRequest = None):
    """Reset the environment for a given task. Accepts JSON body: {task_id: str}"""
    try:
        task_id = request.task_id if request and request.task_id else "task_001_easy"
        obs = env.reset(task_id)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/step")
def step_env(action: Action):
    """Execute one action step. Accepts JSON body matching Action schema."""
    try:
        result = env.step(action)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/state")
def get_state():
    """Return current full environment state."""
    return env.current_state().model_dump()


# ── Frontend static serving (must come AFTER all API routes) ──────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
