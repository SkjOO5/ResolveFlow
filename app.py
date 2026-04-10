from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import traceback
import os

from envs.models import Action
from envs.environment import OpenSupportEnv
from envs.tasks import ALL_TASKS

# redirect_slashes=False prevents POST /reset/ → redirect to GET /reset
app = FastAPI(
    title="ResolveFlow API",
    description="OpenSupportEnv — OpenEnv hackathon benchmark",
    redirect_slashes=False
)

# CORS — allow all origins and methods so the validator can POST freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = OpenSupportEnv()


# ── Request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_001_easy"


# ── API Routes (must be registered BEFORE the static mount) ──────────────────

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/reset")
async def reset_env(request: Optional[ResetRequest] = None):
    """Reset the environment. Accepts optional JSON body: {task_id: str}"""
    try:
        task_id = request.task_id if request and request.task_id else "task_001_easy"
        obs = env.reset(task_id)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step_env(action: Action):
    """Execute one action step. Accepts JSON body matching Action schema."""
    try:
        result = env.step(action)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/state")
async def get_state():
    """Return current full environment state."""
    return env.current_state().model_dump()


@app.get("/tasks")
async def list_tasks():
    """
    List all available tasks with grader information.
    
    This endpoint is required by OpenEnv validator to discover available tasks
    and verify that each task has a grader attached.
    
    Returns:
        List of task metadata including id, difficulty, max_steps, title,
        and grader availability.
    """
    return [
        {
            "id": t.task_id,
            "title": t.title,
            "difficulty": t.difficulty,
            "max_steps": t.max_steps,
            "has_grader": True,  # All tasks have graders
            "grader_type": "deterministic",
            "rubric_dimensions": [
                "classification",
                "priority",
                "tool_usage",
                "policy_compliance",
                "resolution",
                "response_quality",
                "efficiency"
            ]
        }
        for t in ALL_TASKS.values()  # FIXED: was iterating over keys, now values
    ]


# ── React frontend (mounted AFTER all API routes so it only catches unknown paths) ──
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
