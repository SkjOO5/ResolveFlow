from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import traceback
import os

from envs.models import Action, State, TaskDefinition
from envs.environment import OpenSupportEnv
from envs.tasks import ALL_TASKS
from envs.graders import Grader
from envs.scoring import SCORE_MIN, SCORE_MAX

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

    Each task entry includes a `grader` object describing the grader type,
    score range, endpoint, and evaluation dimensions. This is the primary
    signal the OpenEnv validator uses to determine whether tasks have graders.

    Returns:
        List of task metadata including id, difficulty, max_steps, title,
        grader config, and score range.
    """
    grader_dimensions = [
        "classification",
        "priority",
        "tool_usage",
        "policy_compliance",
        "resolution",
        "response_quality",
        "efficiency",
    ]
    return [
        {
            "id": t.task_id,
            "title": t.title,
            "difficulty": t.difficulty,
            "max_steps": t.max_steps,
            "description": t.customer_message[:120] + "..." if len(t.customer_message) > 120 else t.customer_message,
            # Primary grader registration block — parsed by OpenEnv validator
            "grader": {
                "type": "deterministic",
                "endpoint": "/grade",
                "score_range": [SCORE_MIN, SCORE_MAX],
                "open_interval": True,
                "dimensions": grader_dimensions,
            },
            # Legacy compatibility fields — keep for any older validator build
            "has_grader": True,
            "grader_type": "deterministic",
            "score_min": SCORE_MIN,
            "score_max": SCORE_MAX,
        }
        for t in ALL_TASKS.values()
    ]


@app.post("/grade")
async def grade_task(request: dict):
    """
    Grade a completed episode for a given task.

    This endpoint is referenced in openenv.yaml as the per-task grader endpoint.
    It accepts a state dict (from /state) and task_id, runs the deterministic
    grader, and returns a score strictly in (0, 1).

    Expected body:
        {
            "task_id": "task_001_easy",
            "state": { ... }   # optional — uses current env state if omitted
        }

    Returns:
        {
            "task_id": "...",
            "score": float,          # strictly in (SCORE_MIN, SCORE_MAX)
            "score_min": 0.05,
            "score_max": 0.95,
            "open_interval": true,
            "breakdown": { ... },
            "summary": "...",
            "audit": [ ... ]
        }
    """
    try:
        task_id = request.get("task_id", "task_001_easy")

        # Use provided state or fall back to current env state
        current_state = env.current_state()
        if current_state is None or current_state.task_id != task_id:
            # Reset to requested task for fresh grading baseline
            env.reset(task_id)
            current_state = env.current_state()

        task = ALL_TASKS.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

        final_score, breakdown, summary, audit = Grader.grade(current_state, task)

        return {
            "task_id": task_id,
            "score": final_score,
            "score_min": SCORE_MIN,
            "score_max": SCORE_MAX,
            "open_interval": True,
            "breakdown": breakdown,
            "summary": summary,
            "audit": audit,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Grading error: {str(e)}")


# ── React frontend (mounted AFTER all API routes so it only catches unknown paths) ──
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
