from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import traceback

from envs.models import Action
from envs.environment import OpenSupportEnv
from envs.tasks import ALL_TASKS

app = FastAPI(title="ResolveFlow API", description="OpenSupportEnv backend")
env = OpenSupportEnv()

# Serve UI
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "tasks": ALL_TASKS})

# API Routes
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/reset")
def reset_env(task_id: str = None):
    try:
        obs = env.reset(task_id)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/step")
def step_env(action: Action):
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
    try:
        state = env.state()
        return state.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8888))
    uvicorn.run(app, host="0.0.0.0", port=port)
