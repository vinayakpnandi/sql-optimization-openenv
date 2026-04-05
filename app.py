"""
FastAPI application exposing the SQLOptimizationEnv via HTTP,
compatible with the OpenEnv spec used by the competition judge.
"""

import json
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from env.environment import SQLOptimizationEnv, SQLAction

app = FastAPI(
    title="SQL Optimization OpenEnv",
    description="An OpenEnv environment for SQL query optimization tasks.",
    version="1.0.0",
)

# Simple in-memory session store (fine for single-container HF Space)
_sessions: dict[str, SQLOptimizationEnv] = {}


def _get_or_create(session_id: str, task_id: Optional[str] = None) -> SQLOptimizationEnv:
    if session_id not in _sessions:
        _sessions[session_id] = SQLOptimizationEnv(task_id=task_id)
    return _sessions[session_id]


# ── Health / ping ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "env": "sql-optimization-openenv"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── OpenEnv endpoints ────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    task_id: Optional[str] = None   # "task_easy" | "task_medium" | "task_hard"


@app.post("/reset")
def reset(req: ResetRequest):
    sid = req.session_id or str(uuid.uuid4())
    env = _get_or_create(sid, task_id=req.task_id)
    obs = env.reset(task_id=req.task_id)
    return {"session_id": sid, "observation": obs.model_dump()}


class StepRequest(BaseModel):
    session_id: str
    action: dict  # {"query": "...", "message": "..."}


@app.post("/step")
def step(req: StepRequest):
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    env = _sessions[req.session_id]
    try:
        action = SQLAction(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    obs, reward, done, info = env.step(action)
    return {
        "session_id": req.session_id,
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return _sessions[session_id].state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    from env.environment import TASKS
    return {"tasks": [{"id": t["id"], "name": t["name"], "difficulty": t["difficulty"]} for t in TASKS]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
