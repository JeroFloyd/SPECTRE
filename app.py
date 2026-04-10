from __future__ import annotations

import logging
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.environment import SpectreEnv
from grader.grader   import grade_episode

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s")
logger = logging.getLogger("spectre.app")

app = FastAPI(
    title       = "S.P.E.C.T.R.E",
    description = (
        "Self-Programming Environment for Complex Task Reconstruction & Evolution. "
        "OpenEnv-compliant RL environment where AI agents complete real-world "
        "data processing pipelines and learn to self-program reusable macros."
    ),
    version  = "2.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_sessions: dict[str, SpectreEnv] = {}
MAX_SESSIONS = 100


def _safe(v: float) -> float:
    """Clamp any float to strictly (0.01, 0.99) — never 0.0 or 1.0."""
    v = float(v)
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.99
    return min(0.99, max(0.01, v))


def _evict_oldest():
    if len(_sessions) >= MAX_SESSIONS:
        oldest = next(iter(_sessions))
        del _sessions[oldest]
        logger.info("evicted session %s", oldest)

def _get_env(session_id: str) -> SpectreEnv:
    if session_id not in _sessions:
        raise HTTPException(404, f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]

class ResetRequest(BaseModel):
    task: str = Field("medium", description="Task difficulty: easy | medium | hard")
    seed: Optional[int] = Field(42, description="RNG seed for reproducible episodes")

class ActionRequest(BaseModel):
    type:     str                 = Field(...,  description="primitive | create_tool | use_tool")
    name:     Optional[str]       = Field(None, description="Primitive or tool name")
    sequence: Optional[List[str]] = Field(None, description="Op list for create_tool")

    model_config = {"json_schema_extra": {"examples": [
        {"type": "primitive",   "name": "parse_data"},
        {"type": "create_tool", "name": "etl_batch",
         "sequence": ["parse_data", "validate_data", "transform_data"]},
        {"type": "use_tool", "name": "etl_batch"},
    ]}}

@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "ok", "version": "2.0.0", "active_sessions": len(_sessions)}

@app.get("/", tags=["meta"])
def root():
    return {
        "name":    "S.P.E.C.T.R.E",
        "version": "2.0.0",
        "tasks":   ["easy", "medium", "hard"],
        "endpoints": {
            "POST /reset":              "Start new episode -> returns session_id",
            "POST /step?session_id=X":  "Execute an action",
            "GET  /state?session_id=X": "Read current observation",
            "GET  /grade/{session_id}": "Grade a completed episode",
            "GET  /sessions":           "List active sessions",
            "GET  /docs":               "Interactive API docs",
        },
    }

@app.post("/reset", tags=["openenv"])
def reset(body: ResetRequest = None):
    if body is None:
        body = ResetRequest()

    _evict_oldest()
    env        = SpectreEnv(task=body.task, seed=body.seed or 42)
    session_id = env.session_id
    _sessions[session_id] = env
    obs = env.reset(seed=body.seed or 42)
    obs["session_id"] = session_id

    logger.info("reset session=%s task=%s seed=%s", session_id, body.task, body.seed)
    return {
        "session_id":  session_id,
        "observation": obs,
        "reward":      0.01,   # ✅ never 0.0
        "done":        False,
        "info":        {},
    }

@app.post("/step", tags=["openenv"])
def step(
    action:     ActionRequest,
    session_id: str = Query(..., description="Session ID from /reset"),
):
    env         = _get_env(session_id)
    action_dict = action.model_dump(exclude_none=True)
    obs, reward, done, info = env.step(action_dict)

    return {
        "session_id":  session_id,
        "observation": obs,
        "reward":      _safe(reward),   # ✅ clamped, never 0.0 or 1.0
        "done":        done,
        "info":        info,
    }

@app.get("/state", tags=["openenv"])
def state(session_id: str = Query(..., description="Session ID from /reset")):
    env = _get_env(session_id)
    return {"session_id": session_id, "observation": env.state()}

@app.get("/grade/{session_id}", tags=["evaluation"])
def grade(session_id: str = Path(...)):
    env    = _get_env(session_id)
    obs    = env.state()
    report = grade_episode(
        task             = env.task_name,
        step_log         = env._step_log,
        final_obs        = obs,
        total_reward     = _safe(sum(_safe(s["reward"]) for s in env._step_log)),
        pipeline_summary = env._pipeline.summary(),
    )
    # ✅ Return ONLY the score — no integers or out-of-range floats
    return {"score": report["score"]}

@app.get("/sessions", tags=["meta"])
def list_sessions():
    result = {}
    for sid, env in _sessions.items():
        obs = env.state()
        result[sid] = {
            "task":          obs["task"],
            "progress":      f"{obs['progress']} / {obs['target_length']}",
            "step_count":    obs["step_count"],
            "done":          obs["progress"] >= obs["target_length"],
            "tools_defined": obs["custom_tools_defined"],
        }
    return result

@app.delete("/sessions/{session_id}", tags=["meta"])
def delete_session(session_id: str = Path(...)):
    if session_id not in _sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    del _sessions[session_id]
    return {"deleted": session_id}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
