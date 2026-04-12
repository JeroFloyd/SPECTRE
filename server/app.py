from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Path as FPath, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from env.environment import SpectreEnv
from grader.grader import grade_episode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s"
)
logger = logging.getLogger("spectre.app")

app = FastAPI(
    title="S.P.E.C.T.R.E v2",
    description="Self-Programming Environment for Complex Task Reconstruction & Evolution",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Locate data directory
current_file = Path(__file__).resolve()
option_a = current_file.parent / "data"
option_b = current_file.parent.parent / "data"

final_data_path: Path | None = None
if option_a.exists() and option_a.is_dir():
    final_data_path = option_a
elif option_b.exists() and option_b.is_dir():
    final_data_path = option_b

if final_data_path:
    app.mount("/data", StaticFiles(directory=str(final_data_path)), name="data")
    logger.info("MOUNTED DATA AT: %s", final_data_path)
else:
    logger.error("CRITICAL: Could not find 'data' folder.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, SpectreEnv] = {}
MAX_SESSIONS = 100


def _safe(v: float) -> float:
    v = float(v)
    if v <= 0.0: return 0.01
    if v >= 1.0: return 0.99
    return min(0.99, max(0.01, v))


def _to_py(val):
    if val is None: return None
    if isinstance(val, float) and math.isnan(val): return None
    if isinstance(val, np.integer): return int(val)
    if isinstance(val, np.floating): return None if math.isnan(float(val)) else float(val)
    if isinstance(val, np.bool_): return bool(val)
    return val


DATE_FORMATS_LOCAL = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y"]
VALID_STATUSES_LOCAL = {"completed", "pending", "cancelled", "refunded", "processing"}


def _parse_date_local(val: str) -> str | None:
    for fmt in DATE_FORMATS_LOCAL:
        try:
            return datetime.strptime(str(val).strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _is_valid_date(val: str) -> bool:
    return _parse_date_local(str(val)) is not None


def _evict_oldest():
    if len(_sessions) >= MAX_SESSIONS:
        oldest = next(iter(_sessions))
        del _sessions[oldest]


def _get_env(session_id: str) -> SpectreEnv:
    if session_id not in _sessions:
        raise HTTPException(404, f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


class ResetRequest(BaseModel):
    task: str = Field("medium", description="easy | medium | hard | expert")
    seed: Optional[int] = Field(42)
    batch_file: Optional[str] = Field("orders_1.csv", description="Which batch to process")


class ActionRequest(BaseModel):
    type: str = Field(...)
    name: Optional[str] = Field(None)
    sequence: Optional[List[str]] = Field(None)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    base_dir = Path(__file__).resolve().parent
    for candidate in [base_dir / "ui" / "index.html",
                      base_dir.parent / "ui" / "index.html"]:
        if candidate.exists():
            return HTMLResponse(content=candidate.read_text())
    return HTMLResponse(
        content="<h1>UI not found</h1><p>Place index.html in ui/index.html</p>",
        status_code=404,
    )


@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "ok", "version": "2.0.0", "active_sessions": len(_sessions)}


@app.get("/info", tags=["meta"])
def info():
    return {
        "name": "S.P.E.C.T.R.E v2",
        "version": "2.0.0",
        "tasks": ["easy", "medium", "hard", "expert"],
        "primitives": ["parse_data", "validate_data", "transform_data", "aggregate_result", "export_result"],
        "optimal_steps": {"easy": 2, "medium": 3, "hard": 4, "expert": 5},
    }


@app.post("/reset", tags=["openenv"])
def reset(body: ResetRequest = None):
    if body is None:
        body = ResetRequest()
    _evict_oldest()
    
    # CRITICAL FIX: Pass batch_file to environment
    env = SpectreEnv(task=body.task, seed=body.seed or 42, batch_file=body.batch_file or "orders_1.csv")
    session_id = env.session_id
    _sessions[session_id] = env
    obs = env.reset(seed=body.seed or 42)
    obs["session_id"] = session_id
    logger.info("reset session=%s task=%s seed=%s batch=%s", session_id, body.task, body.seed, body.batch_file)
    return {"session_id": session_id, "observation": obs, "reward": 0.01, "done": False, "info": {}}


@app.post("/step", tags=["openenv"])
def step(action: ActionRequest, session_id: str = Query(...)):
    env = _get_env(session_id)
    action_dict = action.dict(exclude_none=True)
    obs, reward, done, info = env.step(action_dict)
    obs["session_id"] = session_id
    return {
        "session_id": session_id,
        "observation": obs,
        "reward": _safe(reward),
        "done": done,
        "info": info,
    }


@app.get("/state", tags=["openenv"])
def state(session_id: str = Query(...)):
    env = _get_env(session_id)
    return {"session_id": session_id, "observation": env.state()}


@app.get("/grade/{session_id}", tags=["evaluation"])
def grade(session_id: str = FPath(...)):
    try:
        env = _get_env(session_id)
        total_reward = _safe(sum(_safe(s.get("reward", 0)) for s in env._step_log))
        report = grade_episode(
            task=env.task_name,
            step_log=env._step_log,
            final_obs=env.state(),
            total_reward=total_reward,
            pipeline_summary=env._pipeline.summary(),
        )
        return report
    except Exception as e:
        logger.error("Grading failed: %s", str(e), exc_info=True)
        return {"success": False, "score": 0.01, "verdict": "Grading failed", "error": str(e)}


@app.get("/grade/{session_id}/full", tags=["evaluation"])
def grade_full(session_id: str = FPath(...)):
    return grade(session_id)


@app.post("/run", tags=["ui"])
async def run_ui_bridge(task: str = "easy", seed: int = 42, batch_file: str = "orders_1.csv", body: Optional[ResetRequest] = None):
    try:
        final_task = body.task if body and body.task else task
        final_seed = body.seed if body and body.seed is not None else seed
        final_batch = body.batch_file if body and body.batch_file else batch_file
        
        _evict_oldest()
        env = SpectreEnv(task=final_task, seed=final_seed, batch_file=final_batch)
        _sessions[env.session_id] = env
        obs = env.reset(seed=final_seed)
        obs["session_id"] = env.session_id
        return {"status": "success", "session_id": env.session_id, "observation": obs,
                "reward": 0.01, "done": False}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/sessions", tags=["meta"])
def list_sessions():
    result = {}
    for sid, env in _sessions.items():
        obs = env.state()
        result[sid] = {
            "task": obs["task"],
            "progress": f"{obs['progress']} / {obs['target_length']}",
            "step_count": obs["step_count"],
            "done": obs["progress"] >= obs["target_length"],
            "tools_defined": obs["custom_tools_defined"],
            "batch": env.batch_file,
        }
    return result


@app.delete("/sessions/{session_id}", tags=["meta"])
def delete_session(session_id: str = FPath(...)):
    if session_id not in _sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    del _sessions[session_id]
    return {"deleted": session_id}


# ── Data preview endpoints ────────────────────────────────────────────────────

@app.get("/api/raw/{filename}", tags=["data"])
def get_raw_data(filename: str):
    """
    Return raw batch file rows with per-row issue detection.
    Issues flagged: missing_quantity, negative_price, bad_status,
                    bad_date, duplicate_id, missing_identity.
    These are exactly what transform() will REPAIR (not drop).
    """
    if final_data_path is None:
        raise HTTPException(500, "Data directory not mounted")
    file_path = final_data_path / "raw" / filename
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    try:
        df = pd.read_csv(str(file_path))

        # Track which order_ids appear more than once
        seen_ids: dict = {}
        dup_set: set = set()
        for _, row in df.iterrows():
            oid = str(row.get("order_id", "")).strip()
            if oid in seen_ids:
                dup_set.add(oid)
            else:
                seen_ids[oid] = True

        rows = []
        dirty_rows = 0

        for _, row in df.iterrows():
            r = {col: _to_py(row[col]) for col in df.columns}
            issues = []

            oid = str(r.get("order_id", "") or "").strip()
            cid = str(r.get("customer_id", "") or "").strip()
            if not oid or oid == "nan":
                issues.append("missing_identity")
            elif oid in dup_set:
                issues.append("duplicate_id")

            if not cid or cid == "nan":
                issues.append("missing_identity")

            qty = r.get("quantity")
            if qty is None:
                issues.append("missing_quantity")
            elif isinstance(qty, (int, float)) and qty < 1:
                issues.append("missing_quantity")

            price = r.get("unit_price")
            if price is None:
                issues.append("missing_price")
            elif isinstance(price, (int, float)) and price < 0:
                issues.append("negative_price")

            status = str(r.get("status", "")).strip().lower()
            if status not in VALID_STATUSES_LOCAL:
                issues.append("bad_status")

            date_val = str(r.get("order_date", ""))
            if not _is_valid_date(date_val):
                issues.append("bad_date")

            if issues:
                dirty_rows += 1

            r["_dirty"] = bool(issues)
            r["_issues"] = issues
            rows.append(r)

        return {"rows": rows, "total_rows": len(rows), "dirty_rows": dirty_rows}

    except Exception as e:
        raise HTTPException(500, f"Failed to load data: {str(e)}")


@app.get("/api/result/{session_id}", tags=["data"])
def get_session_result(session_id: str):
    """
    CRITICAL: Returns the AGENT's actual transformed data from pipeline.all_transformed
    This is the real reconstruction work done by the agent's transform() function.
    """
    try:
        env = _get_env(session_id)
        pipeline = env._pipeline

        if len(pipeline.all_transformed) == 0:
            return {
                "rows": [],
                "total_rows": 0,
                "raw_count": 0,
                "message": "No transformed data yet. Run the episode first."
            }

        import pandas as pd
        combined = pd.concat(pipeline.all_transformed, ignore_index=True)

        rows = [{col: _to_py(row[col]) for col in combined.columns} for _, row in combined.iterrows()]

        rr = pipeline.repair_report.to_dict() if pipeline.repair_report else {}

        repairs = []
        if rr.get("quantities_repaired", 0):
            repairs.append(f"{rr['quantities_repaired']} quantities repaired")
        if rr.get("prices_repaired", 0):
            repairs.append(f"{rr['prices_repaired']} prices repaired")
        if rr.get("statuses_repaired", 0):
            repairs.append(f"{rr['statuses_repaired']} statuses → 'pending'")
        if rr.get("dates_repaired", 0):
            repairs.append(f"{rr['dates_repaired']} dates repaired")
        if rr.get("ids_deduplicated", 0):
            repairs.append(f"{rr['ids_deduplicated']} duplicate IDs renamed")

        return {
            "rows": rows,
            "total_rows": int(len(rows)),
            "raw_count": int(pipeline.total_rows_loaded),
            "clean_count": int(len(rows)),
            "dropped": int(pipeline.total_rows_loaded - len(rows)),
            "repairs": repairs,
            "revenue_total": float(pipeline.revenue_total),
            "task": str(env.task_name),
            "batches": int(pipeline.transform_count),
            "source": "agent_pipeline",
            "batch_file": env.batch_file,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_session_result failed: %s", str(e), exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/api/processed", tags=["data"])
def list_processed_files():
    if final_data_path is None:
        return {"files": []}
    processed_dir = final_data_path / "processed"
    if not processed_dir.exists():
        return {"files": []}
    files = sorted(processed_dir.glob("*.csv"), reverse=True)
    return {"files": [f.name for f in files[:10]]}


@app.get("/api/processed/{filename}", tags=["data"])
def get_processed_data(filename: str):
    if final_data_path is None:
        raise HTTPException(500, "Data directory not mounted")
    file_path = final_data_path / "processed" / filename
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    try:
        df = pd.read_csv(str(file_path))
        rows = [{col: _to_py(row[col]) for col in df.columns} for _, row in df.iterrows()]
        return {"rows": rows, "total_rows": len(rows), "filename": filename}
    except Exception as e:
        raise HTTPException(500, f"Failed to load: {str(e)}")
    
    
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
