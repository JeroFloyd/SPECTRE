from __future__ import annotations

import argparse
import json
from pathlib import Path

OPTIMAL_STEPS = {"easy": 3, "medium": 3, "hard": 5, "expert": 7}
PASSING_QUALITY = 0.70
PASSING_EFFICIENCY = 0.50


def _safe(v: float) -> float:
    v = float(v)
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.99
    return min(0.99, max(0.01, v))


def grade_episode(task, step_log, final_obs, total_reward, pipeline_summary):
    steps_taken = final_obs["step_count"]
    progress = final_obs["progress"]
    target_length = final_obs["target_length"]
    optimal = OPTIMAL_STEPS.get(task, target_length)

    efficiency = optimal / max(steps_taken, 1)
    compression = final_obs.get("compression_ratio", 0.0)
    quality_score = pipeline_summary.get("quality_score", 0.0)
    has_aggregate = bool(pipeline_summary.get("aggregate", {}).get("total_batches", 0))

    has_export = any(s["action"].get("name") == "export_result" for s in step_log)
    success = progress >= target_length and (not has_export or quality_score >= PASSING_QUALITY)

    output_path = pipeline_summary.get("output_path", "")
    output_verified = False
    if output_path:
        p = Path(output_path)
        output_verified = (
            p.exists() and p.stat().st_size > 0 and
            quality_score >= PASSING_QUALITY and
            pipeline_summary.get("rows_exported", 0) > 0
        )

    if not success:
        verdict = "FAIL — incomplete pipeline"
    elif efficiency < PASSING_EFFICIENCY:
        verdict = "PASS (slow) — completed but inefficient"
    elif compression > 1.0 and has_aggregate:
        verdict = "PASS (expert) — hierarchical tools + cross-batch aggregation"
    elif compression > 1.0:
        verdict = "PASS (self-programmed) — efficient tool composition"
    else:
        verdict = "PASS — completed without self-programming"

    tool_creates = [s for s in step_log if s["action"].get("type") == "create_tool"]
    tool_uses = [s for s in step_log if s["action"].get("type") == "use_tool"]

    caps = {"easy": 0.84, "medium": 0.92, "hard": 0.96, "expert": 0.98}
    
    raw_score = min(caps.get(task, 0.95), total_reward)
    score = max(0.01, min(0.99, raw_score))
    if score <= 0.0:
        score = 0.01
    if score >= 1.0:
        score = 0.99

    return {
        "session_id": str(final_obs.get("session_id", "")),
        "task": str(task),
        "success": bool(success),
        "score": float(score),  # Now guaranteed to be in (0, 1)
        "efficiency_ratio": float(max(0.01, min(0.99, efficiency))),
        "compression_ratio": float(max(0.01, min(0.99, compression))),
        "quality_score": float(max(0.01, min(0.99, quality_score))),
        "output_verified": bool(output_verified),
        "output_hash": str(pipeline_summary.get("output_hash", "")),
        "verdict": str(verdict),
        "has_aggregate": bool(has_aggregate),
        "optimal_steps": int(optimal),
        "rows_exported": int(pipeline_summary.get("rows_exported", 0)),
        "revenue_total": float(pipeline_summary.get("revenue_total", 0.0)),
    }
