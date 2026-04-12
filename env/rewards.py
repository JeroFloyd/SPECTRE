from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.pipeline import PipelineState

def _safe(v: float) -> float:
    """Ensure value is STRICTLY between 0 and 1 (exclusive)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.5  # Safe default
    
    if v <= 0.0 or not (0 < v < float('inf')):
        return 0.01
    if v >= 1.0:
        return 0.99
    
    clamped = max(0.01, min(0.99, v))
    
    if clamped <= 0.0:
        return 0.01
    if clamped >= 1.0:
        return 0.99
    
    return clamped

def compute_reward(step_count, max_steps, done, progress, target_length, prev_progress, pipeline, custom_tools):
    step_penalty = -0.01
    progress_bonus = 0.05 * max(0, progress - prev_progress)
    completion_bonus = 0.0
    quality_bonus = 0.0
    compression_bonus = 0.0
    aggregate_bonus = 0.0
    
    if done and progress >= target_length:
        efficiency = max(0.0, 1.0 - (step_count / max_steps))
        completion_bonus = round(efficiency, 4)
        qs = pipeline.quality_score
        if qs >= 0.90:
            quality_bonus = 0.20
        elif qs >= 0.75:
            quality_bonus = 0.10
        if custom_tools and step_count < progress:
            compression_bonus = 0.15
        if pipeline.aggregate_report is not None:
            aggregate_bonus = 0.05
    
    raw = step_penalty + progress_bonus + completion_bonus + quality_bonus + compression_bonus + aggregate_bonus
    reward = _safe(raw)
    reward = round(reward, 6)
    breakdown = {
        "step_penalty": step_penalty,
        "progress_bonus": round(progress_bonus, 4),
        "completion_bonus": round(completion_bonus, 4),
        "quality_bonus": round(quality_bonus, 4),
        "compression_bonus": round(compression_bonus, 4),
        "aggregate_bonus": round(aggregate_bonus, 4),
        "total": reward,
    }
    return reward, breakdown
