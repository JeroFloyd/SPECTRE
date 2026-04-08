from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env.pipeline import PipelineState


def compute_reward(
    step_count:    int,
    max_steps:     int,
    done:          bool,
    progress:      int,
    target_length: int,
    prev_progress: int,
    pipeline:      "PipelineState",
    custom_tools:  dict,
) -> tuple[float, dict]:

    step_penalty      = -0.01
    progress_bonus    = 0.04 * max(0, progress - prev_progress)
    completion_bonus  = 0.0
    quality_bonus     = 0.0
    compression_bonus = 0.0

    if done and progress >= target_length:
        efficiency       = max(0.0, 1.0 - (step_count / max_steps))
        completion_bonus = round(0.6 * efficiency, 4)

        if pipeline.quality_score >= 0.90:
            quality_bonus = 0.15
        elif pipeline.quality_score >= 0.75:
            quality_bonus = 0.07

        if custom_tools and step_count < progress:
            compression_bonus = 0.10

    raw    = step_penalty + progress_bonus + completion_bonus + quality_bonus + compression_bonus
    reward = round(max(0.0, min(1.0, raw)), 6)

    breakdown = {
        "step_penalty":      step_penalty,
        "progress_bonus":    round(progress_bonus, 4),
        "completion_bonus":  round(completion_bonus, 4),
        "quality_bonus":     round(quality_bonus, 4),
        "compression_bonus": round(compression_bonus, 4),
        "raw":               round(raw, 6),
        "total":             reward,
    }
    return reward, breakdown