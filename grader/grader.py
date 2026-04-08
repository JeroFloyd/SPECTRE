from __future__ import annotations

import argparse
import json
from pathlib import Path

OPTIMAL_STEPS = {
    "easy":   4,
    "medium": 3,
    "hard":   4,
}

PASSING_QUALITY    = 0.70
PASSING_EFFICIENCY = 0.50


def grade_episode(
    task:             str,
    step_log:         list[dict],
    final_obs:        dict,
    total_reward:     float,
    pipeline_summary: dict,
) -> dict:
    steps_taken   = final_obs["step_count"]
    progress      = final_obs["progress"]
    target_length = final_obs["target_length"]
    optimal       = OPTIMAL_STEPS.get(task, target_length)

    success = progress >= target_length

    efficiency = optimal / max(steps_taken, 1)
    efficiency = max(1e-6, min(0.9999, efficiency))
    efficiency = round(efficiency, 4)
    compression   = final_obs.get("compression_ratio", 0.0)
    quality_score = pipeline_summary.get("quality_score", 0.0)

    EPS = 1e-6
    quality_score = max(EPS, min(1.0 - EPS, quality_score))

    success = (
        progress >= target_length and
        quality_score >= PASSING_QUALITY
    )
    output_hash   = pipeline_summary.get("output_hash", "")
    output_path   = pipeline_summary.get("output_path", "")

    output_verified = False
    if output_path:
        p = Path(output_path)
        output_verified = (
            p.exists()
            and p.stat().st_size > 0
            and quality_score >= PASSING_QUALITY
            and pipeline_summary.get("rows_exported", 0) > 0
        )

    if not success:
        verdict = "FAIL — incomplete or low-quality output"
    elif efficiency < PASSING_EFFICIENCY:
        verdict = "PASS (slow) — completed but inefficient"
    elif compression > 1.0:
        verdict = "PASS (self-programmed) — efficient tool composition"
    else:
        verdict = "PASS — completed without self-programming"

    tool_creates = [s for s in step_log if s["action"].get("type") == "create_tool"]
    tool_uses    = [s for s in step_log if s["action"].get("type") == "use_tool"]
    errors       = [s for s in step_log if s.get("info", {}).get("error")]

    # Unified reward scaling — strictly within (0, 1), never 0.0 or 1.0
    if task == "hard":
        cap = 0.99
    elif task == "medium":
        cap = 0.95
    else:
        cap = 0.85

    # Clamp strictly between 0.01 and cap (never exactly 0.0 or 1.0)
    capped_reward = round(min(cap, max(0.01, total_reward)), 4)

    return {
        "session_id":        final_obs.get("session_id", ""),
        "task":              task,
        "success":           success,
        "steps_taken":       steps_taken,
        "optimal_steps":     optimal,
        "efficiency_ratio":  efficiency,
        "compression_ratio": compression,
        "quality_score":     round(quality_score, 4),
        "total_reward":      capped_reward,
        "output_verified":   output_verified,
        "output_hash":       output_hash,
        "verdict":           verdict,
        "breakdown": {
            "progress":           progress,
            "target_length":      target_length,
            "tool_creates":       len(tool_creates),
            "tool_uses":          len(tool_uses),
            "tools_defined":      final_obs.get("custom_tools_defined", []),
            "tool_registry":      final_obs.get("tool_registry", {}),
            "errors_encountered": len(errors),
            "rows_exported":      pipeline_summary.get("rows_exported", 0),
            "revenue_total":      pipeline_summary.get("revenue_total", 0.0),
        },
    }


def run_and_grade(
    task:    str  = "hard",
    seed:    int  = 42,
    agent          = None,
    verbose: bool = True,
) -> dict:
    from env.environment      import SpectreEnv
    from agent.baseline_agent import BaselineAgent

    if agent is None:
        agent = BaselineAgent()

    env  = SpectreEnv(task=task, seed=seed)
    obs  = env.reset(seed=seed)
    done = False
    total_reward = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  S.P.E.C.T.R.E Grader — task={task}  seed={seed}")
        print(f"{'='*60}")
        print(f"  {obs['task_description']}\n")

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if verbose:
            err = f"  ERROR: {info['error']}" if info.get("error") else ""
            print(f"  Step {obs['step_count']:>2} │ {str(action):<70} │ r={reward:+.3f}{err}")

    if verbose:
        ps = env._pipeline.summary()
        print(f"\n  Progress : {obs['progress']} / {obs['target_length']}")
        print(f"  Steps    : {obs['step_count']}  (optimal: {OPTIMAL_STEPS.get(task, '?')})")
        print(f"  Compress : {obs['compression_ratio']:.3f}")
        print(f"  Revenue  : ${ps['revenue_total']:,.2f}")
        print(f"  Quality  : {ps['quality_score']:.3f}")
        print(f"  Exported : {ps['rows_exported']} rows → {ps['output_path']}")

    report = grade_episode(
        task             = task,
        step_log         = env._step_log,
        final_obs        = obs,
        total_reward     = total_reward,
        pipeline_summary = env._pipeline.summary(),
    )
    if verbose:
        print(f"  Reward   : {report['total_reward']:.4f}")

    if verbose:
        print(f"\n  Verdict  : {report['verdict']}")
        print(f"{'='*60}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S.P.E.C.T.R.E Grader")
    parser.add_argument("--task",  default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed",  default=42, type=int)
    parser.add_argument("--json",  action="store_true")
    parser.add_argument("--all",   action="store_true")
    args = parser.parse_args()

    tasks   = ["easy", "medium", "hard"] if args.all else [args.task]
    reports = [run_and_grade(task=t, seed=args.seed, verbose=not args.json) for t in tasks]

    print(json.dumps(reports if args.all else reports[0], indent=2))
