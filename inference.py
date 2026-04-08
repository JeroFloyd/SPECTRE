"""
S.P.E.C.T.R.E Inference Script
===================================
OpenEnv-compliant stdout format:
    [START] task=<task> env=spectre model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from env.environment      import SpectreEnv
from agent.baseline_agent import BaselineAgent

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
TASKS        = ["easy", "medium", "hard"]
SEED         = int(os.getenv("SPECTRE_SEED", "42"))

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

SYSTEM_PROMPT = """\
You are an autonomous agent in S.P.E.C.T.R.E — Self-Programming Environment for \
Complex Task Reconstruction & Evolution.

Your goal: complete a real-world data processing pipeline using as FEW steps as possible.

== PRIMITIVES ==
- parse_data     : load a raw CSV batch
- validate_data  : check nulls, types, duplicates, dates, enum values
- transform_data : clean, normalise, compute revenue = quantity × unit_price
- export_result  : write processed output and compute quality score

== ACTION TYPES (respond with valid JSON only) ==
1. {"type": "primitive",   "name": "<name>"}
2. {"type": "create_tool", "name": "<name>", "sequence": ["op1", "op2", ...]}
3. {"type": "use_tool",    "name": "<tool_name>"}

== STRATEGY ==
- Look at `next_required_op` — that is exactly what you must do next.
- If the remaining sequence has a REPEATING PATTERN, build a tool then invoke it.
- Example for medium (6-step sequence = 2× [parse→validate→transform]):
    Step 1: create_tool "etl_batch" = [parse_data, validate_data, transform_data]
    Step 2: use_tool "etl_batch"   (covers ops 1-3)
    Step 3: use_tool "etl_batch"   (covers ops 4-6) — done in 3 steps!
- Tools can compose other tools for even greater compression.
- Fewer total steps = higher reward (efficiency + compression bonus).
- IMPORTANT: `next_required_op` tells you the exact primitive needed now.
  A tool must expand to match the EXACT next ops or it will be rejected.

Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.\
"""


def build_prompt(obs: dict) -> str:
    ps = obs.get("pipeline_state", {})
    vr = ps.get("validation", {})
    return f"""\
== CURRENT STATE ==
Task       : {obs['task']} — {obs['task_description']}
Progress   : {obs['progress']} / {obs['target_length']} primitives completed
Next op    : {obs['next_required_op']}
Remaining  : {obs['remaining_steps']} ops still needed
Steps used : {obs['step_count']} / {obs['max_steps']}
Compress   : {obs['compression_ratio']} (higher = more leverage)

== TOOLBOX ==
Primitives : {obs['available_primitives']}
Tools built: {obs['custom_tools_defined']}
Registry   : {json.dumps(obs['tool_registry'], separators=(',', ':'))}

== PIPELINE STATE ==
Source     : {ps.get('source_file', 'none')} ({ps.get('rows_loaded', 0)} rows)
Validation : quality={vr.get('quality_score', 0):.2f}  flagged={vr.get('rows_flagged', 0)}
Transformed: {ps.get('rows_after_transform', 0)} rows  revenue=${ps.get('revenue_total', 0):.2f}
Exported   : {ps.get('rows_exported', 0)} rows

Choose your next action as a single JSON object.\
"""


def get_llm_action(obs: dict) -> dict | None:
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs)},
            ],
            temperature = 0,
            max_tokens  = 200,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return None


def _clamp(r: float) -> float:
    """Ensure reward is strictly between 0 and 1 (never 0.0 or 1.0)."""
    return round(min(0.99, max(0.01, r)), 4)


def log_start(task: str, model: str):
    print(f"[START] task={task} env=spectre model={model}", flush=True)

def log_step(step: int, action: dict, reward: float, done: bool, error: str | None):
    print(
        f"[STEP] step={step} action={json.dumps(action)} "
        f"reward={_clamp(reward):.2f} done={str(done).lower()} "
        f"error={error or 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: list[float]):
    rewards_str = ",".join(f"{_clamp(r):.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
          flush=True)


def run_task(task_name: str):
    env      = SpectreEnv(task=task_name, seed=SEED)
    fallback = BaselineAgent()

    obs      = env.reset(seed=SEED)
    done     = False
    rewards: list[float] = []
    steps    = 0
    success  = False

    log_start(task=task_name, model=MODEL_NAME)

    try:
        while not done and steps < env.max_steps:
            steps += 1
            try:
                action = get_llm_action(obs) if client else None
                if action is None:
                    action = fallback.act(obs)

                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                log_step(steps, action, reward, done, info.get("error"))

            except Exception as exc:
                log_step(steps, {"error": str(exc)}, 0.01, True, str(exc))
                done = True

        success = obs["progress"] >= obs["target_length"]

    finally:
        log_end(success=success, steps=steps, rewards=rewards)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
        print()
