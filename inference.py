from __future__ import annotations

import json
import os

from openai import OpenAI

from env.environment import SpectreEnv
from agent.baseline_agent import BaselineAgent

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
TASKS = ["easy", "medium", "hard", "expert"]
SEED = int(os.getenv("SPECTRE_SEED", "42"))

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

SYSTEM_PROMPT = """\
You are an autonomous agent in S.P.E.C.T.R.E. Complete the data pipeline in as few steps as possible.

PRIMITIVES: parse_data, validate_data, transform_data, aggregate_result, export_result

ACTION TYPES:
1. {"type": "primitive", "name": "<n>"}
2. {"type": "create_tool", "name": "<n>", "sequence": ["op1", "op2", ...]}
3. {"type": "use_tool", "name": "<tool_name>"}

STRATEGY: If you see a repeating pattern in remaining_steps, build a tool for it then invoke it.
Tools can reference other tools for hierarchical compression.

Expert example (14 ops -> 5 steps):
  create_tool quad_etl = [etl_batch x4]  ->  use quad_etl  ->  aggregate_result  ->  export_result

Respond with ONLY a valid JSON object.\
"""


def get_llm_action(obs):
    if client is None:
        return None
    try:
        prompt = (
            f"Task: {obs['task']} | Next: {obs['next_required_op']} | "
            f"Remaining: {obs['remaining_steps']} | Tools: {obs['custom_tools_defined']}\n"
            f"Registry: {json.dumps(obs['tool_registry'], separators=(',', ':'))}\n"
            f"Choose your next action as a single JSON object."
        )
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return None


def _safe(v):
    v = float(v)
    if v <= 0.0: return 0.01
    if v >= 1.0: return 0.99
    return min(0.99, max(0.01, v))


def log_start(task, model):
    print(f"[START] task={task} env=spectre model={model}", flush=True)


def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={json.dumps(action)} reward={_safe(reward):.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{_safe(r):.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)



def run_task(task_name):
    env = SpectreEnv(task=task_name, seed=SEED)
    fallback = BaselineAgent()
    obs = env.reset(seed=SEED)
    done = False
    rewards = []
    steps = 0
    success = False

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
    except Exception:
        success = False
    finally:
        log_end(success=success, steps=steps, rewards=rewards)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
        print()
