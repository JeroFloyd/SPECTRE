from __future__ import annotations
import os
from typing import List, Optional

from openai import OpenAI
from env.environment import SpectreEnv
from agent.baseline_agent import BaselineAgent

# Environment variables (required by hackathon)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
BENCHMARK = "spectre"
TASKS = ["easy", "medium", "hard", "expert"]
SEED = int(os.getenv("SPECTRE_SEED", "42"))

# Initialize OpenAI client (optional - falls back to baseline agent)
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None


def safe_score(v: float) -> float:
    """Ensure score is STRICTLY between 0 and 1 (not 0.0, not 1.0)."""
    try:
        v = float(v)
    except:
        return 0.5
    
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.99
    
    result = max(0.01, min(0.99, v))
    
    # Paranoid check
    if result <= 0.0 or result >= 1.0:
        return 0.5
    
    return result


def log_start(task: str, env: str, model: str) -> None:
    """[START] task=<task> env=<benchmark> model=<model>"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """[STEP] step=<n> action=<action> reward=<X.XX> done=<true|false> error=<msg|null>"""
    error_val = error if error else "null"
    done_val = "true" if done else "false"
    # Ensure reward is strictly between 0 and 1
    safe_reward = safe_score(reward)
    print(
        f"[STEP] step={step} action={action} reward={safe_reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """[END] success=<true|false> steps=<n> score=<X.XXX> rewards=<r1,r2,...>"""
    success_val = "true" if success else "false"
    # Ensure all rewards and score are strictly between 0 and 1
    safe_rewards = [safe_score(r) for r in rewards]
    safe_final_score = safe_score(score)
    rewards_str = ",".join(f"{r:.2f}" for r in safe_rewards)
    print(
        f"[END] success={success_val} steps={steps} score={safe_final_score:.3f} rewards={rewards_str}",
        flush=True
    )


def run_task(task_name: str) -> None:
    """Run a single SPECTRE task and log results in OpenEnv format."""
    env = SpectreEnv(task=task_name, seed=SEED, batch_file="orders_1.csv")
    agent = BaselineAgent()
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        obs = env.reset(seed=SEED)
        done = False
        
        while not done and steps_taken < env.max_steps:
            # Get action from agent (or could use LLM here)
            action = agent.act(obs)
            
            # Execute step
            obs, reward, done, info = env.step(action)
            steps_taken += 1
            
            # Format action for logging
            action_type = action.get("type", "unknown")
            action_name = action.get("name", "")
            if action_type == "primitive":
                action_str = f"primitive({action_name})"
            elif action_type == "create_tool":
                action_str = f"create_tool({action_name})"
            elif action_type == "use_tool":
                action_str = f"use_tool({action_name})"
            else:
                action_str = str(action)
            
            error = info.get("error")
            rewards.append(reward)
            
            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=done,
                error=error
            )
            
            if done:
                break
        
        # Calculate final score
        total_reward = sum(rewards)
        # Normalize score to [0, 1] based on task difficulty
        max_possible = len(env.target_sequence) * 0.1  # rough estimate
        score = total_reward / max(max_possible, 1.0) if max_possible > 0 else 0.0
        score = safe_score(score)
        
        # Success criteria
        success = obs["progress"] >= obs["target_length"]
        
    except Exception as e:
        print(f"[DEBUG] Task failed: {e}", flush=True)
        success = False
        score = 0.01  # Minimum safe score
        if not rewards:
            rewards = [0.01]
    
    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards
        )


def main() -> None:
    """Run all SPECTRE tasks."""
    for task in TASKS:
        try:
            run_task(task)
            print()  # Blank line between tasks
        except Exception as e:
            print(f"[ERROR] Task {task} crashed: {e}", flush=True)
            # Still log END for failed task
            log_end(success=False, steps=0, score=0.01, rewards=[0.01])
            print()


if __name__ == "__main__":
    main()
