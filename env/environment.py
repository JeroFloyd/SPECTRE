from __future__ import annotations

import uuid
from pathlib import Path

from env.tasks    import TASK_REGISTRY
from env.actions  import validate_action, PRIMITIVES
from env.rewards  import compute_reward
from env.pipeline import PipelineState, parse, validate, transform, export


class SpectreEnv:

    def __init__(self, task: str = "medium", seed: int = 42):
        self.task_name = task
        self.seed      = seed
        self.max_steps = TASK_REGISTRY[task]["max_steps"]
        self.reset(seed=seed)

    def reset(self, seed: int = 42) -> dict:
        task_cfg = TASK_REGISTRY[self.task_name]

        self.target_sequence  = task_cfg["sequence"]
        self.max_steps        = task_cfg["max_steps"]
        self.task_description = task_cfg["description"]

        self.progress      = 0
        self.prev_progress = 0
        self.step_count    = 0

        self.custom_tools  = {}
        self.tool_registry = {}
        self._step_log     = []
        self.session_id    = str(uuid.uuid4())

        self._pipeline = PipelineState(
            task     = self.task_name,
            data_dir = Path("data"),
            seed     = seed,
        )

        return self.state()

    def state(self) -> dict:
        remaining = len(self.target_sequence) - self.progress
        next_op   = (self.target_sequence[self.progress]
                     if self.progress < len(self.target_sequence) else None)
        EPS = 1e-4

        compression = self.progress / max(self.step_count, 1)

        compression = max(EPS, min(0.9999, compression))

        compression = float(f"{compression:.4f}")

        return {
            "task":                 self.task_name,
            "task_description":     self.task_description,
            "session_id":           self.session_id,
            "progress":             self.progress,
            "target_length":        len(self.target_sequence),
            "remaining_steps":      remaining,
            "next_required_op":     next_op,
            "step_count":           self.step_count,
            "max_steps":            self.max_steps,
            "available_primitives": PRIMITIVES,
            "available_tools":      list(self.custom_tools.keys()),
            "custom_tools_defined": list(self.custom_tools.keys()),
            "tool_registry":        self.tool_registry,
            "compression_ratio":    compression,
            "pipeline_state":       self._pipeline.summary(),
        }

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        self.step_count    += 1
        self.prev_progress  = self.progress
        error               = None

        schema_error = validate_action(action, list(self.custom_tools.keys()))
        if schema_error:
            reward, breakdown = compute_reward(
                step_count=self.step_count, max_steps=self.max_steps,
                done=False, progress=self.progress,
                target_length=len(self.target_sequence),
                prev_progress=self.prev_progress,
                pipeline=self._pipeline, custom_tools=self.custom_tools,
            )
            info = {"error": schema_error, "reward_breakdown": breakdown,
                    "action": action, "session_id": self.session_id}
            self._step_log.append({"action": action, "reward": reward, "info": info})
            return self.state(), reward, False, info

        try:
            atype = action["type"]

            if atype == "primitive":
                error = self._apply_primitive(action["name"])

            elif atype == "create_tool":
                name     = action["name"]
                sequence = action["sequence"]
                self.custom_tools[name]  = sequence
                self.tool_registry[name] = {
                    "sequence":        sequence,
                    "expanded_length": self._expand_length(sequence),
                }

            elif atype == "use_tool":
                error = self._apply_tool(action["name"])

        except Exception as exc:
            error = str(exc)

        done = self.progress >= len(self.target_sequence)
        if done and self.task_name in ["medium", "hard"]:
            if self._pipeline.export_count == 0:
                export(self._pipeline)
        if self.step_count >= self.max_steps:
            done = True

        reward, breakdown = compute_reward(
            step_count=self.step_count, max_steps=self.max_steps,
            done=done, progress=self.progress,
            target_length=len(self.target_sequence),
            prev_progress=self.prev_progress,
            pipeline=self._pipeline, custom_tools=self.custom_tools,
        )

        info = {"error": error, "reward_breakdown": breakdown,
                "action": action, "session_id": self.session_id}
        self._step_log.append({"action": action, "reward": reward, "info": info})
        return self.state(), reward, done, info

    def _apply_primitive(self, name: str) -> str | None:
        if self.progress >= len(self.target_sequence):
            return "Episode already complete"

        expected = self.target_sequence[self.progress]

        if name != expected:
            return f"Wrong operation: expected '{expected}', got '{name}'"

        pipeline_error = self._run_pipeline_op(name)
        if pipeline_error:
            return pipeline_error

        self.progress += 1
        return None

    def _run_pipeline_op(self, name: str) -> str | None:
        if name == "parse_data":
            return parse(self._pipeline)
        elif name == "validate_data":
            return validate(self._pipeline)
        elif name == "transform_data":
            return transform(self._pipeline)
        elif name == "export_result":
            return export(self._pipeline)
        return f"Unknown primitive: {name}"

    def _apply_tool(self, name: str) -> str | None:
        sequence      = self.custom_tools[name]
        expanded      = self._expand_sequence(sequence)
        remaining_ops = list(self.target_sequence[self.progress:])

        if not remaining_ops:
            return "No remaining ops to complete"

        if len(expanded) > len(remaining_ops):
            return (f"Tool '{name}' expands to {len(expanded)} ops "
                    f"but only {len(remaining_ops)} remain")

        required_slice = remaining_ops[:len(expanded)]
        if expanded != required_slice:
            return (f"Tool '{name}' expands to {expanded} but "
                    f"next required ops are {required_slice}")

        for op in expanded:
            err = self._apply_primitive(op)
            if err:
                return err

        return None

    def _expand_sequence(self, sequence: list[str]) -> list[str]:
        result = []
        for op in sequence:
            if op in self.custom_tools:
                result.extend(self._expand_sequence(self.custom_tools[op]))
            else:
                result.append(op)
        return result

    def _expand_length(self, sequence: list[str]) -> int:
        return len(self._expand_sequence(sequence))
