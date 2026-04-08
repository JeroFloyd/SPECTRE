from __future__ import annotations


class BaselineAgent:
    def __init__(self):
        self._stage = 0

    def reset(self):
        self._stage = 0

    def act(self, obs: dict) -> dict:
        task      = obs["task"]
        tools     = set(obs["custom_tools_defined"])
        remaining = obs["remaining_steps"]

        if task == "easy":
            return self._act_easy(obs)
        elif task == "medium":
            return self._act_medium(tools, remaining)
        elif task == "hard":
            return self._act_hard(tools, remaining)
        else:
            next_op = obs.get("next_required_op")
            if next_op:
                return {"type": "primitive", "name": next_op}
            return {"type": "primitive", "name": "parse_data"}


    def _act_easy(self, obs: dict) -> dict:
        next_op = obs.get("next_required_op")
        if next_op:
            return {"type": "primitive", "name": next_op}
        return {"type": "primitive", "name": "parse_data"}

    def _act_medium(self, tools: set, remaining: int) -> dict:
        if "etl_batch" not in tools:
            return {
                "type":     "create_tool",
                "name":     "etl_batch",
                "sequence": ["parse_data", "validate_data", "transform_data"],
            }
        if remaining >= 3:
            return {"type": "use_tool", "name": "etl_batch"}
        return {"type": "primitive", "name": "parse_data"}

    def _act_hard(self, tools: set, remaining: int) -> dict:
        if "etl_batch" not in tools:
            return {
                "type":     "create_tool",
                "name":     "etl_batch",
                "sequence": ["parse_data", "validate_data", "transform_data"],
            }
        if "triple_etl" not in tools:
            return {
                "type":     "create_tool",
                "name":     "triple_etl",
                "sequence": ["etl_batch", "etl_batch", "etl_batch"],
            }
        if remaining >= 9:
            return {"type": "use_tool", "name": "triple_etl"}
        if remaining == 1:
            return {"type": "primitive", "name": "export_result"}
        return {"type": "use_tool", "name": "etl_batch"}