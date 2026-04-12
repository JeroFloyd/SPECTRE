from __future__ import annotations

PRIMITIVES = ["parse_data", "validate_data", "transform_data", "aggregate_result", "export_result"]
ACTION_TYPES = ["primitive", "create_tool", "use_tool"]

def validate_action(action: dict, known_tools: list[str]):
    if not isinstance(action, dict):
        return "Action must be a JSON object"
    atype = action.get("type")
    if atype not in ACTION_TYPES:
        return f"Unknown action type '{atype}'. Must be one of {ACTION_TYPES}"
    name = action.get("name", "")
    if atype == "primitive":
        if name not in PRIMITIVES:
            return f"Unknown primitive '{name}'. Must be one of {PRIMITIVES}"
    elif atype == "create_tool":
        if not name:
            return "create_tool requires a non-empty name"
        sequence = action.get("sequence", [])
        if not isinstance(sequence, list) or len(sequence) < 2:
            return "create_tool sequence must be a list with at least 2 operations"
        valid_ops = set(PRIMITIVES) | set(known_tools)
        bad = [op for op in sequence if op not in valid_ops]
        if bad:
            return f"Unknown operations in sequence: {bad}"
        if name in sequence:
            return f"Circular definition: '{name}' cannot reference itself"
    elif atype == "use_tool":
        if not name:
            return "use_tool requires a non-empty name"
        if name not in known_tools:
            return f"Tool '{name}' not defined. Available: {known_tools}"
    return None

def describe_action(action: dict):
    atype = action.get("type", "?")
    name = action.get("name", "?")
    if atype == "primitive":
        return f"primitive({name})"
    if atype == "create_tool":
        return f"create_tool({name} = {action.get('sequence', [])})"
    if atype == "use_tool":
        return f"use_tool({name})"
    return str(action)
