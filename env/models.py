from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ActionRequest(BaseModel):
    type: str = Field(..., description="One of: primitive | create_tool | use_tool")
    name: Optional[str] = Field(None, description="Primitive name or tool name")
    sequence: Optional[List[str]] = Field(None, description="Op list for create_tool")

    model_config = {"json_schema_extra": {"examples": [
        {"type": "primitive",   "name": "parse_data"},
        {"type": "create_tool", "name": "etl_batch", "sequence": ["parse_data", "validate_data", "transform_data"]},
        {"type": "use_tool",    "name": "etl_batch"},
    ]}}

class ToolRegistryEntry(BaseModel):
    sequence:        List[str]
    expanded_length: int

class ValidationReportSchema(BaseModel):
    total_rows:          int
    missing_required:    Dict[str, int]
    type_errors:         Dict[str, int]
    invalid_enum_values: int
    duplicate_rows:      int
    negative_values:     int
    invalid_dates:       int
    rows_flagged:        int
    rows_clean:          int
    passed:              bool
    quality_score:       float

class PipelineStateSchema(BaseModel):
    source_file:          str
    rows_loaded:          int
    columns:              List[str]
    schema_hash:          str
    validation:           Dict[str, Any]
    rows_after_transform: int
    revenue_total:        float
    derived_columns:      List[str]
    output_path:          str
    rows_exported:        int
    quality_score:        float
    output_hash:          str
    parse_count:          int
    validate_count:       int
    transform_count:      int
    export_count:         int

class Observation(BaseModel):
    task:                 str
    task_description:     str
    session_id:           str
    progress:             int
    target_length:        int
    remaining_steps:      int
    next_required_op:     Optional[str]
    step_count:           int
    max_steps:            int
    available_primitives: List[str]
    available_tools:      List[str]
    custom_tools_defined: List[str]
    tool_registry:        Dict[str, Any]
    compression_ratio:    float
    pipeline_state:       Dict[str, Any]

class RewardBreakdown(BaseModel):
    step_penalty:      float
    progress_bonus:    float
    completion_bonus:  float
    quality_bonus:     float
    compression_bonus: float
    total:             float

class StepInfo(BaseModel):
    error:            Optional[str]
    reward_breakdown: RewardBreakdown
    action:           Dict[str, Any]
    session_id:       str

class StepResponse(BaseModel):
    observation: Observation
    reward:      float
    done:        bool
    info:        StepInfo

class ResetRequest(BaseModel):
    task: str = Field("medium", description="Task difficulty: easy | medium | hard")
    seed: Optional[int] = Field(None, description="RNG seed for reproducible episodes")

class ResetResponse(BaseModel):
    observation: Observation
    reward:      float
    done:        bool
    info:        Dict[str, Any]

class GraderReport(BaseModel):
    session_id:        str
    task:              str
    success:           bool
    steps_taken:       int
    optimal_steps:     int
    efficiency_ratio:  float
    compression_ratio: float
    quality_score:     float
    total_reward:      float
    output_verified:   bool
    output_hash:       str
    verdict:           str
    breakdown:         Dict[str, Any]