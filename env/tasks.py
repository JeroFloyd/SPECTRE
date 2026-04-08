TASK_REGISTRY: dict[str, dict] = {

    "easy": {
        "sequence": [
            "parse_data",
            "validate_data",
            "parse_data",
            "validate_data",
        ],
        "max_steps": 20,
        "description": (
            "Ingest and validate two separate data batches. "
            "No transformation or export required. "
            "Optimal agent executes all 4 primitives directly (4 steps)."
        ),
    },

    "medium": {
        "sequence": [
            "parse_data",  "validate_data", "transform_data",
            "parse_data",  "validate_data", "transform_data",
        ],
        "max_steps": 30,
        "description": (
            "Run a full parse → validate → transform ETL cycle on two separate batches. "
            "Optimal agent creates an 'etl_batch' tool for the 3-step cycle and invokes it "
            "twice, completing the pipeline in 4 total steps instead of 6."
        ),
    },

    "hard": {
        "sequence": [
            "parse_data",  "validate_data", "transform_data",
            "parse_data",  "validate_data", "transform_data",
            "parse_data",  "validate_data", "transform_data",
            "export_result",
        ],
        "max_steps": 50,
        "description": (
            "Run three full ETL cycles across three data batches, then export the final result. "
            "Optimal agent builds hierarchical tools: 'etl_batch' (3 ops) → 'triple_etl' (3× etl_batch) "
            "→ invokes triple_etl → export_result. Completes in 5 steps instead of 10, "
            "demonstrating full self-programming leverage."
        ),
    },

}