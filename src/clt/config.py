import math
from pathlib import Path

import yaml


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    positive = ("learning_rate", "gradient_clip", "update_steps", "batch_size", "context_length",
                "max_interactions", "max_checkpoints", "max_states")
    for key in positive:
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("kl_weight", "cost_weight"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("update_steps", "batch_size", "context_length", "max_interactions", "max_checkpoints", "max_states"):
        if not isinstance(config[key], int):
            raise ValueError(f"{key} must be an integer")
    return config
