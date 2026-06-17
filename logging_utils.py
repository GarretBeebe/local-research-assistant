import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_logger = logging.getLogger("pipeline")
_logger.setLevel(logging.DEBUG)
_handler = RotatingFileHandler(
    str(_LOG_DIR / "pipeline.jsonl"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
_handler.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_handler)

_MAX_VALUE_LEN = 2_000


def _truncate_value(v: object) -> object:
    if isinstance(v, str) and len(v) > _MAX_VALUE_LEN:
        return v[:_MAX_VALUE_LEN] + "...[truncated]"
    if isinstance(v, dict):
        return {k: _truncate_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_truncate_value(item) for item in v]
    return v


def log_stage(stage: str, input_data: dict, output_data: dict) -> None:
    entry = {
        "stage": stage,
        "input": _truncate_value(input_data),
        "output": _truncate_value(output_data),
    }
    _logger.info(json.dumps(entry))
