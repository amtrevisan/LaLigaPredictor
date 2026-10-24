"""Serialise pipeline output into the static JSON contract consumed by the frontend.

The contract is mirrored by ``frontend/src/types/data.ts``; bump ``SCHEMA_VERSION`` on any
breaking change. Files are only rewritten when their content changed beyond the volatile
timestamp fields, so the scheduled job commits nothing when nothing happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
VOLATILE_KEYS = {"generatedAt", "runtimeSeconds"}


def r(value: float | None, digits: int = 4) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


def triple(values: np.ndarray | list[float] | None, digits: int = 4) -> list[float] | None:
    if values is None or np.isnan(np.asarray(values, dtype=float)).any():
        return None
    return [round(float(v), digits) for v in values]


def iso(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _strip(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {k: _strip(v) for k, v in payload.items() if k not in VOLATILE_KEYS}
    if isinstance(payload, list):
        return [_strip(v) for v in payload]
    return payload


def write_json(path: Path, payload: dict) -> bool:
    """Write ``payload`` unless only volatile fields differ. Returns whether the file changed."""
    if path.exists():
        try:
            if _strip(json.loads(path.read_text())) == _strip(payload):
                return False
        except json.JSONDecodeError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return True
