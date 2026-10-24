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
# Fields that change on every run without the data changing. The commit hash is volatile
# because the scheduled job commits its own export: counting it would make every run commit.
VOLATILE_KEYS = {"generatedAt", "runtimeSeconds", "commit"}


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
    """Write ``payload`` unless only volatile fields differ. Returns whether the file changed.

    The file is replaced atomically, and NaN/Infinity are rejected because browsers cannot
    parse them: a bad value fails the run instead of publishing a broken file.
    """
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    if path.exists():
        try:
            if _strip(json.loads(path.read_text())) == _strip(payload):
                return False
        except json.JSONDecodeError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return True


def write_all(files: dict[Path, dict]) -> list[Path]:
    """Serialise every payload first, then write; a failure leaves the previous export intact."""
    for payload in files.values():
        json.dumps(payload, allow_nan=False)
    return [path for path, payload in files.items() if write_json(path, payload)]
