"""Per-tick CSV logger + metadata.json written next to the CSV.

Value encoding (NaN-safe, pandas round-trips it):
- None / missing key   -> ""      (read back as NaN)
- float NaN            -> "nan"   (read back as NaN); +-inf -> "inf" / "-inf"
- bool / numpy bool    -> 1 / 0
- numpy scalars        -> Python scalars; floats use repr (full precision)
Columns ending in `_eval_only` and `clearance_gt_m` carry ground truth for evaluation only.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LOG_COLUMNS: list[str] = [
    "sim_time_s", "scenario", "seed", "mode",
    "robot_x", "robot_y", "robot_yaw", "robot_vx", "robot_vy",
    "obstacle_gt_x_eval_only", "obstacle_gt_y_eval_only",
    "detection_x", "detection_y", "num_detections",
    "track_id", "track_x", "track_y", "track_vx", "track_vy", "track_P_xx", "track_P_yy",
    "tca_s", "dca_m", "margin_m", "risk", "risk_score",
    "nav_state", "cmd_vx", "cmd_vy", "cmd_wz",
    "contact_flag", "fall_flag", "goal_distance_m", "clearance_gt_m",
    "sensor_latency_ms", "planner_latency_ms",
]

_FLUSH_EVERY = 200   # rows


def _fmt(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return int(bool(v))
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    return v


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class CsvLogger:
    """Write one CSV row per planner tick with exactly LOG_COLUMNS.

    Args:
        path: CSV file path; parent directories are created. `metadata.json` (JSON-serialised
            `metadata`, numpy-safe) is written in the same directory immediately.
        metadata: run description (robot, scenario, seed, config, upstream SHA, ...).
    """

    def __init__(self, path: Path, metadata: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.path.parent / "metadata.json"
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata or {}, f, indent=2, default=_json_default)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(LOG_COLUMNS)
        self._n = 0
        self.closed = False

    def log(self, row: dict) -> None:
        """Append one row. Missing keys -> "" ; keys not in LOG_COLUMNS are ignored."""
        if self.closed:
            raise ValueError("CsvLogger is closed")
        self._w.writerow([_fmt(row.get(c)) for c in LOG_COLUMNS])
        self._n += 1
        if self._n % _FLUSH_EVERY == 0:
            self._f.flush()

    @property
    def rows_written(self) -> int:
        return self._n

    def close(self) -> None:
        """Flush and close (idempotent)."""
        if not self.closed:
            self._f.flush()
            self._f.close()
            self.closed = True

    def __enter__(self) -> "CsvLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
