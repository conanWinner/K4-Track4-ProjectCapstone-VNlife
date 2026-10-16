"""Config loading. All thresholds live in configs/*.yaml, not in modules."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str | Path] = None, overrides: Optional[dict] = None) -> dict[str, Any]:
    """Load configs/default.yaml (or `path`), apply nested `overrides`."""
    path = Path(path) if path else CONFIG_DIR / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    return cfg


def load_scenarios(path: Optional[str | Path] = None) -> dict[str, Any]:
    path = Path(path) if path else CONFIG_DIR / "scenarios.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["scenarios"]
