from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.paths import ROOT_DIR


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path

