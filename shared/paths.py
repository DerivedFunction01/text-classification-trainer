from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT_DIR / ".cache"

PATHS: dict[str, Any] = {
    "root_dir": str(ROOT_DIR),
    "cache_root": str(CACHE_ROOT),
}

for value in PATHS.values():
    if isinstance(value, str):
        p = Path(value)
    else:
        p = value
    # Skip file paths
    if p.suffix:
        continue
    p.mkdir(parents=True, exist_ok=True)

