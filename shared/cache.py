from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict

DEFAULT_CACHE_META_NAME = "dataset.meta.json"


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def ensure_cache_meta(
    cache_dir: str | Path,
    *,
    meta_path: str | Path | None = None,
    meta: dict[str, Any],
) -> bool:
    """Write a missing metadata file for an existing cache directory."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return False

    resolved_meta_path = Path(meta_path) if meta_path is not None else cache_dir / DEFAULT_CACHE_META_NAME
    if resolved_meta_path.exists():
        return False

    write_json_atomic(resolved_meta_path, meta)
    return True


def save_dataset_cache(
    dataset: DatasetDict,
    cache_dir: str | Path,
    *,
    meta_path: str | Path | None = None,
    meta: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> None:
    """Persist a DatasetDict to parquet splits with an optional manifest."""
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing cache dir: {cache_dir}")
        existing_splits = list(cache_dir.glob("*.parquet"))
        if existing_splits:
            raise FileExistsError(
                f"Refusing to overwrite existing cache dir with different data: {cache_dir}"
            )

    temp_dir = cache_dir.parent / f".{cache_dir.name}.tmp-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for split_name, split in dataset.items():
            frame = split.to_pandas()
            frame.to_parquet(temp_dir / f"{split_name}.parquet", index=False)

        if meta is not None:
            meta_name = Path(meta_path).name if meta_path is not None else DEFAULT_CACHE_META_NAME
            resolved_meta_path = temp_dir / meta_name
            write_json_atomic(resolved_meta_path, meta)

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        temp_dir.replace(cache_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def load_dataset_cache(
    cache_dir: str | Path,
    *,
    meta_path: str | Path | None = None,
    expected_meta: dict[str, Any] | None = None,
) -> DatasetDict | None:
    """Load a cached DatasetDict if the manifest matches the expected metadata."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return None

    resolved_meta_path = Path(meta_path) if meta_path is not None else cache_dir / DEFAULT_CACHE_META_NAME
    if meta_path is not None and not resolved_meta_path.exists():
        fallback_meta_path = cache_dir / Path(meta_path).name
        if fallback_meta_path.exists():
            resolved_meta_path = fallback_meta_path
    if expected_meta is not None:
        if not resolved_meta_path.exists():
            return None
        try:
            with resolved_meta_path.open(encoding="utf-8") as f:
                cached_meta = json.load(f)
        except Exception:
            return None
        for key, value in expected_meta.items():
            if cached_meta.get(key) != value:
                return None

    split_paths = sorted(cache_dir.glob("*.parquet"))
    if not split_paths:
        return None

    splits: dict[str, Dataset] = {}
    try:
        for split_path in split_paths:
            split_name = split_path.stem
            frame = pd.read_parquet(split_path)
            splits[split_name] = Dataset.from_pandas(frame, preserve_index=False)
    except Exception:
        return None
    return DatasetDict(splits)
