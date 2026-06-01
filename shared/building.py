from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from datasets import Dataset, DatasetDict


def _normalize_split_size(size: int | float | None, total: int) -> int:
    if size is None:
        return 0
    if isinstance(size, float):
        if not 0.0 <= size < 1.0:
            raise ValueError(f"Fractional split size must be in [0.0, 1.0), got {size!r}")
        return int(total * size)
    if size < 0:
        raise ValueError(f"Split size must be non-negative, got {size!r}")
    return min(int(size), total)


def split_indices(
    total: int,
    *,
    val_size: int | float | None = None,
    test_size: int | float | None = None,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Return shuffled train/val/test index lists."""
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total!r}")

    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)

    val_count = _normalize_split_size(val_size, total)
    test_count = _normalize_split_size(test_size, total)
    if val_count + test_count > total:
        raise ValueError(f"Requested {val_count + test_count} held-out rows from only {total} rows")

    test_indices = indices[:test_count].tolist()
    val_indices = indices[test_count : test_count + val_count].tolist()
    train_indices = indices[test_count + val_count :].tolist()
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def rows_to_dataset(rows: list[dict[str, Any]], *, columns: Iterable[str] | None = None) -> Dataset:
    """Convert row dicts into a Hugging Face Dataset, preserving empty splits."""
    if rows:
        return Dataset.from_list(rows)
    if columns is None:
        raise ValueError("columns must be provided when rows is empty")
    return Dataset.from_dict({column: [] for column in columns})


def rows_to_dataset_dict(
    rows: list[dict[str, Any]],
    *,
    val_size: int | float | None = None,
    test_size: int | float | None = None,
    seed: int = 42,
    columns: Iterable[str] | None = None,
) -> DatasetDict:
    """Shuffle rows and return a DatasetDict with train/val/test splits."""
    split_map = split_indices(len(rows), val_size=val_size, test_size=test_size, seed=seed)
    inferred_columns = list(columns) if columns is not None else (list(rows[0].keys()) if rows else None)
    if inferred_columns is None:
        raise ValueError("columns must be provided when rows is empty")

    return DatasetDict(
        {
            split_name: rows_to_dataset([rows[i] for i in split_indices_for_split], columns=inferred_columns)
            for split_name, split_indices_for_split in split_map.items()
        }
    )

