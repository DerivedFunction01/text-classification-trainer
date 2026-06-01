from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset


def load_hf_dataset(dataset_name: str, *, split: str, config: str | None = None) -> Dataset:
    """Load a single Hugging Face dataset split."""
    if config is None:
        return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name, config, split=split)


def load_hf_dataset_dict(dataset_name: str, *, config: str | None = None) -> DatasetDict:
    """Load a Hugging Face dataset with all available splits."""
    if config is None:
        return load_dataset(dataset_name)
    return load_dataset(dataset_name, config)
