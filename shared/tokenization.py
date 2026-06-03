from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from shared.cache import load_dataset_cache, save_dataset_cache


def tokenize_text_split(
    split: Dataset,
    *,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    max_length: int,
    padding: str = "max_length",
) -> Dataset:
    return split.map(
        lambda batch: _tokenize_text_batch(
            batch,
            tokenizer=tokenizer,
            text_column=text_column,
            max_length=max_length,
            padding=padding,
        ),
        batched=True,
        remove_columns=split.column_names,
    )


def _tokenize_text_batch(
    batch: dict[str, list[Any]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    max_length: int,
    padding: str,
) -> dict[str, Any]:
    enc = tokenizer(
        batch[text_column],
        truncation=True,
        max_length=max_length,
        padding=padding,
    )
    if "labels" in batch:
        enc["labels"] = batch["labels"]
    return enc


def tokenize_pair_split(
    split: Dataset,
    *,
    tokenizer: PreTrainedTokenizerBase,
    text_a_column: str = "premise",
    text_b_column: str = "hypothesis",
    max_length: int,
    padding: str = "max_length",
) -> Dataset:
    return split.map(
        lambda batch: _tokenize_pair_batch(
            batch,
            tokenizer=tokenizer,
            text_a_column=text_a_column,
            text_b_column=text_b_column,
            max_length=max_length,
            padding=padding,
        ),
        batched=True,
        remove_columns=split.column_names,
    )


def _tokenize_pair_batch(
    batch: dict[str, list[Any]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    text_a_column: str,
    text_b_column: str,
    max_length: int,
    padding: str,
) -> dict[str, Any]:
    enc = tokenizer(
        batch[text_a_column],
        batch[text_b_column],
        truncation=True,
        max_length=max_length,
        padding=padding,
    )
    if "labels" in batch:
        enc["labels"] = batch["labels"]
    return enc


def tokenize_dataset_dict(
    dataset: DatasetDict,
    *,
    tokenizer: PreTrainedTokenizerBase,
    kind: str,
    max_length: int,
    text_columns: tuple[str, ...] = ("text",),
    padding: str = "max_length",
) -> DatasetDict:
    tokenized_splits = {}
    for split_name, split in dataset.items():
        if kind == "text":
            tokenized_splits[split_name] = tokenize_text_split(
                split,
                tokenizer=tokenizer,
                text_column=text_columns[0],
                max_length=max_length,
                padding=padding,
            )
        elif kind == "pair":
            tokenized_splits[split_name] = tokenize_pair_split(
                split,
                tokenizer=tokenizer,
                text_a_column=text_columns[0],
                text_b_column=text_columns[1],
                max_length=max_length,
                padding=padding,
            )
        else:
            raise ValueError(f"Unknown tokenization kind: {kind!r}")
    return DatasetDict(tokenized_splits)


def save_tokenized_dataset_cache(
    dataset: DatasetDict,
    cache_dir: str,
    *,
    meta_path: str | None = None,
    meta: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> None:
    save_dataset_cache(dataset, cache_dir, meta_path=meta_path, meta=meta, overwrite=overwrite)


def load_tokenized_dataset_cache(
    cache_dir: str,
    *,
    meta_path: str | None = None,
    expected_meta: dict[str, Any] | None = None,
) -> DatasetDict | None:
    return load_dataset_cache(cache_dir, meta_path=meta_path, expected_meta=expected_meta)
