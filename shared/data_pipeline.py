from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from shared.building import rows_to_dataset_dict
from shared.fetch import load_hf_dataset, load_hf_dataset_dict
from shared.paths import resolve_repo_path
from shared.tokenization import (
    load_tokenized_dataset_cache,
    save_tokenized_dataset_cache,
    tokenize_dataset_dict,
)
from text_utils import MutationConfig, TextMutator


def _resolve_path(value: str | Path) -> Path:
    return resolve_repo_path(value)


def _to_label_lookup_key(value: Any) -> str:
    return str(value)


def _build_multilabel_vector(
    source_label: Any,
    *,
    label_to_indices: dict[str, list[int]],
    num_labels: int,
) -> list[int]:
    labels = [0] * num_labels
    key = _to_label_lookup_key(source_label)
    if key not in label_to_indices:
        raise KeyError(f"Missing multi-label mapping for source label: {source_label!r}")

    for index in label_to_indices[key]:
        if index < 0 or index >= num_labels:
            raise ValueError(f"Target label index out of range: {index!r} for num_labels={num_labels}")
        labels[index] = 1
    return labels


def _normalize_label_targets(
    targets: list[Any],
    *,
    target_label_to_index: dict[str, int],
) -> list[int]:
    indices = []
    for target in targets:
        key = _to_label_lookup_key(target)
        if key not in target_label_to_index:
            raise KeyError(f"Unknown target label in multi-label map: {target!r}")
        indices.append(target_label_to_index[key])
    return indices


def _infer_label_transform(
    dataset: DatasetDict,
    *,
    source_column: str,
) -> tuple[list[Any], dict[str, list[Any]]]:
    seen: set[str] = set()
    target_labels: list[Any] = []
    label_map: dict[str, list[Any]] = {}

    for split_name, split in dataset.items():
        for row in tqdm(split, desc=f"Inferring labels from {split_name}", unit="row"):
            source_label = row[source_column]
            key = _to_label_lookup_key(source_label)
            if key in seen:
                continue
            seen.add(key)
            target_labels.append(source_label)
            label_map[key] = [source_label]

    if not target_labels:
        raise ValueError(f"Could not infer any labels from column {source_column!r}")

    return target_labels, label_map


def _apply_label_transform(dataset: DatasetDict, dataset_cfg: dict[str, Any]) -> DatasetDict:
    label_transform = dataset_cfg.get("label_transform")
    if not label_transform:
        return dataset

    if label_transform.get("type") != "single_to_multi_label":
        raise ValueError(f"Unsupported label_transform type: {label_transform.get('type')!r}")

    source_column = label_transform.get("source_column", dataset_cfg["label_column"])
    output_column = label_transform.get("output_column", "labels")
    target_labels = label_transform.get("target_labels")
    label_map = label_transform.get("label_map")
    if target_labels is None and label_map is None:
        target_labels, label_map = _infer_label_transform(dataset, source_column=source_column)
    if not isinstance(target_labels, list) or not target_labels:
        raise ValueError("label_transform.target_labels must be a non-empty list or omitted for inference")
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("label_transform.label_map must be a non-empty object or omitted for inference")

    target_label_to_index = {_to_label_lookup_key(label): index for index, label in enumerate(target_labels)}
    label_to_indices: dict[str, list[int]] = {}
    for raw_source_label, raw_targets in label_map.items():
        if not isinstance(raw_targets, list):
            raise ValueError(
                f"label_transform.label_map[{raw_source_label!r}] must be a list of target labels"
            )
        label_to_indices[_to_label_lookup_key(raw_source_label)] = _normalize_label_targets(
            raw_targets,
            target_label_to_index=target_label_to_index,
        )

    def transform_split(split: Dataset) -> Dataset:
        def map_row(row: dict[str, Any]) -> dict[str, Any]:
            row = dict(row)
            row[output_column] = _build_multilabel_vector(
                row[source_column],
                label_to_indices=label_to_indices,
                num_labels=len(target_labels),
            )
            return row

        return split.map(map_row)

    return DatasetDict({split_name: transform_split(split) for split_name, split in dataset.items()})


def _mutation_config_from_dict(config: dict[str, Any] | None) -> MutationConfig:
    if not config:
        return MutationConfig()

    allowed_keys = {
        "keep_original",
        "boundary_strip_prob",
        "sentence_mutation_prob",
        "sentence_casing_prob",
        "word_casing_prob",
        "spacing_noise_prob",
        "char_noise_prob",
        "accent_strip_prob",
        "format_noise_prob",
        "script_letter_prob",
        "script_digit_prob",
        "sentence_uppercase_prob",
        "sentence_lowercase_prob",
        "word_uppercase_prob",
        "word_lowercase_prob",
        "word_titlecase_prob",
        "merge_word_prob",
        "split_word_prob",
        "ocr_char_prob",
        "keyboard_char_prob",
        "unicode_accent_char_prob",
        "max_sentence_edits",
        "max_word_edits",
        "safe_accent_strip_langs",
    }
    unknown_keys = sorted(set(config) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown mutation config keys: {unknown_keys}")

    kwargs = dict(config)
    if "safe_accent_strip_langs" in kwargs and kwargs["safe_accent_strip_langs"] is not None:
        kwargs["safe_accent_strip_langs"] = set(kwargs["safe_accent_strip_langs"])
    return MutationConfig(**kwargs)


def _apply_text_perturbation(dataset: DatasetDict, dataset_cfg: dict[str, Any]) -> DatasetDict:
    perturbation = dataset_cfg.get("text_perturbation")
    if not perturbation:
        return dataset

    source_column = perturbation.get("source_column", dataset_cfg["text_columns"][0])
    output_column = perturbation.get("output_column", source_column)
    lang_column = perturbation.get("lang_column")
    num_variants = int(perturbation.get("num_variants", 1))
    if num_variants < 1:
        raise ValueError("text_perturbation.num_variants must be >= 1")

    mutation_config = _mutation_config_from_dict(perturbation.get("mutation_config"))
    mutator = TextMutator(mutation_config)
    base_seed = int(dataset_cfg.get("seed", 42))

    def transform_split(split_name: str, split: Dataset) -> Dataset:
        if num_variants == 1 and not mutation_config.keep_original:
            raise ValueError("text_perturbation would drop all rows because keep_original is false and num_variants is 1")

        seed_offset = abs(hash(split_name)) % (2**32)

        transformed_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(tqdm(split.to_list(), desc=f"Perturbing {split_name}", unit="row")):
            text = row.get(source_column)
            if not isinstance(text, str):
                raise TypeError(f"Expected text column {source_column!r} to contain strings, got {type(text)!r}")

            rng = random.Random(base_seed + seed_offset + row_index)
            variants = mutator.augment(text, rng=rng, lang=row.get(lang_column) if lang_column else None)
            if num_variants == 1:
                chosen_variant = variants[0] if variants else text
                mutated_row = dict(row)
                mutated_row[output_column] = chosen_variant
                transformed_rows.append(mutated_row)
                continue

            if mutation_config.keep_original:
                transformed_rows.append(dict(row))

            for variant in variants[:num_variants]:
                mutated_row = dict(row)
                mutated_row[output_column] = variant
                transformed_rows.append(mutated_row)

        if not transformed_rows:
            return Dataset.from_dict({column: [] for column in split.column_names})
        return Dataset.from_list(transformed_rows)

    return DatasetDict({split_name: transform_split(split_name, split) for split_name, split in dataset.items()})

def load_dataset_from_config(config: dict[str, Any]) -> DatasetDict:
    source = config["dataset"]["source"]
    source_type = source["type"]
    dataset_cfg = config["dataset"]

    if source_type == "hf_dataset_dict":
        dataset = load_hf_dataset_dict(source["dataset_name"], config=source.get("config"))
        if isinstance(dataset, DatasetDict):
            return dataset
        raise TypeError("Expected DatasetDict from hf_dataset_dict source")

    if source_type == "hf_dataset":
        split = source["split"]
        dataset = load_hf_dataset(source["dataset_name"], split=split, config=source.get("config"))
        if not isinstance(dataset, Dataset):
            raise TypeError("Expected Dataset from hf_dataset source")
        rows = dataset.to_list()
        return rows_to_dataset_dict(
            rows,
            val_size=dataset_cfg["split_strategy"].get("val_size"),
            test_size=dataset_cfg["split_strategy"].get("test_size"),
            seed=dataset_cfg.get("seed", 42),
            columns=dataset.column_names,
        )

    if source_type == "local_parquet":
        loaded = load_from_disk(str(_resolve_path(source["path"])))
        if isinstance(loaded, DatasetDict):
            return loaded
        raise TypeError("Expected DatasetDict from local_parquet source")

    raise ValueError(f"Unknown dataset source type: {source_type!r}")


def build_and_cache_dataset(config: dict[str, Any]) -> DatasetDict:
    dataset_cfg = config["dataset"]
    cache_dir = _resolve_path(dataset_cfg["cache_dir"])
    label_column = dataset_cfg["label_column"]
    meta_path = cache_dir / "dataset.meta.json"
    expected_meta = {
        "dataset": dataset_cfg,
        "tokenization": config["tokenization"],
    }
    cached = load_tokenized_dataset_cache(
        str(cache_dir),
        meta_path=str(meta_path),
        expected_meta=expected_meta,
    )
    if cached is not None:
        return cached

    raw_dataset = load_dataset_from_config(config)
    raw_dataset = _apply_label_transform(raw_dataset, dataset_cfg)
    raw_dataset = _apply_text_perturbation(raw_dataset, dataset_cfg)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenized = tokenize_dataset_dict(
        raw_dataset,
        tokenizer=tokenizer,
        kind=config["tokenization"]["kind"],
        text_columns=tuple(dataset_cfg["text_columns"]),
        label_columns=tuple(config["tokenization"]["label_columns"]),
        max_length=config["tokenization"]["max_length"],
        padding=config["tokenization"].get("padding", "max_length"),
    )
    if label_column != "labels" and "labels" not in tokenized["train"].column_names:
        tokenized = tokenized.rename_column(label_column, "labels")
    save_tokenized_dataset_cache(
        tokenized,
        str(cache_dir),
        meta_path=str(meta_path),
        meta=expected_meta,
        overwrite=True,
    )
    return tokenized
