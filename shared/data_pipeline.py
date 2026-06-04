from __future__ import annotations

import os
import random
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Value, load_from_disk
from datasets import concatenate_datasets
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
from shared.archive import zip_cache_subdir
from text_utils import MutationConfig, TextMutator

CACHE_FORMAT_VERSION = 2
_PERTURBATION_MUTATOR: TextMutator | None = None


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
        if key == "__neutral__":
            return [0] * num_labels
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
    neutral_label_value: str = "__neutral__",
) -> tuple[list[Any], dict[str, list[Any]]]:
    seen: set[str] = set()
    target_labels: list[Any] = []
    label_map: dict[str, list[Any]] = {}

    for split_name, split in dataset.items():
        for row in tqdm(split, desc=f"Inferring labels from {split_name}", unit="row"):
            source_label = row[source_column]
            key = _to_label_lookup_key(source_label)
            if key == neutral_label_value:
                continue
            if key in seen:
                continue
            seen.add(key)
            target_labels.append(source_label)
            label_map[key] = [source_label]

    if not target_labels:
        raise ValueError(f"Could not infer any labels from column {source_column!r}")

    return target_labels, label_map


def _build_label_display_names(
    target_labels: list[Any],
    *,
    label_names: dict[Any, Any] | None = None,
) -> list[str]:
    display_names: list[str] = []
    label_names = label_names or {}
    for label in target_labels:
        display_value = label_names.get(label, label_names.get(_to_label_lookup_key(label), label))
        if display_value is None:
            display_value = label
        display_names.append(_to_label_lookup_key(display_value))
    return display_names


def _combine_text_values(primary: Any, secondary: Any, *, separator: str = " ") -> Any:
    if isinstance(primary, str) and isinstance(secondary, str):
        primary = primary.strip()
        secondary = secondary.strip()
        if primary and secondary:
            return f"{primary}{separator}{secondary}"
        return primary or secondary
    return primary


def _combine_text_series(values: list[Any], *, separator: str = " ") -> Any:
    text_values = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    if not text_values:
        return values[0] if values else ""
    return separator.join(text_values)


def _remove_items_by_identity(rows: list[dict[str, Any]], items_to_remove: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining: list[dict[str, Any]] = []
    removal_ids = {id(item) for item in items_to_remove}
    for row in rows:
        if id(row) not in removal_ids:
            remaining.append(row)
    return remaining


def _cast_multilabel_labels_to_float(dataset: DatasetDict, *, label_column: str = "labels") -> DatasetDict:
    def cast_split(split: Dataset) -> Dataset:
        return split.map(
            lambda row: {label_column: [float(value) for value in row[label_column]]},
            desc="Casting multi-label targets to float",
        )

    return DatasetDict({split_name: cast_split(split) for split_name, split in dataset.items()})


def _normalize_score_dict_labels(
    dataset: DatasetDict,
    *,
    label_column: str,
) -> tuple[DatasetDict, dict[str, Any] | None]:
    label_order: list[str] | None = None

    for split in dataset.values():
        for row in split:
            value = row.get(label_column)
            if isinstance(value, dict):
                label_order = [str(key) for key in value.keys()]
                break
        if label_order is not None:
            break

    if label_order is None:
        return dataset, None

    def map_row(row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        value = row.get(label_column)
        if isinstance(value, dict):
            row[label_column] = [float(value.get(name, 0.0)) for name in label_order]
        return row

    transformed = DatasetDict(
        {split_name: split.map(map_row, desc="Normalizing score-dict labels") for split_name, split in dataset.items()}
    )
    metadata = {
        "label2id": {name: index for index, name in enumerate(label_order)},
        "id2label": {str(index): name for index, name in enumerate(label_order)},
        "num_labels": len(label_order),
        "label_column": label_column,
        "output_column": "labels",
        "task_type": "multi_label_classification",
        "raw_labels": label_order,
    }
    return transformed, metadata


def _apply_label_aliases(
    dataset: DatasetDict,
    aliases: dict[str, Any],
    *,
    label_column: str,
) -> DatasetDict:
    if not aliases:
        return dataset

    normalized_aliases = {_to_label_lookup_key(key): value for key, value in aliases.items()}

    def transform_split(split: Dataset, *, label_column: str) -> Dataset:
        def map_row(row: dict[str, Any]) -> dict[str, Any]:
            row = dict(row)
            label_value = row.get(label_column)
            alias_key = _to_label_lookup_key(label_value)
            if alias_key not in normalized_aliases:
                raise KeyError(f"Missing label alias for value: {label_value!r}")
            row[label_column] = normalized_aliases[alias_key]
            return row

        return split.map(map_row)

    return DatasetDict(
        {split_name: transform_split(split, label_column=label_column) for split_name, split in dataset.items()}
    )


def _apply_neutral_labels(dataset: DatasetDict, output_label_column: str) -> DatasetDict:
    def transform_split(split: Dataset) -> Dataset:
        def map_row(row: dict[str, Any]) -> dict[str, Any]:
            row = dict(row)
            row[output_label_column] = "__neutral__"
            return row

        return split.map(map_row)

    return DatasetDict({split_name: transform_split(split) for split_name, split in dataset.items()})


def _load_single_source_dataset(
    source: dict[str, Any],
    *,
    dataset_cfg: dict[str, Any],
) -> DatasetDict:
    source_type = source["type"]

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


def _load_sources_from_config(config: dict[str, Any]) -> DatasetDict:
    dataset_cfg = config["dataset"]
    sources = dataset_cfg.get("sources")
    if sources is None:
        return _load_single_source_dataset(dataset_cfg["source"], dataset_cfg=dataset_cfg)
    if not isinstance(sources, list) or not sources:
        raise ValueError("dataset.sources must be a non-empty list when provided")

    loaded_splits: dict[str, list[Dataset]] = {}
    for source in sources:
        if not source.get("enabled", True):
            continue
        source_dataset = _load_single_source_dataset(source, dataset_cfg=dataset_cfg)
        source_text_column = source.get("text_column", dataset_cfg["text_columns"][0])
        source_label_column = source.get("label_column")
        aliases = source.get("label_aliases") or source.get("label_names_map") or {}
        if source.get("neutral"):
            source_dataset = _apply_neutral_labels(source_dataset, dataset_cfg["label_column"])
        else:
            if source_label_column is None:
                raise ValueError(f"Source {source.get('name', '<unnamed>')!r} must define label_column")
            source_dataset = _apply_label_aliases(source_dataset, aliases, label_column=source_label_column)
        for split_name, split in source_dataset.items():
            if source_text_column != dataset_cfg["text_columns"][0]:
                split = split.rename_column(source_text_column, dataset_cfg["text_columns"][0])
            if source_label_column and source_label_column != dataset_cfg["label_column"] and source_label_column in split.column_names:
                split = split.rename_column(source_label_column, dataset_cfg["label_column"])
            if dataset_cfg["label_column"] in split.column_names:
                split = split.map(
                    lambda row: {dataset_cfg["label_column"]: str(row[dataset_cfg["label_column"]])},
                    desc=f"Normalizing labels for {source.get('name', '<unnamed>')}/{split_name}",
                )
                split = split.cast_column(dataset_cfg["label_column"], Value("string"))
            keep_columns = [column for column in (dataset_cfg["text_columns"][0], dataset_cfg["label_column"]) if column in split.column_names]
            drop_columns = [column for column in split.column_names if column not in keep_columns]
            if drop_columns:
                split = split.remove_columns(drop_columns)
            loaded_splits.setdefault(split_name, []).append(split)

    if not loaded_splits:
        raise ValueError("No enabled dataset sources were found in dataset.sources")

    combined = {
        split_name: concatenate_datasets(splits)
        for split_name, splits in loaded_splits.items()
    }
    return DatasetDict(combined)


def _source_is_neutral(source: dict[str, Any]) -> bool:
    return bool(source.get("neutral", False))


def _apply_label_transform(
    dataset: DatasetDict,
    dataset_cfg: dict[str, Any],
) -> tuple[DatasetDict, dict[str, Any] | None]:
    label_transform = dataset_cfg.get("label_transform")
    if not label_transform:
        return dataset, None

    if label_transform.get("type") != "single_to_multi_label":
        raise ValueError(f"Unsupported label_transform type: {label_transform.get('type')!r}")

    source_column = label_transform.get("source_column", dataset_cfg["label_column"])
    output_column = label_transform.get("output_column", "labels")
    reserve_fraction = float(label_transform.get("reserve_fraction", 0.0))
    reserve_separator = str(label_transform.get("reserve_separator", " "))
    same_label_pack_size = int(label_transform.get("same_label_pack_size", 3))
    neutral_label_value = _to_label_lookup_key(label_transform.get("neutral_label", "__neutral__"))
    target_labels = label_transform.get("target_labels")
    label_map = label_transform.get("label_map")
    label_names = label_transform.get("label_names")
    neutral_label = label_transform.get("neutral_label", "__neutral__")
    if target_labels is None and label_map is None:
        target_labels, label_map = _infer_label_transform(
            dataset,
            source_column=source_column,
            neutral_label_value=neutral_label_value,
        )
    if not isinstance(target_labels, list) or not target_labels:
        raise ValueError("label_transform.target_labels must be a non-empty list or omitted for inference")
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("label_transform.label_map must be a non-empty object or omitted for inference")
    if label_names is not None and not isinstance(label_names, dict):
        raise ValueError("label_transform.label_names must be an object when provided")
    if not 0.0 <= reserve_fraction <= 1.0:
        raise ValueError("label_transform.reserve_fraction must be in the range [0.0, 1.0]")
    if same_label_pack_size < 1:
        raise ValueError("label_transform.same_label_pack_size must be >= 1")
    if reserve_fraction > 0.0 and len(target_labels) < 2:
        raise ValueError("label_transform.reserve_fraction requires at least 2 target labels")
    if neutral_label is None:
        neutral_label = "__neutral__"
    text_columns = tuple(dataset_cfg.get("text_columns", ()))
    if reserve_fraction > 0.0 and not text_columns:
        raise ValueError("label_transform.reserve_fraction requires dataset.text_columns to be set")

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
        source_label_rows: dict[str, list[dict[str, Any]]] = {}
        source_label_order: list[str] = []
    for split_name, split in dataset.items():
        for row in split:
            key = _to_label_lookup_key(row[source_column])
            if key not in source_label_rows:
                source_label_rows[key] = []
                source_label_order.append(key)
            source_label_rows[key].append(dict(row))

    def _build_pack(chunk: list[dict[str, Any]]) -> dict[str, Any]:
        row = dict(chunk[0])
        for column in text_columns:
            if column in row:
                row[column] = _combine_text_series(
                    [item[column] for item in chunk if column in item],
                    separator=reserve_separator,
                )
        row[output_column] = _build_multilabel_vector(
            row[source_column],
            label_to_indices=label_to_indices,
            num_labels=len(target_labels),
        )
        if source_column != output_column and source_column in row:
            del row[source_column]
        return row

    def transform_split(split_name: str, split: Dataset) -> Dataset:
        split_rows = split.to_list()
        rows_by_label: dict[str, list[dict[str, Any]]] = {}
        for row in split_rows:
            key = _to_label_lookup_key(row[source_column])
            rows_by_label.setdefault(key, []).append(dict(row))

        transformed_rows: list[dict[str, Any]] = []
        split_hash = int.from_bytes(hashlib.sha256(split_name.encode("utf-8")).digest()[:4], "big")
        reserve_rng = random.Random(int(dataset_cfg.get("seed", 42)) + split_hash)

        reserve_rows_by_label: dict[str, list[dict[str, Any]]] = {}
        free_rows_by_label: dict[str, list[dict[str, Any]]] = {}
        for label_key, rows in rows_by_label.items():
            if label_key == neutral_label_value:
                reserve_rows_by_label[label_key] = []
                free_rows_by_label[label_key] = list(rows)
                continue
            reserve_count = int(len(rows) * reserve_fraction) if reserve_fraction > 0.0 else 0
            reserve_rows = reserve_rng.sample(rows, k=reserve_count) if reserve_count > 0 else []
            reserve_rows_by_label[label_key] = reserve_rows
            free_rows_by_label[label_key] = _remove_items_by_identity(rows, reserve_rows)

        for label_key in source_label_order:
            free_rows = free_rows_by_label.get(label_key, [])
            for start in range(0, len(free_rows), same_label_pack_size):
                chunk = free_rows[start : start + same_label_pack_size]
                if chunk:
                    transformed_rows.append(_build_pack(chunk))

        if reserve_fraction > 0.0:
            label_keys = list(source_label_order)
            for label_key in source_label_order:
                if label_key == neutral_label_value:
                    continue
                reserve_rows = reserve_rows_by_label.get(label_key, [])
                other_labels = [candidate for candidate in label_keys if candidate != label_key and candidate != neutral_label_value]
                if not other_labels:
                    raise ValueError("reserve_fraction requires at least 2 distinct labels")
                for reserve_row in reserve_rows:
                    partner_label = reserve_rng.choice(other_labels)
                    partner_rows = free_rows_by_label.get(partner_label, [])
                    if not partner_rows:
                        partner_rows = rows_by_label.get(partner_label, [])
                    partner_row = dict(reserve_rng.choice(partner_rows))
                    mixed_row = dict(reserve_row)
                    for column in text_columns:
                        if column in mixed_row and column in partner_row:
                            mixed_row[column] = _combine_text_values(
                                mixed_row[column],
                                partner_row[column],
                                separator=reserve_separator,
                            )
                    mixed_row[output_column] = _build_multilabel_vector(
                        reserve_row[source_column],
                        label_to_indices=label_to_indices,
                        num_labels=len(target_labels),
                    )
                    for target_index in label_to_indices[partner_label]:
                        mixed_row[output_column][target_index] = 1
                    if source_column != output_column and source_column in mixed_row:
                        del mixed_row[source_column]
                    transformed_rows.append(mixed_row)

        if not transformed_rows:
            return Dataset.from_dict({column: [] for column in split.column_names})
        return Dataset.from_list(transformed_rows)

    display_labels = _build_label_display_names(target_labels, label_names=label_names)
    label2id = {display_label: index for index, display_label in enumerate(display_labels)}
    id2label = {str(index): display_label for index, display_label in enumerate(display_labels)}
    metadata = {
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": len(target_labels),
        "label_column": source_column,
        "output_column": output_column,
        "task_type": "multi_label_classification",
        "raw_labels": [_to_label_lookup_key(label) for label in target_labels],
        "neutral_label": _to_label_lookup_key(neutral_label),
    }
    return DatasetDict({split_name: transform_split(split_name, split) for split_name, split in dataset.items()}), metadata


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


def _init_perturbation_worker(mutation_config: dict[str, Any]) -> None:
    global _PERTURBATION_MUTATOR
    _PERTURBATION_MUTATOR = TextMutator(MutationConfig(**mutation_config))


def _perturb_row(
    row: dict[str, Any],
    *,
    source_column: str,
    output_column: str,
    lang_column: str | None,
    base_seed: int,
    seed_offset: int,
    row_index: int,
    num_variants: int,
) -> list[dict[str, Any]]:
    if _PERTURBATION_MUTATOR is None:
        raise RuntimeError("Perturbation worker was not initialized")

    text = row.get(source_column)
    if not isinstance(text, str):
        raise TypeError(f"Expected text column {source_column!r} to contain strings, got {type(text)!r}")

    rng = random.Random(base_seed + seed_offset + row_index)
    variants = _PERTURBATION_MUTATOR.augment(text, rng=rng, lang=row.get(lang_column) if lang_column else None)
    if num_variants == 1:
        chosen_variant = variants[0] if variants else text
        mutated_row = dict(row)
        mutated_row[output_column] = chosen_variant
        return [mutated_row]

    output_rows: list[dict[str, Any]] = []
    if _PERTURBATION_MUTATOR.config.keep_original:
        output_rows.append(dict(row))

    for variant in variants[:num_variants]:
        mutated_row = dict(row)
        mutated_row[output_column] = variant
        output_rows.append(mutated_row)
    return output_rows


def _perturb_rows_chunk(
    rows: list[dict[str, Any]],
    *,
    source_column: str,
    output_column: str,
    lang_column: str | None,
    base_seed: int,
    seed_offset: int,
    start_row_index: int,
    num_variants: int,
) -> list[dict[str, Any]]:
    if _PERTURBATION_MUTATOR is None:
        raise RuntimeError("Perturbation worker was not initialized")

    transformed_rows: list[dict[str, Any]] = []
    for offset, row in enumerate(rows):
        row_index = start_row_index + offset
        text = row.get(source_column)
        if not isinstance(text, str):
            raise TypeError(f"Expected text column {source_column!r} to contain strings, got {type(text)!r}")

        rng = random.Random(base_seed + seed_offset + row_index)
        variants = _PERTURBATION_MUTATOR.augment(text, rng=rng, lang=row.get(lang_column) if lang_column else None)
        if num_variants == 1:
            chosen_variant = variants[0] if variants else text
            mutated_row = dict(row)
            mutated_row[output_column] = chosen_variant
            transformed_rows.append(mutated_row)
            continue

        if _PERTURBATION_MUTATOR.config.keep_original:
            transformed_rows.append(dict(row))

        for variant in variants[:num_variants]:
            mutated_row = dict(row)
            mutated_row[output_column] = variant
            transformed_rows.append(mutated_row)
    return transformed_rows


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
    base_seed = int(dataset_cfg.get("seed", 42))
    max_workers = max((os.cpu_count() or 1) - 1, 1)
    mutation_config_dict = dict(perturbation.get("mutation_config") or {})

    def transform_split(split_name: str, split: Dataset) -> Dataset:
        if num_variants == 1 and not mutation_config.keep_original:
            raise ValueError("text_perturbation would drop all rows because keep_original is false and num_variants is 1")

        seed_offset = abs(hash(split_name)) % (2**32)

        split_rows = split.to_list()
        transformed_rows: list[dict[str, Any]] = []
        chunk_count = min(max_workers, len(split_rows))
        if chunk_count <= 0:
            return Dataset.from_dict({column: [] for column in split.column_names})
        chunk_size = (len(split_rows) + chunk_count - 1) // chunk_count
        chunks: list[tuple[int, list[dict[str, Any]]]] = []
        for chunk_index in range(chunk_count):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, len(split_rows))
            if start >= end:
                break
            chunks.append((start, split_rows[start:end]))

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_perturbation_worker,
            initargs=(mutation_config_dict,),
        ) as executor:
            tasks = [
                executor.submit(
                    _perturb_rows_chunk,
                    rows,
                    source_column=source_column,
                    output_column=output_column,
                    lang_column=lang_column,
                    base_seed=base_seed,
                    seed_offset=seed_offset,
                    start_row_index=start_row_index,
                    num_variants=num_variants,
                )
                for start_row_index, rows in chunks
            ]
            for task in tqdm(tasks, total=len(tasks), desc=f"Perturbing {split_name}", unit="chunk"):
                transformed_rows.extend(task.result())

        if not transformed_rows:
            return Dataset.from_dict({column: [] for column in split.column_names})
        return Dataset.from_list(transformed_rows)

    return DatasetDict({split_name: transform_split(split_name, split) for split_name, split in dataset.items()})

def load_dataset_from_config(config: dict[str, Any]) -> DatasetDict:
    return _load_sources_from_config(config)


def build_and_cache_dataset(config: dict[str, Any]) -> DatasetDict:
    dataset_cfg = config["dataset"]
    cache_dir = _resolve_path(dataset_cfg["cache_dir"])
    label_column = dataset_cfg["label_column"]
    meta_path = cache_dir / "dataset.meta.json"
    expected_meta = {
        "dataset": dataset_cfg,
        "tokenization": config["tokenization"],
        "cache_format_version": CACHE_FORMAT_VERSION,
    }
    cached = load_tokenized_dataset_cache(
        str(cache_dir),
        meta_path=str(meta_path),
        expected_meta=expected_meta,
    )
    if cached is not None:
        return cached

    raw_dataset = load_dataset_from_config(config)
    raw_dataset, label_metadata = _apply_label_transform(raw_dataset, dataset_cfg)
    if config["task_type"] == "multi_label_classification":
        raw_dataset, score_dict_metadata = _normalize_score_dict_labels(
            raw_dataset,
            label_column=dataset_cfg["label_column"],
        )
        if score_dict_metadata is not None:
            label_metadata = score_dict_metadata if label_metadata is None else {**label_metadata, **score_dict_metadata}

    if dataset_cfg["label_column"] != "labels" and "labels" not in raw_dataset["train"].column_names:
        raw_dataset = raw_dataset.rename_column(dataset_cfg["label_column"], "labels")

    raw_dataset = _apply_text_perturbation(raw_dataset, dataset_cfg)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenized = tokenize_dataset_dict(
        raw_dataset,
        tokenizer=tokenizer,
        kind=config["tokenization"]["kind"],
        text_columns=tuple(dataset_cfg["text_columns"]),
        max_length=config["tokenization"]["max_length"],
        padding=config["tokenization"].get("padding", "max_length"),
    )
    if config["task_type"] == "multi_label_classification":
        tokenized = _cast_multilabel_labels_to_float(tokenized, label_column="labels")
    expected_meta["label_metadata"] = label_metadata
    save_tokenized_dataset_cache(
        tokenized,
        str(cache_dir),
        meta_path=str(meta_path),
        meta=expected_meta,
        overwrite=True,
    )
    zip_cache_subdir(cache_dir.parent.name)
    return tokenized


def build_tokenized_dataset_cache(config: dict[str, Any]) -> DatasetDict:
    """Build or load the tokenized dataset cache without starting training."""
    return build_and_cache_dataset(config)
