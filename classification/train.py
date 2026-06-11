from __future__ import annotations

# %%
import math
import argparse
import random
import warnings
from collections.abc import Callable
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from datasets import DatasetDict
from huggingface_hub import login
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    Trainer,
    TrainingArguments,
)

from shared.config import load_json_config, resolve_path
from shared.data_pipeline import build_and_cache_dataset
from shared.paths import HF_TOKEN_PATH

WARMUP_RATIO = 0.1
SUPPORTED_TASK_TYPES = {"classification", "multi_label_classification"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text classification model from JSON config.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the training config JSON file.",
    )
    return parser.parse_args()


def load_config() -> dict[str, object]:
    args = parse_args()
    config_path = (
        Path(args.config)
        if args.config is not None
        else Path(__file__).with_name("config.json")
    )
    working_config_path = config_path.with_name(f".{config_path.name}")

    if config_path.name.startswith("."):
        return load_json_config(config_path)

    if not working_config_path.exists():
        working_config_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Created editable config copy: {working_config_path}")
        raise SystemExit(0)

    return load_json_config(working_config_path)


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return max(1, torch.cuda.device_count())


def get_effective_batch_size(config: dict[str, object]) -> int:
    training = config["training"]
    return (
        training["per_device_train_batch_size"]
        * training["gradient_accumulation_steps"]
        * get_world_size()
    )


def compute_warmup_steps(train_size: int, *, config: dict[str, object]) -> int:
    training = config["training"]
    steps_per_epoch = math.ceil(train_size / get_effective_batch_size(config))
    total_steps = steps_per_epoch * training["num_train_epochs"]
    return max(1, int(total_steps * WARMUP_RATIO))


def compute_step_interval(
    train_size: int, *, config: dict[str, object], checkpoints_per_epoch: int
) -> int:
    steps_per_epoch = math.ceil(train_size / get_effective_batch_size(config))
    return max(1, math.ceil(steps_per_epoch / checkpoints_per_epoch))


def make_training_args(
    *,
    config: dict[str, object],
    warmup_steps: int,
    eval_steps: int,
    save_steps: int,
    metric_for_best_model: str,
) -> TrainingArguments:
    training = config["training"]
    model = config["model"]
    return TrainingArguments(
        output_dir=str(resolve_path(model["output_dir"])),
        num_train_epochs=training["num_train_epochs"],
        learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        logging_strategy="steps",
        logging_steps=training["logging_steps"],
        warmup_steps=warmup_steps,
        weight_decay=training["weight_decay"],
        fp16=training["fp16"],
        bf16=training["bf16"],
        dataloader_num_workers=training["dataloader_num_workers"],
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        seed=training["seed"],
        report_to="tensorboard",
        push_to_hub=training["push_to_hub"],
    )


def get_metric_for_best_model(task_type: str, metric_prefix: str) -> str:
    if task_type == "classification":
        return f"{metric_prefix}_f1"
    if task_type == "multi_label_classification":
        return f"{metric_prefix}_f1_micro"
    raise ValueError(f"Unsupported task_type for this trainer: {task_type!r}")


def get_probability_transform(task_type: str) -> Callable[[np.ndarray], np.ndarray]:
    if task_type == "classification":
        return _single_label_probabilities
    if task_type == "multi_label_classification":
        return _multi_label_probabilities
    raise ValueError(f"Unsupported task_type for this trainer: {task_type!r}")


def _single_label_probabilities(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return 1.0 / (1.0 + np.exp(-logits.reshape(-1)))

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits[:, 1] / np.sum(exp_logits, axis=-1)


def _multi_label_probabilities(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _compute_single_label_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    metric_prefix: str,
) -> dict[str, float]:
    return {
        f"{metric_prefix}_f1": f1_score(labels, predictions, zero_division=0),
        f"{metric_prefix}_precision": precision_score(labels, predictions, zero_division=0),
        f"{metric_prefix}_recall": recall_score(labels, predictions, zero_division=0),
    }


def _compute_multi_label_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    metric_prefix: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    binary_labels = (labels >= threshold).astype(int)
    return {
        f"{metric_prefix}_f1_micro": f1_score(binary_labels, predictions, average="micro", zero_division=0),
        f"{metric_prefix}_f1_macro": f1_score(binary_labels, predictions, average="macro", zero_division=0),
        f"{metric_prefix}_precision_micro": precision_score(
            binary_labels, predictions, average="micro", zero_division=0
        ),
        f"{metric_prefix}_recall_micro": recall_score(
            binary_labels, predictions, average="micro", zero_division=0
        ),
    }


def infer_num_labels_from_dataset(ds: DatasetDict) -> int:
    sample = ds["train"][0]
    labels = sample["labels"]
    if not isinstance(labels, (list, tuple, np.ndarray)):
        raise TypeError(f"Expected multi-label dataset labels to be a sequence, got {type(labels)!r}")
    num_labels = len(labels)
    if num_labels < 2:
        raise ValueError(f"Expected at least 2 labels for multi-label classification, got {num_labels}")
    return num_labels


def load_label_metadata(cache_dir: Path) -> dict[str, object] | None:
    meta_path = cache_dir / "dataset.meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    label_metadata = meta.get("label_metadata")
    if not isinstance(label_metadata, dict):
        return None
    return label_metadata


def make_data_collator(task_type: str):
    if task_type != "multi_label_classification":
        return default_data_collator

    def collate(features):
        labels = torch.stack(
            [torch.as_tensor(feature["labels"], dtype=torch.float32) for feature in features]
        )
        batch = default_data_collator([{k: v for k, v in feature.items() if k != "labels"} for feature in features])
        batch["labels"] = labels
        return batch

    return collate


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    tokenization_cfg = config["tokenization"]
    training_cfg = config["training"]
    task_type = config["task_type"]

    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unsupported task_type: {task_type!r}. Supported values: {sorted(SUPPORTED_TASK_TYPES)}"
        )

    random.seed(training_cfg["seed"])
    np.random.seed(training_cfg["seed"])
    torch.manual_seed(training_cfg["seed"])
    warnings.filterwarnings("ignore", category=UserWarning)

    if not HF_TOKEN_PATH.exists():
        hf_token = None
    else:
        hf_token = HF_TOKEN_PATH.read_text(encoding="utf-8").strip() or None
        if hf_token is not None:
            login(token=hf_token, add_to_git_credential=False)

    ds = build_and_cache_dataset(config)
    label_metadata = load_label_metadata(resolve_path(dataset_cfg["cache_dir"]))

    print(f"  Train: {len(ds['train']):,}")
    print(f"  Val:   {len(ds['val']):,}")
    print(f"  Test:  {len(ds['test']):,}")
    print(f"  GPU count: {get_world_size()}")

    warmup_steps = compute_warmup_steps(len(ds["train"]), config=config)
    eval_interval = compute_step_interval(
        len(ds["train"]), config=config, checkpoints_per_epoch=training_cfg["evals_per_epoch"]
    )
    save_interval = compute_step_interval(
        len(ds["train"]), config=config, checkpoints_per_epoch=training_cfg["saves_per_epoch"]
    )
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Eval interval: {eval_interval}")
    print(f"  Save interval: {save_interval}")

    for split_name in ("train", "val", "test"):
        ds[split_name].set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Loading model: {model_cfg['name']}")
    model_kwargs = {"token": hf_token}
    if task_type == "multi_label_classification" and label_metadata is not None:
        inferred_num_labels = int(label_metadata["num_labels"])
        model_kwargs["num_labels"] = inferred_num_labels
        model_kwargs["id2label"] = dict(label_metadata["id2label"])
        model_kwargs["label2id"] = dict(label_metadata["label2id"])
    elif model_cfg.get("num_labels") is not None:
        model_kwargs["num_labels"] = int(model_cfg["num_labels"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"],
        **model_kwargs,
    )
    if task_type == "multi_label_classification":
        model.config.problem_type = "multi_label_classification"
        inferred_num_labels = int(label_metadata["num_labels"]) if label_metadata is not None else infer_num_labels_from_dataset(ds)
        if label_metadata is not None:
            model.config.label2id = dict(label_metadata["label2id"])
            model.config.id2label = dict(label_metadata["id2label"])
        configured_num_labels = model_cfg.get("num_labels")
        if configured_num_labels is not None and int(configured_num_labels) != inferred_num_labels:
            raise ValueError(
                f"Configured model.num_labels={configured_num_labels} does not match inferred "
                f"dataset label count {inferred_num_labels}"
            )
        model.config.num_labels = inferred_num_labels

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        labels = np.asarray(labels)

        probabilities = get_probability_transform(task_type)(logits)
        threshold = float(training_cfg["threshold"])
        predictions = (probabilities >= threshold).astype(int)
        if task_type == "classification":
            labels = (labels >= threshold).astype(int)
            return _compute_single_label_metrics(labels, predictions, metric_prefix=training_cfg["metric_prefix"])

        labels = (labels >= threshold).astype(int)
        return _compute_multi_label_metrics(
            labels,
            predictions,
            metric_prefix=training_cfg["metric_prefix"],
        )

    trainer = Trainer(
        model=model,
        args=make_training_args(
            config=config,
            warmup_steps=warmup_steps,
            eval_steps=eval_interval,
            save_steps=save_interval,
            metric_for_best_model=get_metric_for_best_model(task_type, training_cfg["metric_prefix"]),
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=make_data_collator(task_type),
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=training_cfg["resume_from_checkpoint"])
    print("\nEvaluating on test split ...")
    print(trainer.evaluate(ds["test"]))  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], token=hf_token)
    output_dir = resolve_path(model_cfg["output_dir"])
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nModel saved to: {output_dir}")
    if training_cfg["push_to_hub"]:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
