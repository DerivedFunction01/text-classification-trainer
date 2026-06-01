from __future__ import annotations

# %%
import math
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict
from huggingface_hub import login
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from shared.config import load_json_config, resolve_path
from shared.data_pipeline import build_and_cache_dataset
from shared.paths import HF_TOKEN_PATH

WARMUP_RATIO = 0.1


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
    *, config: dict[str, object], warmup_steps: int, eval_steps: int, save_steps: int
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
        dataloader_num_workers=training["dataloader_num_workers"],
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model=f'{training["metric_prefix"]}_f1',
        greater_is_better=True,
        seed=training["seed"],
        report_to="tensorboard",
        push_to_hub=True,
    )


def main() -> None:
    config = load_config()
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    tokenization_cfg = config["tokenization"]
    training_cfg = config["training"]
    task_type = config["task_type"]

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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"],
        num_labels=model_cfg["num_labels"],
        token=hf_token,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        labels = np.asarray(labels)

        if logits.ndim == 1 or logits.shape[-1] == 1:
            probabilities = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
        else:
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(shifted)
            probabilities = exp_logits[:, 1] / np.sum(exp_logits, axis=-1)

        predictions = (probabilities >= training_cfg["threshold"]).astype(int)
        if task_type != "binary_classification":
            raise ValueError(f"Unsupported task_type for this trainer: {task_type!r}")
        metric_prefix = training_cfg["metric_prefix"]
        return {
            f"{metric_prefix}_f1": f1_score(labels, predictions, zero_division=0),
            f"{metric_prefix}_precision": precision_score(labels, predictions, zero_division=0),
            f"{metric_prefix}_recall": recall_score(labels, predictions, zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=make_training_args(
            config=config,
            warmup_steps=warmup_steps,
            eval_steps=eval_interval,
            save_steps=save_interval,
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("\nEvaluating on test split ...")
    print(trainer.evaluate(ds["test"]))  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], token=hf_token)
    output_dir = resolve_path(model_cfg["output_dir"])
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nModel saved to: {output_dir}")
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
