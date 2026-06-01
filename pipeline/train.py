from __future__ import annotations

# %%
import math
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# %%
CONFIG = {
    "model_name": "distilbert/distilbert-base-uncased",
    "output_dir": "./distilbert-classifier",
    "max_length": 512,
    "num_train_epochs": 2,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "evals_per_epoch": 2,
    "saves_per_epoch": 2,
    "logging_steps": 100,
    "weight_decay": 0.01,
    "fp16": True,
    "dataloader_num_workers": 4,
    "seed": 42,
    "threshold": 0.5,
}

WARMUP_RATIO = 0.1

BASE_DIR = Path(".")
HF_TOKEN_PATH = BASE_DIR / "hf_token"
TOKENIZED_CACHE_DIR = BASE_DIR / ".cache" / "xlm_roberta_other" / "safety" / "tokenized"
TOKENIZED_CACHE_META = TOKENIZED_CACHE_DIR / "dataset.meta.json"

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
warnings.filterwarnings("ignore", category=UserWarning)

if not HF_TOKEN_PATH.exists():
    HF_TOKEN = None
else:
    HF_TOKEN = HF_TOKEN_PATH.read_text(encoding="utf-8").strip() or None
    if HF_TOKEN is not None:
        login(token=HF_TOKEN, add_to_git_credential=False)


# %%
def load_cached_dataset(cache_dir: Path) -> DatasetDict:
    split_files = {path.stem: str(path) for path in sorted(cache_dir.glob("*.parquet"))}
    if not split_files:
        raise FileNotFoundError(f"No parquet splits found in {cache_dir}")
    return load_dataset("parquet", data_files=split_files)


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return max(1, torch.cuda.device_count())


def get_effective_batch_size() -> int:
    return (
        CONFIG["per_device_train_batch_size"]
        * CONFIG["gradient_accumulation_steps"]
        * get_world_size()
    )


def compute_warmup_steps(train_size: int) -> int:
    steps_per_epoch = math.ceil(train_size / get_effective_batch_size())
    total_steps = steps_per_epoch * CONFIG["num_train_epochs"]
    return max(1, int(total_steps * WARMUP_RATIO))


def compute_step_interval(train_size: int, *, checkpoints_per_epoch: int) -> int:
    steps_per_epoch = math.ceil(train_size / get_effective_batch_size())
    return max(1, math.ceil(steps_per_epoch / checkpoints_per_epoch))


def make_training_args(
    *, warmup_steps: int, eval_steps: int, save_steps: int
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        logging_strategy="steps",
        logging_steps=CONFIG["logging_steps"],
        warmup_steps=warmup_steps,
        weight_decay=CONFIG["weight_decay"],
        fp16=CONFIG["fp16"],
        dataloader_num_workers=CONFIG["dataloader_num_workers"],
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="binary_f1",
        greater_is_better=True,
        seed=CONFIG["seed"],
        report_to="tensorboard",
        push_to_hub=True,
    )


# %%
if not TOKENIZED_CACHE_META.exists():
    raise RuntimeError("Tokenized safety cache not found. Run the build script first.")

with TOKENIZED_CACHE_META.open(encoding="utf-8") as f:
    meta = json.load(f)

if (
    meta.get("model_name") != CONFIG["model_name"]
    or meta.get("max_length") != CONFIG["max_length"]
):
    raise RuntimeError("Safety cache metadata does not match the current config.")

ds = load_cached_dataset(TOKENIZED_CACHE_DIR)
binary_label2id = {"safe": 0, "unsafe": 1}
binary_id2label = {0: "safe", 1: "unsafe"}

print(f"  Train: {len(ds['train']):,}")
print(f"  Val:   {len(ds['val']):,}")
print(f"  Test:  {len(ds['test']):,}")
print(f"  GPU count: {get_world_size()}")
binary_counts = meta.get("binary_label_counts", {})
num_examples = int(meta.get("num_examples", len(ds["train"])))
if binary_counts:
    print("  Binary distribution:")
    for label, count in sorted(binary_counts.items()):
        rate = meta.get("binary_label_rates", {}).get(label)
        if rate is None:
            print(f"    {label}: {count}")
        else:
            print(f"    {label}: {count} ({rate:.3%})")
warmup_steps = compute_warmup_steps(len(ds["train"]))
eval_interval = compute_step_interval(
    len(ds["train"]), checkpoints_per_epoch=CONFIG["evals_per_epoch"]
)
save_interval = compute_step_interval(
    len(ds["train"]), checkpoints_per_epoch=CONFIG["saves_per_epoch"]
)
print(f"  Warmup steps: {warmup_steps}")
print(f"  Eval interval: {eval_interval}")
print(f"  Save interval: {save_interval}")

for split_name in ("train", "val", "test"):
    ds[split_name].set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )


# %%
print(f"Loading model: {CONFIG['model_name']}")
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=2,
    id2label=binary_id2label,
    label2id=binary_label2id,
    token=HF_TOKEN,
)

def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    predictions = np.asarray(predictions)
    if predictions.ndim != 2 or predictions.shape[1] < 2:
        raise RuntimeError("Expected binary classification logits from Trainer")
    shifted = predictions - predictions.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    binary_probs = probs[:, 1] / probs.sum(axis=1)
    binary_preds = (binary_probs >= CONFIG["threshold"]).astype(int)
    binary_labels = np.asarray(label_ids).astype(int)

    binary_f1 = f1_score(binary_labels, binary_preds, zero_division=0)
    binary_precision = precision_score(binary_labels, binary_preds, zero_division=0)
    binary_recall = recall_score(binary_labels, binary_preds, zero_division=0)
    return {
        "binary_f1": float(binary_f1),
        "binary_precision": float(binary_precision),
        "binary_recall": float(binary_recall),
    }


trainer = Trainer(
    model=model,
    args=make_training_args(
        warmup_steps=warmup_steps,
        eval_steps=eval_interval,
        save_steps=save_interval,
    ),
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    compute_metrics=compute_metrics,
)


# %%
trainer.train()
print("\nEvaluating on test split ...")
print(trainer.evaluate(ds["test"])) # type: ignore
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], token=HF_TOKEN)
trainer.save_model(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"\nBinary-only model saved to: {CONFIG['output_dir']}")
trainer.push_to_hub()
