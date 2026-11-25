"""Shared XLM-R training and preprocessing for notebook and pilot callers."""

import gc
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class ClassifierConfig:
    model: str = "FacebookAI/xlm-roberta-base"
    revision: str | None = None
    seed: int = 42
    learning_rate: float = 2e-5
    epochs: int = 2
    batch_size: int = 16
    eval_batch_size: int = 16
    cache_dir: str | None = None
    optimizer: str = "adamw_torch"


def mask_cues(text):
    """Preserve the previous notebook's literal masking behavior."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub("[\U0001f300-\U0001faff\u2600-\u27bf]", " ", text)
    return text.replace("\n", " ").replace("\r", " ")


def mask_and_keep_long(frame, tokenizer, min_tokens=32):
    masked = frame.assign(text=frame.text.map(mask_cues))
    lengths = np.array(
        [len(ids) for ids in tokenizer(list(masked.text), truncation=False)["input_ids"]]
    )
    return masked.loc[lengths >= min_tokens].copy()


def _encode(frame, tokenizer, max_length):
    encoded = tokenizer(
        list(frame.text), truncation=True, max_length=max_length, padding="max_length"
    )
    return [
        dict(input_ids=ids, attention_mask=mask, labels=int(label))
        for ids, mask, label in zip(encoded["input_ids"], encoded["attention_mask"], frame.label)
    ]


def train_and_score(
    name, fit, holdout, max_length, *, tokenizer, config=None, output_dir="/tmp/adversarial"
):
    """Fit one fresh classifier, then return AUC and aligned held-out scores.

    Inputs require only ``text`` and binary ``label`` columns. Evaluation data is
    never used for checkpoint selection. AUC always orients target posts as 1.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config = config or ClassifierConfig()
    if set(fit.label) != {0, 1} or set(holdout.label) != {0, 1}:
        raise ValueError("Both fit and evaluation must contain both classes.")
    if max_length < 2:
        raise ValueError("max_length must accommodate special tokens.")
    set_seed(config.seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model, revision=config.revision, cache_dir=config.cache_dir, num_labels=2
    )
    args = TrainingArguments(
        output_dir=str(Path(output_dir) / name),
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        logging_steps=10,
        save_strategy="no",
        seed=config.seed,
        data_seed=config.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        disable_tqdm=True,
        optim=config.optimizer,
    )
    trainer = Trainer(model=model, args=args, train_dataset=_encode(fit, tokenizer, max_length))
    trainer.train()
    prediction = trainer.predict(_encode(holdout, tokenizer, max_length))
    scores = prediction.predictions[:, 1] - prediction.predictions[:, 0]
    auc = float(roc_auc_score(np.asarray(holdout.label), scores))
    result = {
        "auc": auc,
        "scores": scores.tolist(),
        "metrics": prediction.metrics,
        "device": str(args.device),
    }
    del prediction, trainer, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
