"""Inference from a verified Drive export, without fetching the base model."""

import json
from pathlib import Path

import torch

from transformers import AutoModelForTokenClassification, AutoTokenizer

from .artifacts import verify_bundle
from .predict import decode_all, token_probs


def predict_saved(run_dir, texts, batch_size=None, precision=None):
    run_dir = Path(run_dir)
    verify_bundle(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    decoding = json.loads((run_dir / "decoding.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "model", local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        run_dir / "model", local_files_only=True
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = token_probs(
        model,
        tokenizer,
        texts,
        cfg["max_length"],
        cfg["stride"],
        batch_size=batch_size or cfg["inference_batch_size"],
        precision=precision or cfg["precision"],
    )
    return decode_all(texts, outputs, **decoding), outputs
