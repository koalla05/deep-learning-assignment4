"""Precision and model loading shared by training, preflight and restored inference."""

from contextlib import nullcontext

import torch

from transformers import AutoModelForTokenClassification, AutoTokenizer

from .encoding import LABEL2ID, LABELS


def resolved_precision(precision):
    if precision not in ("auto", "fp32", "fp16", "bf16"):
        raise ValueError(f"unknown precision {precision!r}")
    if precision == "auto":
        if not torch.cuda.is_available():
            return "fp32"
        return "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"
    if precision != "fp32" and not torch.cuda.is_available():
        raise ValueError(f"{precision} requires a CUDA GPU; use fp32 for a CPU check")
    if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("this GPU does not support bf16; use fp16 or fp32")
    return precision


def autocast(device, precision):
    precision = resolved_precision(precision)
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def load_initial_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("NER requires a fast tokenizer with character offsets")
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model,
        revision=cfg.revision,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )
    return model, tokenizer


def configure_model(model, cfg):
    if cfg.freeze_embeddings:
        for name, parameter in model.named_parameters():
            if "embeddings" in name:
                parameter.requires_grad = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
