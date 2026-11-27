"""Measure a real model's training and inference memory before the full Colab run."""

import dataclasses
import gc
import math
from pathlib import Path

import torch

from transformers import set_seed

from .artifacts import write_json
from .runtime import autocast, configure_model, load_initial_model, resolved_precision


def memory_check(cfg, output, require_cuda=True):
    """Two full-length optimizer steps and inference batches; experiment weights are not reused."""
    from .train import build_optimizer

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before starting this notebook")
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = optimizer = None
    try:
        model, tokenizer = load_initial_model(cfg)
        configure_model(model, cfg)
        model.to(device).train()
        # Pad a representative Ukrainian sentence to the configured window, including its attention mask.
        encoded = tokenizer("У Києві перекрили вулицю Хрещатик.", add_special_tokens=True)[
            "input_ids"
        ]
        ids = (encoded * math.ceil(cfg.max_length / len(encoded)))[: cfg.max_length]
        inputs = torch.tensor([ids] * cfg.batch_size, device=device)
        mask = torch.ones_like(inputs)
        labels = torch.zeros_like(inputs)
        optimizer = build_optimizer(model, cfg)
        precision = resolved_precision(cfg.precision)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        successful_steps = 0
        # The second step includes activations AND allocated optimizer moments.
        for attempt in range(6):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accumulation):
                with autocast(device, precision):
                    loss = model(input_ids=inputs, attention_mask=mask, labels=labels).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("preflight loss is not finite; try fp32")
                scaler.scale(loss / cfg.grad_accumulation).backward()
            scaler.unscale_(optimizer)
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if not finite:
                if precision == "fp16":
                    scaler.step(optimizer)  # skips the update after non-finite gradients
                    scaler.update()  # lower scale before retrying
                    continue
                raise FloatingPointError("preflight gradients are not finite; try fp32")
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            successful_steps += 1
            if successful_steps == 2:
                break
        if successful_steps != 2:
            raise FloatingPointError(
                "preflight scaler could not stabilize after six attempts; try fp32"
            )
        # Retain optimizer state, as the training-time evaluation callback does.
        model.eval()
        for batch_size in (cfg.eval_batch_size, cfg.inference_batch_size):
            with torch.no_grad(), autocast(device, precision):
                logits = model(
                    input_ids=inputs[:1].repeat(batch_size, 1),
                    attention_mask=mask[:1].repeat(batch_size, 1),
                ).logits
            if not torch.isfinite(logits).all():
                raise FloatingPointError("preflight evaluation is not finite; try fp32")
        report = {
            "config": dataclasses.asdict(cfg),
            "precision": precision,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "parameters": sum(p.numel() for p in model.parameters()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30
            if device.type == "cuda"
            else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30
            if device.type == "cuda"
            else None,
            "total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30
            if device.type == "cuda"
            else None,
            "note": "Two full-window steps; passing is a preflight, not a whole-run memory guarantee.",
        }
        write_json(Path(output), report)
        return report
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            "GPU preflight ran out of memory. Reduce microbatch/eval/inference batches; "
            "increase accumulation to preserve the effective batch."
        ) from error
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
