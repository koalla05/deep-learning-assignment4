"""One training run: config -> Hugging Face Trainer (+ W&B) -> metrics, curves and saved probabilities.

Everything an experiment varies is a field of ``RunConfig``, so a notebook cell is just a config and a
call to ``run``. Every run writes into ``<out_dir>/<name>/``:

    config.json     the full config
    metrics.json    F1 / precision / recall per evaluation set (see evaluate.py)
    history.json    the Trainer log (losses, learning rate, grad norm) and per-epoch post-level F1
    grad_flow.json  gradient norm per layer group, every ``grad_flow_every`` steps
    probs.pkl       token probabilities for every evaluation set (for 07_postprocessing and ensembles)
    submission.csv  Kaggle file decoded with the default rule (threshold 0.5, token level, BIO)
"""

import dataclasses
import gc
import inspect
import json
import math
import os
import pickle
import re
import shutil
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.optimization import Adafactor

from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from .artifacts import (
    latest_checkpoint,
    provenance,
    publish_bundle,
    row_metadata,
    verify_bundle,
    write_json,
)
from .data import DataSpec, load_test_posts, training_frame
from .encoding import IGNORE, LABELS, encode_frame
from .evaluate import eval_sets, evaluate_model, submission
from .predict import decode_all
from .runtime import configure_model, load_initial_model, resolved_precision


@dataclass
class RunConfig:
    name: str
    model: str = "FacebookAI/xlm-roberta-base"
    revision: str | None = None  # model/tokenizer commit SHA, resolved before a production run
    data: DataSpec = field(default_factory=DataSpec)
    # optional second stage: continue training the same model on other data (e.g. synthetic only)
    stage2_data: DataSpec | None = None
    stage2_epochs: float = 2
    stage2_lr: float | None = None  # None = same as learning_rate
    fold: int | None = 0  # validation fold; None trains on all folds (final models, no CV score)
    # optimization
    optimizer: str = "adamw"  # adamw | adafactor | sgd
    learning_rate: float = 3e-5
    scheduler: str = "linear"  # any Hugging Face lr_scheduler_type: linear, cosine, constant_with_warmup, polynomial...
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    layer_lr_decay: float = (
        1.0  # < 1: each lower layer gets this factor times the lr of the layer above
    )
    epochs: float = 3
    batch_size: int = 32
    grad_accumulation: int = 1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    freeze_embeddings: bool = False
    # tokenization
    max_length: int = 256
    stride: int = 64
    label_all_subtokens: bool = True
    # infrastructure
    eval_batch_size: int = 16
    inference_batch_size: int = 16
    gradient_checkpointing: bool = False
    checkpoint_steps: int = 250
    precision: str = (
        "auto"  # auto (bf16 if supported, else fp16 on GPU, fp32 elsewhere) | fp16 | bf16 | fp32
    )
    seed: int = 42
    eval_sentences_per_epoch: int = 3000  # CV sentences scored after each epoch
    final_eval_sentences: int | None = (
        None  # CV sentences scored at the end (None = the whole fold)
    )
    grad_flow_every: int = 50
    tags: tuple = ()


# ---------------------------------------------------------------- optimizer


def _layer_id(name, n_layers):
    """0 for embeddings, i + 1 for encoder layer i, n_layers + 1 for everything above (head)."""
    if "embeddings" in name:
        return 0
    m = re.search(r"\.layers?\.(\d+)\.", name)
    return int(m.group(1)) + 1 if m else n_layers + 1


def build_optimizer(model, cfg: RunConfig):
    """Parameter groups with optional layer-wise lr decay; no weight decay on biases and LayerNorm."""
    n_layers = model.config.num_hidden_layers
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        depth = n_layers + 1 - _layer_id(name, n_layers)  # 0 for the head, grows downwards
        lr = cfg.learning_rate * cfg.layer_lr_decay**depth
        no_decay = p.ndim == 1 or "LayerNorm" in name or "layer_norm" in name
        key = (lr, no_decay)
        groups.setdefault(
            key, {"params": [], "lr": lr, "weight_decay": 0.0 if no_decay else cfg.weight_decay}
        )
        groups[key]["params"].append(p)
    params = list(groups.values())
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.learning_rate, momentum=0.9, nesterov=True)
    if cfg.optimizer == "adafactor":
        return Adafactor(
            params,
            lr=cfg.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


# ---------------------------------------------------------------- trainer pieces


class WeightedTokenTrainer(Trainer):
    """Token cross-entropy with label smoothing and a per-text weight (all 1 unless the data says otherwise)."""

    label_smoothing = 0.0

    def log(self, logs, start_time=None):
        # The scaler can skip an overflowing update at a logging step, too. Preserve a missing norm
        # before Trainer copies it into history and forwards it to callbacks.
        norm = logs.get("grad_norm")
        if norm is not None and not math.isfinite(norm):
            logs = {**logs, "grad_norm": None}
        return super().log(logs, start_time)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # from_pretrained applies the architecture's key conversions (e.g. BERT LayerNorm
        # gamma/beta). A raw safetensors load in Transformers 5.0 misses those weights.
        target = model if model is not None else self.model
        restored = type(target).from_pretrained(resume_from_checkpoint, local_files_only=True)
        target.load_state_dict(restored.state_dict(), strict=True)
        del restored

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        weight = inputs.pop("weight")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE,
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).view(labels.shape)
        mask = (labels != IGNORE).float()
        token_weight = mask * weight.view(-1, 1).to(per_token.dtype)
        # Normalize over all valid tokens in the accumulated batch, not each microbatch.
        # This preserves token weighting when memory constraints change the microbatch size.
        denominator = (
            mask.sum()
            if num_items_in_batch is None
            else torch.as_tensor(num_items_in_batch, device=mask.device)
        )
        loss = (per_token * token_weight).sum() / denominator.clamp(min=1)
        return (loss, outputs) if return_outputs else loss


class WeightedCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        weights = torch.tensor([f.pop("weight") for f in features], dtype=torch.float32)
        batch = super().__call__(features)
        batch["weight"] = weights
        for f, w in zip(features, weights.tolist()):  # the Trainer may reuse the feature dicts
            f["weight"] = w
        return batch


def layer_groups(model):
    """Parameter names grouped as embeddings, layer_00..layer_NN and head, for gradient-flow plots."""
    n = model.config.num_hidden_layers
    names = {}
    for name, p in model.named_parameters():
        i = _layer_id(name, n)
        names[name] = "embeddings" if i == 0 else ("head" if i == n + 1 else f"layer_{i - 1:02d}")
    return names


class GradFlowCallback(TrainerCallback):
    """Gradient L2 norm per layer group (after clipping, before the optimizer step)."""

    def __init__(self, every):
        self.every, self.history, self.groups = every, [], None
        self.stage = 1

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every:
            return
        if self.groups is None:
            self.groups = layer_groups(model)
        sq = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                g = self.groups[name]
                sq[g] = sq.get(g, 0.0) + float(
                    torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32).square()
                )
        finite = all(math.isfinite(value) for value in sq.values())
        # AMP may skip an early overflowing update while its scaler settles. Keep that observation
        # explicit without writing NaN/Infinity into JSON or presenting it as a zero gradient.
        row = {
            "step": state.global_step,
            "stage": self.stage,
            "finite": finite,
            **{group: value**0.5 if math.isfinite(value) else None for group, value in sq.items()},
        }
        self.history.append(row)
        _wandb_log(
            {
                f"grad_flow/{g}": v
                for g, v in row.items()
                if g not in ("step", "stage") and v is not None
            },
            state.global_step,
            self.stage,
        )


class PostEvalCallback(TrainerCallback):
    """After every evaluation: competition F1 on half B, the gold posts and a sample of CV sentences."""

    def __init__(self, tokenizer, sets, cfg):
        self.tokenizer, self.sets, self.cfg, self.history, self.stage = tokenizer, sets, cfg, [], 1

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # the Trainer calls this once per held-out loss set; score the posts only once per evaluation
        if self.history and (self.history[-1]["stage"], self.history[-1]["step"]) == (
            self.stage,
            state.global_step,
        ):
            return
        metrics, _ = evaluate_model(
            model,
            self.tokenizer,
            self.sets,
            self.cfg.max_length,
            self.cfg.stride,
            batch_size=self.cfg.inference_batch_size,
            precision=self.cfg.precision,
        )
        row = {
            "stage": self.stage,
            "step": state.global_step,
            "epoch": state.epoch,
            **{f"{k}_f1": v["f1"] for k, v in metrics.items()},
        }
        self.history.append(row)
        _wandb_log(
            {f"eval/{k}": v for k, v in row.items() if k.endswith("_f1")},
            state.global_step,
            self.stage,
        )


class RecoveryHistoryCallback(TrainerCallback):
    """Keep custom curves with the checkpoint that can actually be resumed."""

    def __init__(self, post_eval, grad_flow, post_start, grad_start):
        self.post_eval, self.grad_flow = post_eval, grad_flow
        self.post_start, self.grad_start = post_start, grad_start

    def on_save(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        write_json(
            checkpoint / "observations.json",
            {
                "post_eval": self.post_eval.history[self.post_start :],
                "grad_flow": self.grad_flow.history[self.grad_start :],
            },
        )

        publish_bundle(
            checkpoint,
            [p for p in checkpoint.rglob("*") if p.is_file() and p.name != "complete.json"],
        )

    def on_train_end(self, args, state, control, **kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint.exists():
            self.on_save(args, state, control)


class StageLogCallback(TrainerCallback):
    def __init__(self, stage):
        self.stage = stage

    def on_log(self, args, state, control, logs=None, **kwargs):
        _wandb_log(logs or {}, state.global_step, self.stage)


def _wandb_log(values, step, stage):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log(
            {
                **{f"stage_{stage}/{key}": value for key, value in values.items()},
                f"stage_{stage}/step": step,
            }
        )


def _precision_flags(precision):
    precision = resolved_precision(precision)
    return {"bf16": precision == "bf16", "fp16": precision == "fp16"}


def _training_args(cfg, out, epochs, lr, report_to, has_eval):
    kwargs = dict(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        lr_scheduler_type=cfg.scheduler,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=25,
        eval_strategy="epoch" if has_eval else "no",
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=cfg.checkpoint_steps,
        save_total_limit=2,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        run_name=cfg.name,
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        **_precision_flags(cfg.precision),
    )
    params = inspect.signature(TrainingArguments).parameters
    if "warmup_ratio" in params:  # transformers 4; in 5 a float warmup_steps below 1 is a ratio
        kwargs["warmup_ratio"] = cfg.warmup_ratio
    else:
        kwargs["warmup_steps"] = cfg.warmup_ratio
    return TrainingArguments(**kwargs)


# ---------------------------------------------------------------- the run


def run(cfg: RunConfig, out_dir, wandb_project=None):
    """Train with stage-aware resume, then publish a verified inference bundle on Drive."""
    out = Path(out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(dataclasses.asdict(cfg), default=str))
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"{cfg.name} exists with a different config; use a new run name")
    write_json(config_path, config)
    current = provenance()
    provenance_path = out / "provenance.json"
    previous = {}
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if any(previous[key] != current[key] for key in ("inputs", "source", "packages")):
            raise ValueError("run inputs, source or packages changed; use a new run name")
    else:
        write_json(provenance_path, current)
    precision = resolved_precision(previous.get("precision", cfg.precision))
    set_seed(cfg.seed)
    report_to = "none"
    if wandb_project:
        import wandb

        identity = out / "wandb.json"
        run_id = (
            json.loads(identity.read_text())["id"]
            if identity.exists()
            else wandb.util.generate_id()
        )
        wb = wandb.init(
            project=wandb_project,
            name=cfg.name,
            id=run_id,
            resume="allow",
            config=config,
            tags=list(cfg.tags),
            reinit=True,
        )
        write_json(identity, {"id": wb.id, "url": wb.url, "project": wandb_project})
        for stage in (1, 2):
            wandb.define_metric(f"stage_{stage}/step")
            wandb.define_metric(f"stage_{stage}/*", step_metric=f"stage_{stage}/step")

    model, tokenizer = load_initial_model(cfg)
    configure_model(model, cfg)
    current.update(
        {
            "model_revision": getattr(model.config, "_commit_hash", None) or cfg.revision,
            "precision": precision,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )
    write_json(provenance_path, current)
    quick_sets = dict(eval_sets(cfg.fold, max_sentences=cfg.eval_sentences_per_epoch))
    quick_sets.pop("half_A", None)
    loss_sets = {}
    if cfg.fold is not None:
        from .data import validation_frames

        for name, frame in validation_frames(
            cfg.fold, max_sentences=cfg.eval_sentences_per_epoch
        ).items():
            loss_sets[name] = encode_frame(
                frame, tokenizer, cfg.max_length, cfg.stride, cfg.label_all_subtokens
            )
    grad_flow = GradFlowCallback(cfg.grad_flow_every)
    post_eval = PostEvalCallback(
        tokenizer, quick_sets, dataclasses.replace(cfg, precision=precision)
    )
    collator = WeightedCollator(tokenizer)
    stages = [(cfg.data, cfg.epochs, cfg.learning_rate)]
    if cfg.stage2_data is not None:
        stages.append((cfg.stage2_data, cfg.stage2_epochs, cfg.stage2_lr or cfg.learning_rate))
    history, started = [], time.time()
    for stage, (spec, epochs, lr) in enumerate(stages, start=1):
        stage_dir = out / "work" / f"stage-{stage}"
        if (stage_dir / "complete.json").exists():
            verify_bundle(stage_dir)
            model = AutoModelForTokenClassification.from_pretrained(
                stage_dir / "model", local_files_only=True
            )
            configure_model(model, cfg)
            saved = json.loads((stage_dir / "history.json").read_text())
            history += saved["trainer"]
            post_eval.history += saved["post_eval"]
            grad_flow.history += saved["grad_flow"]
            shutil.rmtree(stage_dir / "checkpoints", ignore_errors=True)
            print(f"stage {stage}: restored completed stage")
            continue
        stage_dir.mkdir(parents=True, exist_ok=True)
        frame = training_frame(spec, cfg.fold, seed=cfg.seed)
        features = encode_frame(
            frame, tokenizer, cfg.max_length, cfg.stride, cfg.label_all_subtokens
        )
        print(
            f"stage {stage}: {spec.describe()} -> {len(frame):,} texts, {len(features):,} windows"
        )
        args = _training_args(
            dataclasses.replace(cfg, precision=precision),
            stage_dir,
            epochs,
            lr,
            report_to,
            has_eval=bool(loss_sets),
        )
        trainer = WeightedTokenTrainer(
            model=model,
            args=args,
            train_dataset=features,
            eval_dataset=loss_sets or None,
            data_collator=collator,
            processing_class=tokenizer,
            optimizers=(build_optimizer(model, dataclasses.replace(cfg, learning_rate=lr)), None),
            callbacks=[grad_flow, post_eval, StageLogCallback(stage)],
        )
        # compute_loss consumes Trainer's valid-token count across the accumulated batch.
        trainer.model_accepts_loss_kwargs = True
        trainer.label_smoothing = cfg.label_smoothing
        post_eval.stage = stage
        grad_flow.stage = stage
        checkpoint_dir = stage_dir / "checkpoints"
        checkpoint = latest_checkpoint(checkpoint_dir)
        post_start, grad_start = len(post_eval.history), len(grad_flow.history)
        if checkpoint and (Path(checkpoint) / "observations.json").exists():
            observed = json.loads((Path(checkpoint) / "observations.json").read_text())
            post_eval.history += observed["post_eval"]
            grad_flow.history += observed["grad_flow"]
        trainer.add_callback(RecoveryHistoryCallback(post_eval, grad_flow, post_start, grad_start))
        trainer.train(resume_from_checkpoint=checkpoint)
        stage_history = [{**row, "stage": stage} for row in trainer.state.log_history]
        history += stage_history
        model.save_pretrained(stage_dir / "model")
        write_json(
            stage_dir / "history.json",
            {
                "trainer": stage_history,
                "post_eval": post_eval.history[post_start:],
                "grad_flow": grad_flow.history[grad_start:],
            },
        )
        publish_bundle(
            stage_dir, [stage_dir / "history.json", *sorted((stage_dir / "model").glob("*"))]
        )
        shutil.rmtree(stage_dir / "checkpoints", ignore_errors=True)
        # Release Adam's moments before allocating the next stage's optimizer or running inference.
        trainer.accelerator.free_memory()
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    train_minutes = (time.time() - started) / 60
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "model")
    sets = eval_sets(cfg.fold, max_sentences=cfg.final_eval_sentences)
    metrics, probs = evaluate_model(
        model,
        tokenizer,
        sets,
        cfg.max_length,
        cfg.stride,
        batch_size=cfg.inference_batch_size,
        precision=precision,
    )
    metrics["train_minutes"] = round(train_minutes, 1)
    test, predicted = load_test_posts(), {}
    for half in "AB":
        part = test[test.half == half]
        predicted.update(zip(part.text_id, decode_all(list(part.text), probs[f"half_{half}"])))
    submission(list(test.text_id), [predicted[t] for t in test.text_id]).to_csv(
        out / "submission.csv", index=False
    )
    write_json(out / "metrics.json", metrics)
    write_json(out / "history.json", {"trainer": history, "post_eval": post_eval.history})
    write_json(out / "grad_flow.json", grad_flow.history)
    rows = row_metadata(sets)
    for half in "AB":
        for row, text_id in zip(rows[f"half_{half}"], test[test.half == half].text_id):
            row["text_id"] = int(text_id)
    write_json(out / "rows.json", rows)
    write_json(out / "decoding.json", {"threshold": 0.5, "aggregation": "token", "merge": "bio"})
    with (out / "probs.pkl").open("wb") as stream:
        small = {k: [(o, p.astype(np.float16)) for o, p in v] for k, v in probs.items()}
        pickle.dump(
            {
                "model": cfg.model,
                "sets": {k: v[0] for k, v in sets.items()},
                "probs": small,
                "rows": rows,
                "labels": LABELS,
            },
            stream,
        )
    files = [
        out / name
        for name in (
            "config.json",
            "provenance.json",
            "metrics.json",
            "history.json",
            "grad_flow.json",
            "rows.json",
            "decoding.json",
            "probs.pkl",
            "submission.csv",
        )
    ] + list((out / "model").glob("*"))
    if wandb_project:
        files.append(out / "wandb.json")
    publish_bundle(out, files)
    if wandb_project:
        import wandb

        try:
            wandb.log(
                {
                    f"final/{k}_{m}": v[m]
                    for k, v in metrics.items()
                    if isinstance(v, dict)
                    for m in v
                }
            )
            wandb.save(str(out / "metrics.json"), base_path=str(out), policy="now")
            wandb.finish()
            write_json(out / "logging.json", {"status": "uploaded"})
        except Exception as error:
            write_json(
                out / "logging.json",
                {"status": "upload_failed", "error_type": type(error).__name__},
            )
            warnings.warn(
                "Drive export is complete; W&B upload failed. See logging.json.", stacklevel=2
            )
    # Checkpoints are only recovery material; final weights and predictions are retained.
    shutil.rmtree(out / "work")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def smoke(cfg: RunConfig):
    """A tiny version of the run: a few hundred texts and a fraction of an epoch.

    Used when the environment variable NER_SMOKE is set, to check a notebook end to end (data, training,
    evaluation, saving, plots) in minutes before spending GPU time. The setup cell of every notebook then
    writes into transformers/runs_smoke instead of the real runs folder.
    """

    def shrink(spec):
        if spec is None:
            return None
        return dataclasses.replace(
            spec,
            sentences=0 if spec.sentences == 0 else 300,
            synthetic="none" if spec.synthetic == "none" else "top25",
            synthetic_repeat=1,
        )

    return dataclasses.replace(
        cfg,
        data=shrink(cfg.data),
        stage2_data=shrink(cfg.stage2_data),
        epochs=0.3,
        stage2_epochs=0.3,
        max_length=min(cfg.max_length, 128),
        stride=min(cfg.stride, 32),
        eval_sentences_per_epoch=100,
        final_eval_sentences=200,
        grad_flow_every=2,
    )


def run_or_load(cfg: RunConfig, out_dir, wandb_project=None):
    """Train unless a finished run with this name is already saved (Colab sessions disconnect)."""
    if os.environ.get("NER_SMOKE"):
        cfg, wandb_project = smoke(cfg), None
    done = Path(out_dir) / cfg.name / "complete.json"
    if done.exists():
        verify_bundle(done.parent)
        saved = json.loads((done.parent / "config.json").read_text())
        if saved != json.loads(json.dumps(dataclasses.asdict(cfg), default=str)):
            raise ValueError(f"{cfg.name} exists with a different config; rename the run")
        previous = json.loads((done.parent / "provenance.json").read_text())
        current = provenance()
        if any(previous[key] != current[key] for key in ("inputs", "source", "packages")):
            raise ValueError(
                "saved run has different inputs, source or packages; use a new run name"
            )
        print(f"{cfg.name}: loaded verified results")
        return json.loads((done.parent / "metrics.json").read_text())

    return run(cfg, out_dir, wandb_project)


def load_run(out_dir, name):
    """Saved results of a run: config, metrics and histories (probabilities only on request)."""
    d = Path(out_dir) / name
    return {
        "config": json.loads((d / "config.json").read_text()),
        "metrics": json.loads((d / "metrics.json").read_text()),
        "history": json.loads((d / "history.json").read_text()),
        "grad_flow": json.loads((d / "grad_flow.json").read_text()),
    }


def load_probs(out_dir, name):
    with (Path(out_dir) / name / "probs.pkl").open("rb") as f:
        return pickle.load(f)
