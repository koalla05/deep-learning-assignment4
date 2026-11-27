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
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.optimization import Adafactor

from .data import DataSpec, load_test_posts, training_frame
from .encoding import IGNORE, LABEL2ID, LABELS, encode_frame
from .evaluate import eval_sets, evaluate_model, submission
from .predict import decode_all


@dataclass
class RunConfig:
    name: str
    model: str = "FacebookAI/xlm-roberta-base"
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
    layer_lr_decay: float = 1.0  # < 1: each lower layer gets this factor times the lr of the layer above
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
    precision: str = "auto"  # auto (bf16 if supported, else fp16 on GPU, fp32 elsewhere) | fp16 | bf16 | fp32
    seed: int = 42
    eval_sentences_per_epoch: int = 3000  # CV sentences scored after each epoch
    final_eval_sentences: int | None = None  # CV sentences scored at the end (None = the whole fold)
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
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": 0.0 if no_decay else cfg.weight_decay})
        groups[key]["params"].append(p)
    params = list(groups.values())
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.learning_rate, momentum=0.9, nesterov=True)
    if cfg.optimizer == "adafactor":
        return Adafactor(params, lr=cfg.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


# ---------------------------------------------------------------- trainer pieces


class WeightedTokenTrainer(Trainer):
    """Token cross-entropy with label smoothing and a per-text weight (all 1 unless the data says otherwise)."""

    label_smoothing = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        weight = inputs.pop("weight")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1),
            ignore_index=IGNORE, reduction="none", label_smoothing=self.label_smoothing,
        ).view(labels.shape)
        mask = (labels != IGNORE).float()
        token_weight = mask * weight.view(-1, 1).to(per_token.dtype)
        loss = (per_token * token_weight).sum() / mask.sum().clamp(min=1)
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

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every:
            return
        if self.groups is None:
            self.groups = layer_groups(model)
        sq = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                g = self.groups[name]
                sq[g] = sq.get(g, 0.0) + float(p.grad.detach().float().pow(2).sum())
        row = {"step": state.global_step, **{g: v**0.5 for g, v in sq.items()}}
        self.history.append(row)
        _wandb_log({f"grad_flow/{g}": v for g, v in row.items() if g != "step"}, state.global_step)


class PostEvalCallback(TrainerCallback):
    """After every evaluation: competition F1 on half B, the gold posts and a sample of CV sentences."""

    def __init__(self, tokenizer, sets, cfg):
        self.tokenizer, self.sets, self.cfg, self.history, self.stage = tokenizer, sets, cfg, [], 1

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # the Trainer calls this once per held-out loss set; score the posts only once per evaluation
        if self.history and (self.history[-1]["stage"], self.history[-1]["step"]) == (self.stage, state.global_step):
            return
        metrics, _ = evaluate_model(model, self.tokenizer, self.sets, self.cfg.max_length, self.cfg.stride)
        row = {"stage": self.stage, "step": state.global_step, "epoch": state.epoch,
               **{f"{k}_f1": v["f1"] for k, v in metrics.items()}}
        self.history.append(row)
        _wandb_log({f"eval/{k}": v for k, v in row.items() if k.endswith("_f1")}, state.global_step)


def _wandb_log(values, step):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log({**values, "train/global_step": step})


def _precision_flags(precision):
    if precision == "auto":
        if torch.cuda.is_available():
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            precision = "fp32"
    return {"bf16": precision == "bf16", "fp16": precision == "fp16"}


def _training_args(cfg, out, epochs, lr, report_to, has_eval):
    kwargs = dict(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=cfg.grad_accumulation,
        lr_scheduler_type=cfg.scheduler,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=25,
        eval_strategy="epoch" if has_eval else "no",
        prediction_loss_only=True,
        save_strategy="no",
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
    if "train_sampling_strategy" in params:  # transformers 5
        kwargs["train_sampling_strategy"] = "group_by_length"
    elif "group_by_length" in params:  # transformers 4
        kwargs["group_by_length"] = True
    return TrainingArguments(**kwargs)


# ---------------------------------------------------------------- the run


def run(cfg: RunConfig, out_dir, wandb_project=None):
    """Train, evaluate and save one configuration. Returns the metrics dict."""
    out = Path(out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    report_to = "none"
    if wandb_project:
        import wandb

        wandb.init(project=wandb_project, name=cfg.name, config=dataclasses.asdict(cfg), tags=list(cfg.tags), reinit=True)
        report_to = "wandb"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID
    )
    if cfg.freeze_embeddings:
        for name, p in model.named_parameters():
            if "embeddings" in name:
                p.requires_grad = False

    # evaluation during training: token loss on held-out texts, post-level F1 on a few sets
    quick_sets = eval_sets(cfg.fold, max_sentences=cfg.eval_sentences_per_epoch)
    quick_sets.pop("half_A", None)
    loss_sets = {}
    if cfg.fold is not None:
        from .data import validation_frames

        for name, frame in validation_frames(cfg.fold, max_sentences=cfg.eval_sentences_per_epoch).items():
            loss_sets[name] = encode_frame(frame, tokenizer, cfg.max_length, cfg.stride, cfg.label_all_subtokens)
    grad_flow = GradFlowCallback(cfg.grad_flow_every)
    post_eval = PostEvalCallback(tokenizer, quick_sets, cfg)
    collator = WeightedCollator(tokenizer)

    stages = [(cfg.data, cfg.epochs, cfg.learning_rate)]
    if cfg.stage2_data is not None:
        stages.append((cfg.stage2_data, cfg.stage2_epochs, cfg.stage2_lr or cfg.learning_rate))
    history, started = [], time.time()
    for stage, (spec, epochs, lr) in enumerate(stages, start=1):
        frame = training_frame(spec, cfg.fold, seed=cfg.seed)
        features = encode_frame(frame, tokenizer, cfg.max_length, cfg.stride, cfg.label_all_subtokens)
        print(f"stage {stage}: {spec.describe()} -> {len(frame):,} texts, {len(features):,} windows")
        args = _training_args(cfg, out, epochs, lr, report_to, has_eval=bool(loss_sets))
        trainer = WeightedTokenTrainer(
            model=model, args=args, train_dataset=features, eval_dataset=loss_sets or None,
            data_collator=collator, processing_class=tokenizer,
            optimizers=(build_optimizer(model, dataclasses.replace(cfg, learning_rate=lr)), None),
            callbacks=[grad_flow, post_eval],
        )
        trainer.label_smoothing = cfg.label_smoothing
        post_eval.stage = stage
        trainer.train()
        history += [{**row, "stage": stage} for row in trainer.state.log_history]
    train_minutes = (time.time() - started) / 60

    # final evaluation on the full sets, and the test posts for the submission
    sets = eval_sets(cfg.fold, max_sentences=cfg.final_eval_sentences)
    metrics, probs = evaluate_model(model, tokenizer, sets, cfg.max_length, cfg.stride)
    metrics["train_minutes"] = round(train_minutes, 1)
    test, predicted = load_test_posts(), {}
    for half in "AB":  # half A + half B are all 477 test posts
        part = test[test.half == half]
        predicted.update(zip(part.text_id, decode_all(list(part.text), probs[f"half_{half}"])))
    submission(list(test.text_id), [predicted[t] for t in test.text_id]).to_csv(out / "submission.csv", index=False)

    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps({"trainer": history, "post_eval": post_eval.history}, indent=2))
    (out / "grad_flow.json").write_text(json.dumps(grad_flow.history))
    with (out / "probs.pkl").open("wb") as f:
        small = {k: [(o, p.astype(np.float16)) for o, p in v] for k, v in probs.items()}
        pickle.dump({"model": cfg.model, "sets": {k: v[0] for k, v in sets.items()}, "probs": small}, f)

    if wandb_project:
        import wandb

        wandb.log({f"final/{k}_{m}": v[m] for k, v in metrics.items() if isinstance(v, dict) for m in v})
        wandb.finish()
    del trainer, model
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
        cfg, data=shrink(cfg.data), stage2_data=shrink(cfg.stage2_data), epochs=0.3, stage2_epochs=0.3,
        max_length=min(cfg.max_length, 128), stride=min(cfg.stride, 32),
        eval_sentences_per_epoch=100, final_eval_sentences=200, grad_flow_every=2,
    )


def run_or_load(cfg: RunConfig, out_dir, wandb_project=None):
    """Train unless a finished run with this name is already saved (Colab sessions disconnect)."""
    if os.environ.get("NER_SMOKE"):
        cfg, wandb_project = smoke(cfg), None
    done = Path(out_dir) / cfg.name / "metrics.json"
    if done.exists():
        saved = json.loads((Path(out_dir) / cfg.name / "config.json").read_text())
        if saved != json.loads(json.dumps(dataclasses.asdict(cfg), default=str)):
            raise ValueError(f"{cfg.name} exists with a different config; rename the run or delete its folder")
        print(f"{cfg.name}: loaded saved results")
        return json.loads(done.read_text())
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
