"""Notebook setup, experiment identity and durable result tables."""

import dataclasses
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import model_info

from .artifacts import INPUT_FILES, provenance, sha256, write_json
from .data import REPO


def setup(output, project="dl-a4-ner", entity=None):
    """Mount Drive on Colab, check inputs, and authenticate W&B once per notebook."""
    import sys

    if "google.colab" in sys.modules:
        from google.colab import drive

        drive.mount("/content/drive")
        if not torch.cuda.is_available():
            raise RuntimeError("Choose Runtime > Change runtime type > GPU before training")
    for name in INPUT_FILES:
        if not (REPO / name).is_file():
            raise FileNotFoundError(f"Missing repository input: {name}")
    output = Path(output)
    if os.environ.get("NER_SMOKE"):
        output = output.parent / "runs_smoke"
        project = None
    output.mkdir(parents=True, exist_ok=True)
    # Exercise a write now, before any model is loaded.
    probe = output / f".write-check-{os.getpid()}"
    probe.write_text("ok")
    probe.unlink()
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    if project:
        import wandb

        wandb.login()
    print("Artifacts:", output, "| W&B:", project or "disabled")
    return output, project


def freeze_model_revision(model, revision="main"):
    """Resolve a mutable Hub name once; all folds then use the same commit."""
    return model_info(model, revision=revision).sha


def ensemble_directory(out, prefix):
    out = Path(out)
    return out.parent / ("ensembles_smoke" if out.name == "runs_smoke" else "ensembles") / prefix


def experiment_prefix(prefix, recipe):
    config = dataclasses.asdict(recipe)
    for key in ("name", "fold"):
        config.pop(key)
    evidence = provenance()
    config["implementation"] = {key: evidence[key] for key in ("inputs", "source", "packages")}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[
        :12
    ]
    return f"{prefix}-{digest}"


def choose_learning_rate(out, run_name, learning_rates):
    """Select by half-B development F1; ties favor the lower learning rate."""
    from .artifacts import verify_bundle

    rows = []
    for lr in learning_rates:
        directory = Path(out) / run_name(lr, 0)
        verify_bundle(directory)
        metrics = json.loads((directory / "metrics.json").read_text())
        rows.append({"lr": lr, "half_B": metrics["half_B"]["f1"], "run": directory.name})
    if not all(math.isfinite(row["half_B"]) for row in rows):
        raise ValueError("cannot select a learning rate from non-finite scores")
    best = sorted(rows, key=lambda row: (-row["half_B"], row["lr"]))[0]
    return best["lr"], {
        "criterion": "half_B development F1",
        "tie_break": "lower LR",
        "trials": rows,
        "selected": best,
    }


def record_kaggle_result(directory, score, submission_id, split="public"):
    """Record an actual Kaggle result beside its exact submission; never submit automatically."""
    directory = Path(directory)
    if (
        not 0 <= score <= 1
        or not str(submission_id).isdigit()
        or split not in ("public", "private")
    ):
        raise ValueError(
            "supply a score in [0, 1], numeric Kaggle submission ID and public/private split"
        )
    path = directory / "kaggle" / f"{submission_id}-{split}.json"
    if path.exists():
        raise FileExistsError(f"Kaggle result already recorded: {path}")
    write_json(
        path,
        {
            "score": score,
            "submission_id": str(submission_id),
            "split": split,
            "submission_sha256": sha256(directory / "submission.csv"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "bundle": directory.name,
        },
    )


def export_figures(directory):
    """Persist all figures currently displayed by the notebook."""
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for number in plt.get_fignums():
        plt.figure(number).savefig(
            directory / f"figure-{number:02d}.png", dpi=150, bbox_inches="tight"
        )


def log_ensemble(directory, metrics, selection, project):
    """Log ensemble scores and figures to W&B; weights remain in the Drive bundle."""
    import wandb

    directory = Path(directory)
    run = wandb.init(
        project=project,
        name=directory.name + "-ensemble",
        job_type="ensemble",
        config=selection,
        reinit=True,
    )
    write_json(directory / "wandb.json", {"id": run.id, "url": run.url, "project": project})
    run.log(
        {
            f"ensemble/{name}_{metric}": value
            for name, scores in metrics.items()
            for metric, value in scores.items()
        }
    )
    for figure in sorted((directory / "figures").glob("*.png")):
        run.log({f"figures/{figure.stem}": wandb.Image(str(figure))})
    artifact = wandb.Artifact(directory.name, type="evaluation")
    for name in ("metrics.json", "selection.json", "members.json", "cv.csv", "lr_comparison.csv"):
        artifact.add_file(str(directory / name))
    run.log_artifact(artifact)
    run.finish()
