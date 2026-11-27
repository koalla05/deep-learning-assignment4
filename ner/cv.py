"""Three fold models of one recipe: CV score (mean over folds) and their averaged test predictions.

Fold k is trained without fold k of the sentences and of the synthetic posts, so the mean of the three
"cv" scores is the cross-validation score, the same protocol as in Assignment 3. On the test posts the
three models are averaged (same tokenizer, so the token offsets match), which is both the submission of
the recipe and a small ensemble.
"""

import json
import pickle
import warnings
from pathlib import Path

import pandas as pd

from .artifacts import publish_bundle, verify_bundle, write_json
from .data import load_test_posts
from .evaluate import eval_sets, score, submission
from .predict import average_probs, decode_all
from .train import load_probs, load_run


def cv_table(out_dir, names):
    """Per-fold F1 on every set, plus mean and standard deviation over folds."""
    rows = {}
    for name in names:
        m = load_run(out_dir, name)["metrics"]
        rows[name] = {k: v["f1"] for k, v in m.items() if isinstance(v, dict)}
    table = pd.DataFrame(rows).T
    table.loc["mean"] = table.mean()
    table.loc["std"] = table.iloc[:-1].std()
    return table


def fold_average(out_dir, names):
    """Averaged token probabilities of the fold models for half A, half B and the gold posts."""
    for name in names:
        verify_bundle(Path(out_dir) / name)
    saved = [load_probs(out_dir, n) for n in names]
    if not saved:
        raise ValueError("at least one ensemble member is required")
    for member in saved[1:]:
        for key in ("half_A", "half_B", "gold26"):
            if (
                member["sets"][key] != saved[0]["sets"][key]
                or member["rows"][key] != saved[0]["rows"][key]
            ):
                raise ValueError(f"ensemble rows differ for {key}")
        if member["labels"] != saved[0]["labels"]:
            raise ValueError("ensemble label order differs")
    current = eval_sets(None)
    for key in ("half_A", "half_B", "gold26"):
        if saved[0]["sets"][key] != current[key][0]:
            raise ValueError(f"current evaluation text order differs for {key}")
        if [row["locations"] for row in saved[0]["rows"][key]] != current[key][1]:
            raise ValueError(f"current evaluation labels differ for {key}")
    return {
        s: average_probs([p["probs"][s] for p in saved]) for s in ("half_A", "half_B", "gold26")
    }, saved[0]["sets"]


def fold_ensemble(out_dir, names, decode_kwargs=None):
    """Scores of the averaged fold models and the Kaggle submission they give."""
    probs, texts = fold_average(out_dir, names)
    return _ensemble_results(probs, texts, decode_kwargs)


def _ensemble_results(probs, texts, decode_kwargs):
    truth = {k: v[1] for k, v in eval_sets(None).items()}
    preds = {s: decode_all(texts[s], probs[s], **(decode_kwargs or {})) for s in probs}
    metrics = {s: score(truth[s], preds[s]) for s in preds}
    test = load_test_posts()
    by_id = {}
    for half in "AB":
        by_id.update(zip(test[test.half == half].text_id, preds[f"half_{half}"]))
    sub = submission(list(test.text_id), [by_id[t] for t in test.text_id])
    return metrics, sub


def export_fold_ensemble(
    out_dir, names, directory, selection, lr_table, decode_kwargs=None, wandb_project=None
):
    """Save the submission, averaged probabilities and all result tables as one verified bundle."""
    from .experiment import export_figures

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    members = {
        "runs": names,
        "manifests": {
            name: json.loads((Path(out_dir) / name / "complete.json").read_text()) for name in names
        },
    }
    decoding = decode_kwargs or {"threshold": 0.5, "aggregation": "token", "merge": "bio"}
    if (directory / "complete.json").exists():
        verify_bundle(directory)
        for filename, expected in (
            ("members.json", members),
            ("selection.json", selection),
            ("decoding.json", decoding),
        ):
            if json.loads((directory / filename).read_text()) != expected:
                raise ValueError("ensemble identity changed; choose a new output directory")
        if (directory / "lr_comparison.csv").read_text() != lr_table.to_csv():
            raise ValueError("learning-rate results changed; choose a new ensemble directory")
        return json.loads((directory / "metrics.json").read_text()), pd.read_csv(
            directory / "submission.csv"
        )
    if (directory / "kaggle").exists():
        raise ValueError("refusing to replace an incomplete ensemble that has Kaggle score records")
    probs, texts = fold_average(out_dir, names)
    metrics, sub = _ensemble_results(probs, texts, decode_kwargs)
    first = load_probs(out_dir, names[0])
    with (directory / "probs.pkl").open("wb") as stream:
        pickle.dump(
            {
                "sets": texts,
                "probs": probs,
                "labels": first["labels"],
                "rows": {key: first["rows"][key] for key in probs},
            },
            stream,
        )
    sub.to_csv(directory / "submission.csv", index=False)
    cv_table(out_dir, names).to_csv(directory / "cv.csv")
    lr_table.to_csv(directory / "lr_comparison.csv")
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "selection.json", selection)
    write_json(
        directory / "members.json",
        members,
    )
    write_json(
        directory / "decoding.json",
        decoding,
    )
    export_figures(directory / "figures")
    publish_bundle(
        directory,
        [
            p
            for p in directory.rglob("*")
            if p.is_file()
            and p.name not in ("complete.json", "wandb.json", "logging.json")
            and "kaggle" not in p.parts
        ],
    )
    if wandb_project:
        from .experiment import log_ensemble

        try:
            log_ensemble(directory, metrics, selection, wandb_project)
            write_json(directory / "logging.json", {"status": "uploaded"})
        except Exception as error:
            write_json(
                directory / "logging.json",
                {"status": "upload_failed", "error_type": type(error).__name__},
            )
            warnings.warn(
                "Drive ensemble is complete; W&B upload failed. See logging.json.", stacklevel=2
            )
    return metrics, sub
