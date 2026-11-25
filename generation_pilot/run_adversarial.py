"""Exploratory paired XLM-R comparison, with frozen pilot group assignments.

Prepare once before training; run uses immutable example manifests, not labels.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.domain_classifier import (  # noqa: E402 -- allow direct script execution
    ClassifierConfig,
    mask_and_keep_long,
    train_and_score,
)

ARTIFACTS = ROOT / "generation_pilot/artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "auc_050_seed42"
MODEL_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_text(text):
    """Keep the normalization used when the scored manifests were frozen."""
    return " ".join(text.casefold().split())


def _dependency_groups(rows):
    """Connect shared parents/families for source-side cluster bootstrap."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        keys = ["family:" + row["template_family"]] + [
            "parent:" + g for g in row["parent_group_ids"]
        ]
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    return [f"synthetic_cluster:{find(i)}" for i in range(len(rows))]


def load_pools():
    synth_path = ARTIFACTS / "combined_050/candidates.jsonl"
    records = [json.loads(line) for line in synth_path.read_text().splitlines()]
    if len(records) != 50 or any(r["agent_review_status"] != "reviewed" for r in records):
        raise ValueError("Expected the 50 reviewed preview texts.")
    synthetic = pd.DataFrame(
        {
            "example_id": [r["candidate_id"] for r in records],
            "text": [r["text"] for r in records],
            "language": [r["language"] for r in records],
            "domain_pool": [r["domain_pool"] for r in records],
            "group_id": _dependency_groups(records),
            "label": 0,
        }
    )
    manifest_dir = ARTIFACTS / "preview_002"
    target = pd.read_csv(ROOT / "data/test.csv", dtype={"text_id": str})[["text_id", "text"]]
    tm = pd.DataFrame(read_json(manifest_dir / "target_manifest.json"))
    tm["text_id"] = tm.text_id.astype(str)
    target = target.merge(tm, on="text_id", validate="one_to_one").rename(
        columns={"text_id": "example_id"}
    )
    target["label"], target["language"] = 1, "unknown"
    cv = pd.read_parquet(ROOT / "data/cv_rows.parquet")
    # Only explicitly permitted training-side CV rows, not protected fold 0.
    cv = cv[cv.fold.isin([1, 2]) & cv.role_v2.isin(["train", "development"])].copy()
    sm = pd.DataFrame(read_json(manifest_dir / "source_manifest.json"))
    original = cv.merge(
        sm[["source", "row_id", "domain_pool"]], on=["source", "row_id"], validate="one_to_one"
    )
    original["example_id"] = original.source + ":" + original.row_id.astype(str)
    original = original.rename(columns={"source": "language"})
    original["label"] = 0
    # Remove cross-domain exact/normalized overlap before deterministic sampling.
    target_norm = set(target.text.map(_normalized_text))
    original = original.loc[~original.text.map(_normalized_text).isin(target_norm)].copy()
    original = original.drop_duplicates("group_id").drop_duplicates("text")
    columns = ["example_id", "text", "language", "domain_pool", "group_id", "label"]
    return original[columns], synthetic[columns], target[columns]


def _sample_matching(original, synthetic, seed):
    chunks = []
    for (pool, language), group in synthetic.groupby(["domain_pool", "language"], sort=True):
        eligible = original[(original.domain_pool == pool) & (original.language == language)]
        if len(eligible) < len(group):
            raise ValueError(f"Insufficient original rows for {pool}/{language}.")
        chunks.append(eligible.sample(len(group), random_state=seed))
    return pd.concat(chunks, ignore_index=True)


def _check_partition(frame):
    fit, holdout = (frame[frame.domain_pool == p] for p in ["fit", "evaluation"])
    if set(fit.group_id) & set(holdout.group_id):
        raise ValueError("A dependency group crosses fit/evaluation.")
    if set(fit.text.map(_normalized_text)) & set(holdout.text.map(_normalized_text)):
        raise ValueError("Normalized text overlaps fit/evaluation.")
    if set(fit.label) != {0, 1} or set(holdout.label) != {0, 1}:
        raise ValueError("Each partition requires both classes.")


def prepare(output):
    import torch
    from transformers import AutoTokenizer

    if output.exists():
        raise ValueError("Refusing to overwrite a prepared experiment.")
    config = ClassifierConfig(revision=MODEL_REVISION, cache_dir=str(ARTIFACTS / "model_cache"))
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, revision=config.revision, cache_dir=config.cache_dir, local_files_only=True
    )
    original, synthetic, target = load_pools()
    conditions = {}
    for condition in ["raw", "controlled"]:
        source_pool, s, t = original, synthetic, target
        if condition == "controlled":
            source_pool, s, t = [mask_and_keep_long(f, tokenizer) for f in [source_pool, s, t]]
        o = _sample_matching(source_pool, s, config.seed)
        conditions[condition] = {}
        for arm, source in [("original", o), ("synthetic", s)]:
            combined = pd.concat([source, t], ignore_index=True)
            _check_partition(combined)
            conditions[condition][arm] = combined
        # Same target rows and exact per-pool source language counts in both arms.
        assert (
            o.groupby(["domain_pool", "language"]).size().to_dict()
            == s.groupby(["domain_pool", "language"]).size().to_dict()
        )
    protocol = {
        "purpose": "User-requested exploratory AUC on 50 agent-reviewed candidate texts; not primary pilot success assessment.",
        "classifier": asdict(config),
        "conditions": {"raw": 256, "controlled": 32},
        "fits": 4,
        "seeds": [config.seed],
        "bootstrap_samples": 2000,
        "target_sampling": "All eligible frozen target groups, identical across arms; natural class imbalance retained.",
        "source_sampling": "Original source matches every synthetic pool/language count per condition; one row/group.",
        "checkpoint_selection": "Final epoch only; no held-out tuning.",
        "ci": "Percentile 95% cluster bootstrap of fixed held-out predictions; shared target resamples, independent source resamples.",
        "historical_auc_comparison": "Not directly comparable: previous fit/evaluation sizes and random row split differ.",
        "location_labels_used": False,
        "limits": [
            "Only 10 synthetic holdout posts before 32-token filtering.",
            "Pending human text/label and target-near-duplicate review.",
            "Synthetic dependency grouping uses parent/family connected components.",
            "One seed; bootstrap does not include retraining variability.",
            "Historical target exposure and prior aggregate-guided generation are disclosed.",
            "No claim about location-extraction F1.",
        ],
        "input_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                ROOT / "data/cv_rows.parquet",
                ROOT / "data/test.csv",
                ARTIFACTS / "combined_050/candidates.jsonl",
                ARTIFACTS / "preview_002/target_manifest.json",
                ARTIFACTS / "preview_002/source_manifest.json",
            ]
        },
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "accelerate", "numpy", "pandas", "scikit-learn"]
        },
        "python": platform.python_version(),
        "hardware": platform.platform(),
        "mps_available": torch.backends.mps.is_available(),
    }
    output.mkdir(parents=True)
    counts = {}
    for condition, arms in conditions.items():
        counts[condition] = {}
        for arm, frame in arms.items():
            path = output / f"{condition}_{arm}_examples.jsonl"
            frame.to_json(path, orient="records", lines=True, force_ascii=False)
            counts[condition][arm] = {
                "counts": {
                    f"{p}:{int(label)}": int(n)
                    for (p, label), n in frame.groupby(["domain_pool", "label"]).size().items()
                },
                "evaluation_groups": {
                    str(int(label)): int(group.group_id.nunique())
                    for label, group in frame[frame.domain_pool == "evaluation"].groupby("label")
                },
                "sha256": sha256(path),
            }
    protocol["prepared_examples"] = counts
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"output": str(output), "counts": counts}, indent=2), flush=True)


def _bootstrap(predictions, samples, seed):
    rng = np.random.default_rng(seed)
    frames = {arm: pd.DataFrame(rows) for arm, rows in predictions.items()}

    def groups(frame, label):
        return [g.index.to_numpy() for _, g in frame[frame.label == label].groupby("group_id")]

    targets = groups(frames["original"], 1)
    # Target row ordering is identical across the paired arms.
    assert (
        frames["original"].query("label==1").example_id.tolist()
        == frames["synthetic"].query("label==1").example_id.tolist()
    )
    target_indices = {arm: groups(frame, 1) for arm, frame in frames.items()}
    sources = {arm: groups(frame, 0) for arm, frame in frames.items()}
    draws = {"original": [], "synthetic": [], "delta": []}
    for _ in range(samples):
        target_draw = rng.integers(len(targets), size=len(targets))
        scores = {}
        for arm, frame in frames.items():
            source = sources[arm]
            source_draw = rng.integers(len(source), size=len(source))
            indices = np.concatenate(
                [*[source[i] for i in source_draw], *[target_indices[arm][i] for i in target_draw]]
            )
            scores[arm] = float(
                roc_auc_score(frame.label.to_numpy()[indices], frame.score.to_numpy()[indices])
            )
            draws[arm].append(scores[arm])
        draws["delta"].append(scores["original"] - scores["synthetic"])
    return {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in draws.items()}


def run(output):
    from transformers import AutoTokenizer

    protocol = read_json(output / "protocol.json")
    if (output / "results.json").exists():
        raise ValueError("Scores already exist; do not rerun/tune against this holdout.")
    config = ClassifierConfig(**protocol["classifier"])
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, revision=config.revision, cache_dir=config.cache_dir, local_files_only=True
    )
    results = {"protocol_sha256": sha256(output / "protocol.json"), "conditions": {}}
    for condition, max_length in protocol["conditions"].items():
        scores, predictions = {}, {}
        for arm in ["original", "synthetic"]:
            path = output / f"{condition}_{arm}_examples.jsonl"
            if sha256(path) != protocol["prepared_examples"][condition][arm]["sha256"]:
                raise ValueError("Frozen example file changed.")
            frame = pd.read_json(path, lines=True)
            fit = frame[frame.domain_pool == "fit"].reset_index(drop=True)
            holdout = frame[frame.domain_pool == "evaluation"].reset_index(drop=True)
            prediction_path = output / f"{condition}_{arm}_predictions.json"
            if prediction_path.exists():
                saved = read_json(prediction_path)
                scores[arm], predictions[arm] = saved["result"], saved["predictions"]
                continue
            started = time.monotonic()
            print(
                f"START {condition}/{arm}: fit {dict(Counter(fit.label))}, eval {dict(Counter(holdout.label))}",
                flush=True,
            )
            result = train_and_score(
                f"{condition}_{arm}",
                fit,
                holdout,
                max_length,
                tokenizer=tokenizer,
                config=config,
                output_dir=output / "training",
            )
            values = result.pop("scores")
            result["wall_seconds"] = time.monotonic() - started
            predictions[arm] = (
                holdout[["example_id", "label", "group_id"]].assign(score=values).to_dict("records")
            )
            scores[arm] = result
            prediction_path.write_text(
                json.dumps({"result": result, "predictions": predictions[arm]}, indent=2) + "\n"
            )
            print(f"FINISH {condition}/{arm}: AUC {result['auc']:.6f}", flush=True)
        intervals = _bootstrap(predictions, protocol["bootstrap_samples"], config.seed)
        results["conditions"][condition] = {
            "arms": scores,
            "delta": scores["original"]["auc"] - scores["synthetic"]["auc"],
            "ci95": intervals,
        }
        (output / "partial_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.action](args.output)


if __name__ == "__main__":
    main()
