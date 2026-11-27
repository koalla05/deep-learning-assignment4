"""The same evaluation for every run.

- cv: held-out Kaggle sentences of the validation fold (comparable with Assignment 3 CV).
- synthetic: held-out synthetic posts of the validation fold.
- half_B: test posts never shown to the generator, with the teacher's pseudo-labels. The main score for
  choosing models, since it is the closest to the leaderboard.
- half_A: test posts the generator saw as examples (a gap between A and B means the synthetic data leaks
  seed-specific knowledge). Also the set for tuning thresholds in 07_postprocessing.
- gold26: the 26 Kaggle example posts with human labels. Reported, never tuned on.
"""

import pandas as pd

from evaluation.metric import competition_f1

from .data import load_gold_sample, load_test_posts, span_strings, validation_frames
from .predict import decode_all, token_probs


def eval_sets(val_fold, max_sentences=None):
    """{name: (texts, true place lists)} for every evaluation set available for this fold."""
    sets = {}
    for name, frame in validation_frames(val_fold, max_sentences=max_sentences).items():
        if len(frame):
            sets["cv" if name == "sentences" else name] = (list(frame.text), span_strings(frame))
    test = load_test_posts()
    for half in "BA":
        part = test[test.half == half]
        sets[f"half_{half}"] = (list(part.text), list(part.locations))
    gold = load_gold_sample()
    sets["gold26"] = (list(gold.text), list(gold.locations))
    return sets


def score(truth, predictions):
    m = competition_f1(truth, predictions)
    return {"f1": m["f1"], "precision": round(m["precision"], 4), "recall": round(m["recall"], 4)}


def evaluate_model(model, tokenizer, sets, max_length, stride, batch_size=64, decode_kwargs=None):
    """F1/precision/recall per set and the token probabilities (kept for post-processing and ensembles)."""
    metrics, probs = {}, {}
    for name, (texts, truth) in sets.items():
        probs[name] = token_probs(model, tokenizer, texts, max_length, stride, batch_size)
        metrics[name] = score(truth, decode_all(texts, probs[name], **(decode_kwargs or {})))
    return metrics, probs


def metrics_table(metrics):
    """Flat one-row-per-set table for display."""
    return pd.DataFrame(metrics).T[["f1", "precision", "recall"]]


def submission(test_texts_ids, predictions):
    """Kaggle file: text_id and the Python-list string of places, the format of the sample submission."""
    return pd.DataFrame({"text_id": test_texts_ids, "locations": [str(p) for p in predictions]})
