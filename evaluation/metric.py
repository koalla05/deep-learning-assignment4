"""Competition metric, same code as in Assignment 3 (eda/core/analysis.py)."""

import re


def published_normalize(text: str) -> str:
    """Literal code from the published Kaggle evaluation page."""
    # no f prefix here or on Kaggle, so this strips { } ( ) . and the letters r e s c a p t i n g u o
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\b\w\b\s?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def competition_f1(truth: list[list[str]], predictions: list[list[str]]) -> dict:
    if len(truth) != len(predictions):
        raise ValueError("truth and predictions must have the same number of texts")
    tp = fp = fn = 0
    for actual, predicted in zip(truth, predictions):
        a, p = {published_normalize(x) for x in actual}, {published_normalize(x) for x in predicted}
        tp += len(a & p)
        fp += len(p - a)
        fn += len(a - p)
    precision = tp / (tp + fp) if tp + fp else int(fn == 0)
    recall = tp / (tp + fn) if tp + fn else int(fp == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": round(f1, 8)}
