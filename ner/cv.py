"""Three fold models of one recipe: CV score (mean over folds) and their averaged test predictions.

Fold k is trained without fold k of the sentences and of the synthetic posts, so the mean of the three
"cv" scores is the cross-validation score, the same protocol as in Assignment 3. On the test posts the
three models are averaged (same tokenizer, so the token offsets match), which is both the submission of
the recipe and a small ensemble.
"""

import pandas as pd

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
    saved = [load_probs(out_dir, n) for n in names]
    return {s: average_probs([p["probs"][s] for p in saved]) for s in ("half_A", "half_B", "gold26")}, saved[0]["sets"]


def fold_ensemble(out_dir, names, decode_kwargs=None):
    """Scores of the averaged fold models and the Kaggle submission they give."""
    probs, texts = fold_average(out_dir, names)
    truth = {k: v[1] for k, v in eval_sets(None).items()}
    preds = {s: decode_all(texts[s], probs[s], **(decode_kwargs or {})) for s in probs}
    metrics = {s: score(truth[s], preds[s]) for s in preds}
    test = load_test_posts()
    by_id = {}
    for half in "AB":
        by_id.update(zip(test[test.half == half].text_id, preds[f"half_{half}"]))
    sub = submission(list(test.text_id), [by_id[t] for t in test.text_id])
    return metrics, sub
