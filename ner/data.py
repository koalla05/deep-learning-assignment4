"""Labeled texts for NER: Kaggle training sentences, synthetic posts and the evaluation posts.

Every text comes as a row with ``text`` and ``spans`` (character [start, end) pairs of the places), so all
sources go through the same tokenization. Folds: the sentences keep the 3 folds of Assignment 3, and the
synthetic posts get 3 folds by repost group, so the fold-k model never sees fold k of either source.
"""

import ast
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pseudo_label"))
import tools  # noqa: E402  (find_spans, strip_edges)

N_FOLDS = 3
SEED = 42
COLUMNS = ["text", "spans", "fold", "group", "lang", "kind", "weight"]


@cache
def load_sentences():
    """The 63,388 Kaggle training sentences with their Assignment 3 folds."""
    cv = pd.read_parquet(REPO / "data" / "cv_rows.parquet")
    return pd.DataFrame({
        "text": cv.text,
        "spans": cv.loc_markers.map(lambda s: [list(x) for x in ast.literal_eval(s)]),
        "fold": cv.fold,
        "group": "cv:" + cv.group_id,
        "lang": cv.source,
        "kind": "sentence",
        "weight": 1.0,
    })


@cache
def load_synthetic():
    """Clean synthetic posts (synthetic_data.ipynb) with their adversarial test-likeness percentile."""
    out = REPO / "synthetic" / "output"
    syn = pd.read_parquet(out / "full1_clean.parquet").merge(
        pd.read_parquet(out / "full1_adversarial.parquet")[["syn_id", "test_like"]], on="syn_id", validate="1:1"
    )
    groups = np.sort(syn.duplicate_group.unique())
    order = np.random.default_rng(SEED).permutation(len(groups))
    fold_of = dict(zip(groups[order], np.arange(len(groups)) % N_FOLDS))
    return pd.DataFrame({
        "text": syn.text,
        "spans": syn.loc_markers.map(lambda m: [[int(a), int(b)] for a, b in m]),
        "fold": syn.duplicate_group.map(fold_of),
        "group": "syn:" + syn.duplicate_group.astype(str),
        "lang": syn.lang,
        "kind": "synthetic",
        "weight": 1.0,
        "test_like": syn.test_like,
    })


@cache
def load_test_posts():
    """All 477 test posts with the teacher's pseudo-labels and seed half (A was shown to the generator)."""
    p = pd.read_parquet(REPO / "pseudo_label" / "output" / "test_pseudo.parquet")
    return pd.DataFrame({
        "text_id": p.text_id,
        "text": p.text,
        "locations": p.locations.map(list),
        "half": p.seed_half,
    })


@cache
def load_gold_sample():
    """The 26 labeled example posts from Kaggle (the only human labels on posts)."""
    s = pd.read_csv(REPO / "data" / "labeling_sample.csv")
    return pd.DataFrame({"text": s.text, "locations": s.locations.map(ast.literal_eval)})


def span_strings(frame):
    """Place strings of each row, as the metric sees them."""
    return [[text[a:b] for a, b in spans] for text, spans in zip(frame.text, frame.spans)]


@dataclass(frozen=True)
class DataSpec:
    """Which training texts to use.

    sentences: number of Kaggle sentences sampled from the training folds (None = all of them, 0 = none).
    synthetic: "none", "all", or "top75" / "top50" / "top25" (keep the most test-like share by the
        adversarial score).
    synthetic_repeat: how many times every synthetic post appears per epoch (oversampling).
    weight_by_test_like: weight each synthetic post's loss by its test-likeness (mean weight 1).
    """

    sentences: int | None = None
    synthetic: str = "all"
    synthetic_repeat: int = 1
    weight_by_test_like: bool = False

    def describe(self):
        s = "all" if self.sentences is None else f"{self.sentences:,}"
        syn = self.synthetic + (f"x{self.synthetic_repeat}" if self.synthetic_repeat > 1 else "")
        return f"sentences={s}, synthetic={syn}" + (", weighted" if self.weight_by_test_like else "")


def training_frame(spec: DataSpec, val_fold: int | None, seed: int = SEED):
    """Training rows: every fold except ``val_fold`` (all folds when it is None)."""
    parts = []
    if spec.sentences != 0:
        s = load_sentences()
        s = s[s.fold != val_fold]
        if spec.sentences is not None:
            s = s.sample(min(spec.sentences, len(s)), random_state=seed)
        parts.append(s[COLUMNS])
    if spec.synthetic != "none":
        syn = load_synthetic()
        syn = syn[syn.fold != val_fold]
        if spec.synthetic.startswith("top"):
            keep = int(spec.synthetic[3:]) / 100
            syn = syn[syn.test_like > 1 - keep]
        elif spec.synthetic != "all":
            raise ValueError(f"unknown synthetic option {spec.synthetic!r}")
        if spec.weight_by_test_like:
            syn = syn.assign(weight=syn.test_like / syn.test_like.mean())
        parts += [syn[COLUMNS]] * spec.synthetic_repeat
    if not parts:
        raise ValueError("empty training data")
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def validation_frames(val_fold: int | None, max_sentences: int | None = None, seed: int = SEED):
    """Held-out sentences and synthetic posts of ``val_fold`` (empty frames when it is None)."""
    if val_fold is None:
        empty = pd.DataFrame(columns=COLUMNS)
        return {"sentences": empty, "synthetic": empty}
    s = load_sentences()
    s = s[s.fold == val_fold]
    if max_sentences is not None and len(s) > max_sentences:
        s = s.sample(max_sentences, random_state=seed)
    syn = load_synthetic()
    return {"sentences": s[COLUMNS].reset_index(drop=True), "synthetic": syn[syn.fold == val_fold][COLUMNS].reset_index(drop=True)}
