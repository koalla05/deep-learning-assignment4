"""Inference: token probabilities for whole texts, then decoding them into place strings.

Prediction is split in two so that thresholds and merging rules can be tuned on saved probabilities
without running the model again (transformers/07_postprocessing.ipynb), and so that several models can be
ensembled by averaging their probabilities.
"""

import numpy as np
import torch

from .data import tools
from .encoding import contained, model_text, windows


@torch.no_grad()
def token_probs(model, tokenizer, texts, max_length=256, stride=64, batch_size=32):
    """For every text: (offsets [n, 2], probabilities [n, 3] for O / B-LOC / I-LOC).

    Long texts are cut into windows overlapping by ``stride`` tokens; a token seen in two windows gets the
    mean of its two probability vectors.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    enc = windows(tokenizer, texts, max_length, stride)
    sample_of = enc["overflow_to_sample_mapping"]
    sums = [dict() for _ in texts]
    order = np.argsort([len(ids) for ids in enc["input_ids"]])  # similar lengths together: less padding
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = tokenizer.pad(
            {"input_ids": [enc["input_ids"][i] for i in idx], "attention_mask": [enc["attention_mask"][i] for i in idx]},
            return_tensors="pt",
        ).to(device)
        probs = torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy()
        for row, i in enumerate(idx):
            acc = sums[sample_of[i]]
            offsets = [tuple(o) for o in enc["offset_mapping"][i]]
            skip = contained(offsets)
            for j, (a, b) in enumerate(offsets):
                if a == b or skip[j]:
                    continue
                total, count = acc.get((a, b), (0.0, 0))
                acc[(a, b)] = (total + probs[row, j], count + 1)
    if was_training:
        model.train()
    out = []
    for acc in sums:
        keys = sorted(acc)
        offsets = np.array(keys, dtype=np.int32).reshape(-1, 2)
        probs = np.array([acc[k][0] / acc[k][1] for k in keys], dtype=np.float32).reshape(-1, 3)
        out.append((offsets, probs))
    return out


def _words(text, offsets):
    """Word index of every token: a token continues the previous word when it touches it and doesn't start
    with whitespace (mDeBERTa offsets include the space or line break before a word)."""
    word, ids = 0, []
    for i, (a, _) in enumerate(offsets):
        if i > 0 and (a > offsets[i - 1][1] or text[a].isspace()):
            word += 1
        ids.append(word)
    return np.array(ids)


def decode(text, offsets, probs, threshold=0.5, aggregation="token", merge="bio"):
    """Place strings from token probabilities.

    threshold: a token is part of a place when P(B) + P(I) >= threshold (0.5 is plain argmax over O/LOC).
    aggregation: "token" decides every token on its own; "first", "mean" or "max" decide once per word
        (tokens without a gap between them), using the first token, the mean or the max of P(place).
        Word-level decisions never cut a word in the middle.
    merge: "bio" starts a new place at a word whose first token is more likely B than I;
        "io" ignores B/I and joins all adjacent place tokens into one place.
    """
    if len(offsets) == 0:
        return []
    p_place = probs[:, 1] + probs[:, 2]
    begins = probs[:, 1] >= probs[:, 2]
    clean = model_text(text)  # what the model read: emoji are spaces there
    words = _words(clean, offsets)
    if aggregation != "token":
        reduce = {"first": lambda x: x[0], "mean": np.mean, "max": np.max}[aggregation]
        word_p = {w: reduce(p_place[words == w]) for w in np.unique(words)}
        p_place = np.array([word_p[w] for w in words])
    is_place = p_place >= threshold

    places, current = [], None
    for i, (a, b) in enumerate(offsets):
        new_word = i == 0 or words[i] != words[i - 1]
        if not is_place[i]:
            if current:
                places.append(current)
            current = None
            continue
        if current is None or (merge == "bio" and new_word and begins[i]):
            if current:
                places.append(current)
            current = [a, b]
        else:
            current[1] = b
    if current:
        places.append(current)

    out = []
    for a, b in places:
        while a < b and clean[a].isspace():
            a += 1
        while b > a and clean[b - 1].isspace():
            b -= 1
        s = tools.strip_edges(text[a:b])
        if s and s not in out:
            out.append(s)
    return out


def decode_all(texts, token_outputs, **kwargs):
    return [decode(t, o, p, **kwargs) for t, (o, p) in zip(texts, token_outputs)]


def average_probs(outputs_per_model):
    """Ensemble: mean probabilities of several models over the same texts (offsets must match)."""
    merged = []
    for per_text in zip(*outputs_per_model):
        offsets = per_text[0][0]
        if not all(np.array_equal(o, offsets) for o, _ in per_text):
            raise ValueError("models tokenize differently; ensemble them at the character level instead")
        merged.append((offsets, np.mean([p for _, p in per_text], axis=0)))
    return merged
