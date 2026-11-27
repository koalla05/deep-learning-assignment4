"""Character spans -> BIO token labels, with long texts split into overlapping windows."""

import unicodedata

import numpy as np

LABELS = ["O", "B-LOC", "I-LOC"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
IGNORE = -100


def model_text(text):
    """The text the model reads: emoji and invisible format characters become spaces.

    The length stays the same, so character offsets still point into the original text. Without this,
    WordPiece (mBERT) turns "📍Феодосія" into a single [UNK], and a variation selector glued to a word
    ends up inside the predicted place.
    """
    return "".join(" " if _hidden(c) else c for c in text)


def _hidden(c):
    # symbols (emoji), format characters (zero-width joiner) and variation selectors (U+FE0F after emoji).
    # Other combining marks stay: Ukrainian stress accents are combining marks too.
    return unicodedata.category(c) in ("So", "Sk", "Cf", "Co", "Cs") or "\ufe00" <= c <= "\ufe0f"


def windows(tokenizer, texts, max_length, stride):
    """Tokenize with offsets; texts longer than ``max_length`` become windows overlapping by ``stride``."""
    return tokenizer(
        [model_text(t) for t in texts],
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )


def char_owner(text_length, spans):
    """For every character, the index of the span it belongs to, or -1."""
    owner = np.full(text_length, -1, dtype=np.int32)
    for k, (a, b) in enumerate(spans):
        owner[a:b] = k
    return owner


def contained(offsets):
    """True for a token whose characters all belong to another token of the window.

    XLM-R's standalone "▁" piece gets the offset of the first letter of the next word, which would
    otherwise look like a one-letter word of its own.
    """
    out = np.zeros(len(offsets), dtype=bool)
    for i, (a, b) in enumerate(offsets):
        for j in (i - 1, i + 1):
            if 0 <= j < len(offsets) and offsets[j] != (a, b) and offsets[j][0] <= a and b <= offsets[j][1] and a < b:
                out[i] = True
    return out


def token_labels(offsets, spans, text_length, word_ids=None, label_all_subtokens=True):
    """BIO label ids for one window.

    A token is part of a place if any of its characters is. It is B when it holds the first character of
    the place, I otherwise. Special tokens get IGNORE. With ``label_all_subtokens=False`` only the first
    piece of every word is trained (the rest get IGNORE), as in the original BERT NER recipe.
    """
    owner = char_owner(text_length, spans)
    offsets = [tuple(o) for o in offsets]
    skip = contained(offsets)
    labels, previous_word = [], None
    for i, (a, b) in enumerate(offsets):
        if a == b or skip[i]:  # special, empty or contained token
            labels.append(IGNORE)
            continue
        word = word_ids[i] if word_ids is not None else None
        if not label_all_subtokens and word is not None and word == previous_word:
            labels.append(IGNORE)
            continue
        previous_word = word
        inside = owner[a:b]
        hit = inside[inside >= 0]
        if len(hit) == 0:
            labels.append(LABEL2ID["O"])
        else:
            start = spans[hit[0]][0]
            labels.append(LABEL2ID["B-LOC"] if a <= start < b else LABEL2ID["I-LOC"])
    return labels


def encode_frame(frame, tokenizer, max_length=256, stride=64, label_all_subtokens=True):
    """Training features (input_ids, attention_mask, labels, weight) for a frame of texts and spans."""
    enc = windows(tokenizer, frame.text, max_length, stride)
    texts, spans, weights = list(frame.text), list(frame.spans), list(frame.weight)
    features = []
    for i, sample in enumerate(enc["overflow_to_sample_mapping"]):
        labels = token_labels(
            enc["offset_mapping"][i], spans[sample], len(texts[sample]),
            word_ids=enc.word_ids(i), label_all_subtokens=label_all_subtokens,
        )
        features.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": labels,
            "weight": float(weights[sample]),
        })
    return features
