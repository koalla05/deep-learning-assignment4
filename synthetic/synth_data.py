"""Load a Distil Labs synthetic run as a table, find near-duplicates and clean it for NER training.

Used by synthetic_data.ipynb and synthetic_adversarial.ipynb.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "pseudo_label"))
import tools  # noqa: E402  (find_spans and strip_edges, the same rules as the pseudo-labeled test posts)

UK_LETTERS = "[іїєґІЇЄҐ]"
RU_LETTERS = "[ыэъёЫЭЪЁ]"
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
LINK = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+")
PREPOSITION = re.compile(r"^(?:в|у|на|до|з|із|зі|від|біля|по|под|около|из|от|к)\s", re.IGNORECASE)
STREET_TYPE = re.compile(
    r"^(?:вул|вулиц|просп|пр-т|пров|пл\.|площ|бульв|б-р|наб|узвіз|шосе|ул\.|улиц|переул)", re.IGNORECASE
)


def read_jsonl(path):
    """(user text, assistant answer) pairs of a chat-format jsonl file."""
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            messages = json.loads(line)["messages"]
            user = next(m["content"] for m in messages if m["role"] == "user")
            answer = [m["content"] for m in messages if m["role"] == "assistant"][-1]
            rows.append((user, answer))
    return rows


def parse_answer(answer):
    """JSON list of strings, or None when the teacher's answer is not one."""
    try:
        places = json.loads(answer)
    except json.JSONDecodeError:
        return None
    if not isinstance(places, list) or not all(isinstance(p, str) for p in places):
        return None
    return places


def language(text):
    """'ru' when the text has letters only Russian uses, else 'uk' (the same rule as the adversarial notebook)."""
    return "ru" if re.search(RU_LETTERS, text) and not re.search(UK_LETTERS, text) else "uk"


def load_run(run_dir):
    """Generated rows of a run (the seed posts the run copies into train.jsonl are removed)."""
    seeds = {text for text, _ in read_jsonl(ROOT / "distil_input" / "train.jsonl")}
    rows = [(t, a) for t, a in read_jsonl(Path(run_dir) / "train.jsonl") if t not in seeds]
    frame = pd.DataFrame(rows, columns=["text", "answer"])
    frame.insert(0, "syn_id", np.arange(len(frame)))
    frame["locations"] = frame["answer"].map(parse_answer)
    return frame


def style(frame):
    """Per-text format features shared by the synthetic rows and the test posts."""
    text = frame["text"]
    return pd.DataFrame(
        {
            "chars": text.str.len(),
            "lines": text.str.count("\n") + 1,
            "has_line_break": text.str.contains("\n"),
            "has_emoji": text.map(lambda t: bool(EMOJI.search(t))),
            "has_link": text.map(lambda t: bool(LINK.search(t))),
            "has_hashtag": text.str.contains(r"#\w", regex=True),
            "lang": text.map(language),
        },
        index=frame.index,
    )


# ---------------------------------------------------------------- near-duplicates


def _normalize(text):
    return " ".join(text.casefold().split())


def _shingle_matrix(texts, vectorizer=None):
    """Binary 5-character shingle matrix, the same shingles as tools.near_duplicate_groups."""
    analyzer = lambda t: {t[i : i + 5] for i in range(max(0, len(t) - 4))}  # noqa: E731
    norm = [_normalize(t) for t in texts]
    if vectorizer is None:
        vectorizer = CountVectorizer(analyzer=analyzer, binary=True, dtype=np.int32).fit(norm)
    return vectorizer.transform(norm).tocsr(), vectorizer


def _overlaps(a, b, chunk=500):
    """Yield (row slice, intersection, |a|, |b|) of shingle counts, chunked to keep memory small."""
    size_a, size_b = np.asarray(a.sum(1)).ravel(), np.asarray(b.sum(1)).ravel()
    for start in range(0, a.shape[0], chunk):
        inter = (a[start : start + chunk] @ b.T).toarray()
        yield slice(start, start + chunk), inter, size_a[start : start + chunk, None], size_b[None, :]


def near_duplicate_groups(texts, min_len=40, jaccard=0.5, containment=0.8):
    """Group ids with the rule of tools.near_duplicate_groups, vectorized for thousands of texts."""
    x, _ = _shingle_matrix(texts)
    norm = [_normalize(t) for t in texts]
    long_enough = np.array([len(t) >= min_len for t in norm])
    exact = pd.Series(norm).groupby(norm).ngroup().to_numpy()
    edges = []
    for rows, inter, sa, sb in _overlaps(x, x):
        union = sa + sb - inter
        jac = np.divide(inter, union, out=np.zeros(inter.shape), where=union > 0)
        cont = np.divide(inter, np.minimum(sa, sb), out=np.zeros(inter.shape), where=np.minimum(sa, sb) > 0)
        i_idx = np.arange(inter.shape[0]) + rows.start
        both_long = long_enough[i_idx][:, None] & long_enough[None, :]
        same = (exact[i_idx][:, None] == exact[None, :]) | (both_long & ((jac >= jaccard) | (cont >= containment)))
        i, j = np.nonzero(same)
        edges.append((i + rows.start, j))
    i = np.concatenate([e[0] for e in edges])
    j = np.concatenate([e[1] for e in edges])
    graph = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(len(texts), len(texts)))
    _, labels = sparse.csgraph.connected_components(graph, directed=False)
    return labels


def max_copied_share(texts, reference):
    """For every text, the largest share of its own 5-character pieces found in one reference text (0..1).

    Dividing by the text's own pieces (not by the smaller of the two, as near_duplicate_groups does) matters
    here: a test post of three words is fully contained in many long synthetic posts, but that doesn't make
    them copies.
    """
    x, _ = _shingle_matrix(list(texts) + list(reference))
    x, ref = x[: len(texts)], x[len(texts) :]
    best = np.zeros(len(texts))
    for rows, inter, sa, _ in _overlaps(x, ref):
        best[rows] = (inter / np.maximum(sa, 1)).max(1)
    return best


def repeats_earlier(texts, jaccard=0.5):
    """True for a text whose pieces overlap an earlier text with Jaccard >= threshold (or equal it).

    Unlike the groups, this doesn't chain: a post is dropped only for its own near-twin.
    """
    x, _ = _shingle_matrix(texts)
    out = np.zeros(len(texts), dtype=bool)
    for rows, inter, sa, sb in _overlaps(x, x):
        jac = inter / np.maximum(sa + sb - inter, 1)
        i_idx = np.arange(inter.shape[0]) + rows.start
        earlier = np.arange(len(texts))[None, :] < i_idx[:, None]
        out[rows] = ((jac >= jaccard) & earlier).any(1)
    return out


# ---------------------------------------------------------------- cleaning


def tidy_locations(places):
    """Strip edge punctuation (as before every submission) and drop empty or repeated strings."""
    out = []
    for p in places:
        p = tools.strip_edges(p)
        if p and p not in out:
            out.append(p)
    return out


def add_spans(frame):
    """Character spans of every label string and the strings that never occur in the text."""
    found = [tools.find_spans(t, locs) if locs is not None else ([], None) for t, locs in zip(frame.text, frame.locations)]
    return frame.assign(loc_markers=[s for s, _ in found], missing=[m for _, m in found])


def drop_reason(frame, copied_share, copy_threshold=0.8):
    """First reason to drop each row, or '' to keep it. Expects a ``repeats_earlier`` column."""
    reasons = pd.Series("", index=frame.index)
    rules = [
        ("answer is not a JSON list", frame.locations.isna()),
        ("place not found in text", frame.missing.map(lambda m: bool(m))),
        ("near-copy of a test post", pd.Series(copied_share >= copy_threshold, index=frame.index)),
        ("near-duplicate of an earlier synthetic post", frame.repeats_earlier),
    ]
    for name, mask in rules:
        reasons[(reasons == "") & mask] = name
    return reasons
