"""Helpers for pseudo-labeling the test posts: export texts, read labels back, build spans."""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
BATCHES = ROOT / "batches"
LABELS = ROOT / "labels"


def read_posts():
    """Test posts as t<id> and the 26 labeled sample posts as s<row>, texts only."""
    with (DATA / "test.csv").open(encoding="utf-8") as f:
        test = [{"id": f"t{r['text_id']}", "text": r["text"]} for r in csv.DictReader(f)]
    with (DATA / "labeling_sample.csv").open(encoding="utf-8") as f:
        sample = [{"id": f"s{i}", "text": r["text"]} for i, r in enumerate(csv.DictReader(f))]
    return test, sample


def export_batches(n_test_batches=5):
    """Write texts (never labels) to batches/*.jsonl, test split into batches of similar size."""
    test, sample = read_posts()
    BATCHES.mkdir(exist_ok=True)
    batches = {"sample": sample}
    total = sum(len(p["text"]) for p in test)
    current, size, k = [], 0, 1
    for post in test:
        current.append(post)
        size += len(post["text"])
        if size >= total * k / n_test_batches and k < n_test_batches:
            batches[f"test_{k}"], current, k = current, [], k + 1
    batches[f"test_{k}"] = current
    for name, posts in batches.items():
        with (BATCHES / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for p in posts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return {name: len(posts) for name, posts in batches.items()}


def read_labels():
    """All labels/*.json files merged into {id: {"locations": [...], "uncertain": [...]}}."""
    labels = {}
    for path in sorted(LABELS.glob("*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            if item["id"] in labels:
                raise ValueError(f"{item['id']} labeled twice")
            labels[item["id"]] = {
                "locations": item.get("locations", []),
                "uncertain": item.get("uncertain", []),
            }
    return labels


# Consistency pass over the five test batches (see RULES.md, "Adjudication"). The labelers disagreed
# on these strings, so one decision is applied everywhere.
PROMOTE = {
    # street names written without a type word (RULES.md labels streets; three batches were cautious)
    "Межовій", "Жилянській", "Жилянську", "Левка Лукʼяненка, 3", "Дмитровской", "Хрещатику 36",
    "Драйзера", "Червоной Калины", "Цветаевой", "Рейтарської", "Щусєва", "Мілютенка", "Хрещатиком",
    "Бориспільська, 9", "Кіото",
    # countries and cities used as places
    "Великої Британії", "Білорусі", "Moldova", "Краків", "Донецьк", "РФ", "рф", "російської федерації",
    # bare district/region adjectives left from coordinated phrases
    "Дніпровському", "Дарницькому", "Голосіївському", "Подільському", "Деснянському", "Печерському",
    "Шевченківському", "Солом’янському", "Баришівська", "Васильківська", "брянської", "бєлгородської",
}
PLURAL_HEAD = re.compile(r"\s+(районах|районів|областей|областях|областям|громад|громадах|громади)$")
COORDINATION = re.compile(r",\s*|\s+(?:та|і|и|й)\s+")


def split_coordinated(s):
    """'Луганських та Донецьких областей' -> ['Луганських', 'Донецьких'] (the sample labels them bare).

    Single names followed by a plural head ('Оболонському районах') lose the head too.
    """
    if s.startswith("вулиць "):
        s = s[len("вулиць ") :]
    elif not PLURAL_HEAD.search(s):
        return [s]
    s = PLURAL_HEAD.sub("", s)
    return [part for part in COORDINATION.split(s) if part]


def adjudicate(label):
    locations = [p for s in label["locations"] for p in split_coordinated(s)]
    promoted = [s for s in label["uncertain"] if s in PROMOTE]
    return {
        "locations": locations + promoted,
        "uncertain": [s for s in label["uncertain"] if s not in PROMOTE],
    }


def near_duplicate_groups(texts, min_len=40, jaccard=0.5, containment=0.8):
    """Group reposts: same rule as generation_pilot/prepare_preview.py (5-character shingles)."""
    norm = [" ".join(t.casefold().split()) for t in texts]
    shingles = [{t[i : i + 5] for i in range(max(0, len(t) - 4))} for t in norm]
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(shingles):
        for j in range(i):
            b = shingles[j]
            inter = len(a & b)
            jac = inter / len(a | b) if a | b else 0
            cont = inter / min(len(a), len(b)) if a and b else 0
            if norm[i] == norm[j] or (
                min(len(norm[i]), len(norm[j])) >= min_len and (jac >= jaccard or cont >= containment)
            ):
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(texts))]


def find_spans(text, strings):
    """Every occurrence of every string as [start, end), whole words only, without overlaps.

    Longer strings win, so "вулиці Хрещатик" is not also split into "Хрещатик".
    Returns (spans, missing strings).
    """
    taken, spans, missing = [False] * len(text), [], []
    for s in sorted(set(strings), key=len, reverse=True):
        found = False
        for m in re.finditer(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text):
            a, b = m.span()
            if any(taken[a:b]):
                continue
            taken[a:b] = [True] * (b - a)
            spans.append([a, b])
            found = True
        if not found:
            missing.append(s)
    return sorted(spans), missing


# Edge punctuation removed from every label and prediction (the metric doesn't strip it, see README).
STRIP = " .,:;!?\"'«»“”()[]—–-"
PAIRS = {"»": "«", "”": "“", ")": "(", "]": "["}


def strip_edges(p):
    """Strip edge punctuation, but keep a quote or bracket whose partner is inside the string.

    "станції метро «Оболонь»" stays whole (a half-quoted "станції метро «Оболонь" would be a wrong
    label and a span that ends one character early); a stray trailing "»" or "." is still removed, and
    a name wrapped whole ("«Почайна»") loses both quotes, as the gold labels have no quotes.
    """
    for opener, closer in [("«", "»"), ("“", "”"), ('"', '"')]:
        if len(p) > 2 and p[0] == opener and p[-1] == closer and opener not in p[1:-1] and closer not in p[1:-1]:
            p = p[1:-1]
    while p and p[-1] in STRIP:
        c = p[-1]
        if (c in PAIRS and PAIRS[c] in p[:-1]) or (c == '"' and p[:-1].count('"') % 2 == 1):
            break
        p = p[:-1]
    while p and p[0] in STRIP:
        c = p[0]
        closer = {v: k for k, v in PAIRS.items()}.get(c)
        if (closer and closer in p[1:]) or (c == '"' and p[1:].count('"') % 2 == 1):
            break
        p = p[1:]
    return p
