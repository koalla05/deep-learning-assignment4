"""Compare Distil Labs synthetic rows with the test-post seeds.

Usage (repo root):  python synthetic/check_synthetic.py synthetic/runs/smoke-1 synthetic/runs/smoke-2
"""

import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "pseudo_label"))
import tools  # noqa: E402

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def shingles(text):
    t = " ".join(text.casefold().split())
    return {t[i : i + 5] for i in range(max(0, len(t) - 4))}


def read_rows(path):
    rows = []
    for line in path.open(encoding="utf-8"):
        messages = json.loads(line)["messages"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        answer = [m["content"] for m in messages if m["role"] == "assistant"][-1]
        rows.append((user, answer))
    return rows


def new_rows(run_dir):
    """Rows of the downloaded train split that are not seeds."""
    seeds = {u for u, _ in read_rows(ROOT / "distil_input" / "train.jsonl")}
    return [(u, a) for u, a in read_rows(Path(run_dir) / "train.jsonl") if u not in seeds]


def profile(rows):
    """Style and label statistics for (text, answer) pairs."""
    texts = [u for u, _ in rows]
    labels, bad_json, not_found = [], 0, 0
    for u, a in rows:
        try:
            places = json.loads(a)
            assert isinstance(places, list)
        except (json.JSONDecodeError, AssertionError):
            bad_json += 1
            continue
        labels.append(len(set(places)))
        not_found += bool(tools.find_spans(u, places)[1])
    lengths = [len(t) for t in texts]
    q = st.quantiles(lengths, n=4)
    sh = [shingles(t) for t in texts]
    dup_rows = {
        x
        for i in range(len(sh))
        for j in range(i)
        if sh[i] and sh[j] and len(sh[i] & sh[j]) / len(sh[i] | sh[j]) >= 0.4
        for x in (i, j)
    }
    return {
        "rows": len(rows),
        "chars p25/p50/p75": f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}",
        "chars max": max(lengths),
        "over 500 chars": f"{sum(n > 500 for n in lengths) / len(rows):.0%}",
        "line breaks": f"{sum(chr(10) in t for t in texts) / len(rows):.0%}",
        "emoji": f"{sum(bool(EMOJI.search(t)) for t in texts) / len(rows):.0%}",
        "russian (ы/э/ъ/ё)": f"{sum(bool(re.search('[ыэъё]', t)) for t in texts) / len(rows):.0%}",
        "no places": f"{sum(n == 0 for n in labels) / len(labels):.0%}",
        "places per post": round(st.mean(labels), 2),
        "bad JSON answers": bad_json,
        "rows with a place not in text": not_found,
        "rows in near-duplicate pairs": len(dup_rows),
    }


def max_overlap_with_test(rows):
    test = [shingles(u) for f in ("train", "test") for u, _ in read_rows(ROOT / "distil_input" / f"{f}.jsonl")]
    out = []
    for u, _ in rows:
        s = shingles(u)
        out.append(max(len(s & t) / min(len(s), len(t)) for t in test if t and s))
    return out


if __name__ == "__main__":
    seeds = read_rows(ROOT / "distil_input" / "train.jsonl") + read_rows(ROOT / "distil_input" / "test.jsonl")
    table = {"test seeds": profile(seeds)}
    for run in sys.argv[1:]:
        rows = new_rows(run)
        table[Path(run).name] = profile(rows)
        overlap = max_overlap_with_test(rows)
        table[Path(run).name]["copies of a test post (>=0.8 overlap)"] = sum(o >= 0.8 for o in overlap)
    keys = list(table["test seeds"]) + ["copies of a test post (>=0.8 overlap)"]
    print(f"{'':38}" + "".join(f"{name:>16}" for name in table))
    for k in keys:
        print(f"{k:38}" + "".join(f"{str(t.get(k, '')):>16}" for t in table.values()))
