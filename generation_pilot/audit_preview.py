"""Independent provenance, duplicate and span checks; no protected text export."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from preview_tools import DEFAULT_BATCH, ROOT, load_json, normalized


def shingles(text):
    text = normalized(text)
    return {text[i : i + 5] for i in range(max(0, len(text) - 4))}


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    batch = parser.parse_args().batch
    records = [json.loads(line) for line in (batch / "candidates.jsonl").read_text().splitlines()]
    batch_count = len(records)
    config = load_json(batch / "frozen_config.json")
    if config.get("prior_batch"):
        prior = batch / config["prior_batch"]
        records = [
            json.loads(line) for line in (prior / "candidates.jsonl").read_text().splitlines()
        ] + records
    resources = {r["resource_id"]: r for r in load_json(batch / "surface_bank.json")}
    for filename, expected in config["input_sha256"].items():
        assert (
            hashlib.sha256((ROOT.parent / "data" / filename).read_bytes()).hexdigest() == expected
        )
    cv = pd.read_parquet(ROOT.parent / "data/cv_rows.parquet")
    eligible = cv[cv.fold.isin([1, 2])]
    lookup = {f"{r.source}:{r.row_id}": r for r in eligible.itertuples()}
    for resource in resources.values():
        parent = lookup[resource["parent_id"]]
        a, b = resource["parent_span"]
        assert parent.text[a:b] == resource["surface"]
        assert parent.group_id == resource["parent_group_id"]

    protected = [("source", f"{r.source}:{r.row_id}", r.text) for r in cv.itertuples()]
    for filename in ["test.csv", "labeling_sample.csv"]:
        with (ROOT.parent / "data" / filename).open() as f:
            for i, r in enumerate(csv.DictReader(f)):
                protected.append((filename, r.get("text_id", str(i)), r["text"]))
    protected_norm = {normalized(text) for _, _, text in protected}
    for r in records:
        assert normalized(r["text"]) not in protected_norm, r["candidate_id"]
        previous = 0
        for (a, b), location in zip(r["loc_markers"], r["locations"], strict=True):
            assert 0 <= previous <= a < b <= len(r["text"])
            assert r["text"][a:b] == location
            previous = b
        assert all(lookup[p].fold in [1, 2] for p in r["parent_ids"])
    group_use = Counter(g for r in records for g in set(r["parent_group_ids"]))
    assert max(group_use.values()) <= 3

    # Five-character shingle overlap; common city names alone cannot trigger this.
    # All source rows participate only as exclusion text, never as generator input.
    candidate_sets = [shingles(r["text"]) for r in records]
    candidate_lengths = [len(normalized(r["text"])) for r in records]
    hits = []
    for source, ident, text in protected:
        if len(normalized(text)) < 40:
            continue
        other = shingles(text)
        for r, own, length in zip(records, candidate_sets, candidate_lengths):
            if length < 40 or not other or not own:
                continue
            common = len(own & other)
            if not common:
                continue
            jaccard = common / (len(own) + len(other) - common)
            containment = common / min(len(own), len(other))
            if jaccard >= 0.5 or containment >= 0.8:
                hits.append(
                    {
                        "candidate_id": r["candidate_id"],
                        "source": source,
                        "protected_id": ident,
                        "jaccard": round(jaccard, 4),
                        "containment": round(containment, 4),
                    }
                )
    internal = []
    for i, a in enumerate(candidate_sets):
        for j in range(i):
            b = candidate_sets[j]
            common = len(a & b)
            score = common / len(a | b)
            if score >= 0.35:
                internal.append(
                    {
                        "a": records[i]["candidate_id"],
                        "b": records[j]["candidate_id"],
                        "jaccard": round(score, 4),
                    }
                )
    result = {
        "records": len(records),
        "records_in_current_batch": batch_count,
        "span_integrity": "passed",
        "surface_resource_integrity": "passed",
        "eligible_parent_integrity": "passed",
        "exact_normalized_protected_overlap": 0,
        "max_candidates_per_parent_group": max(group_use.values()),
        "protected_near_duplicate_flags": hits,
        "internal_similarity_flags": internal,
        "near_duplicate_method": "normalized character 5-shingle Jaccard>=.5 or containment>=.8; minimum40characters",
        "limitation": "Heuristics cannot prove absence of semantic paraphrases or missed locations.",
        "tokenizer_offsets": "not yet checked",
        "human_review": "pending",
    }
    (batch / "audit_checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
