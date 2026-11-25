"""Assemble model-authored drafts and export pending review records.

No generation API, network access, model training, or automatic human acceptance.
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

SLOT = re.compile(r"<LOC_[1-9][0-9]*>")
ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH = ROOT / "artifacts" / "preview_001"


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def assemble(draft, slots, resources):
    """Append segments and location insertions, recording Unicode code points."""
    mapping = {}
    for slot in slots:
        key = slot["placeholder"]
        if not SLOT.fullmatch(key) or key in mapping:
            raise ValueError(f"Invalid or duplicate slot specification: {key}")
        mapping[key] = resources[slot["resource_id"]]
    matches = list(SLOT.finditer(draft))
    if Counter(m.group() for m in matches) != Counter({k: 1 for k in mapping}):
        raise ValueError("Every declared slot must occur exactly once.")
    chunks, spans, edit_map, location_ids = [], [], [], []
    cursor = length = 0
    for match in matches:
        literal = draft[cursor : match.start()]
        chunks.append(literal)
        edit_map.append(
            {
                "kind": "literal",
                "draft_span": [cursor, match.start()],
                "final_span": [length, length + len(literal)],
            }
        )
        length += len(literal)
        resource = mapping[match.group()]
        surface = resource["surface"]
        chunks.append(surface)
        spans.append([length, length + len(surface)])
        location_ids.append(resource["resource_id"])
        edit_map.append(
            {
                "kind": "location",
                "placeholder": match.group(),
                "resource_id": resource["resource_id"],
                "draft_span": [match.start(), match.end()],
                "final_span": spans[-1],
            }
        )
        length += len(surface)
        cursor = match.end()
    chunks.append(draft[cursor:])
    edit_map.append(
        {
            "kind": "literal",
            "draft_span": [cursor, len(draft)],
            "final_span": [length, length + len(draft[cursor:])],
        }
    )
    text = "".join(chunks)
    if "<" in text or ">" in text:
        raise ValueError("Unresolved or unsupported placeholder/tag.")
    locations = [text[a:b] for a, b in spans]
    assert locations == [resources[key]["surface"] for key in location_ids]
    return text, spans, locations, edit_map


def length_bucket(text):
    n = len(text)
    return (
        "<80"
        if n < 80
        else "80-220"
        if n <= 220
        else "221-500"
        if n <= 500
        else "501-1100"
        if n <= 1100
        else "1101-3700"
    )


def load_json(path):
    return json.loads(path.read_text())


def build(batch, apply_revision=False, apply_editorial=False):
    if apply_revision and apply_editorial:
        raise ValueError("Choose either a recipe revision or editorial corrections.")
    if (batch / "candidates.jsonl").exists() and not (apply_revision or apply_editorial):
        raise ValueError(
            "Candidates already exist; preserve reviewed records rather than overwriting."
        )
    config = load_json(batch / "frozen_config.json")
    resources = {r["resource_id"]: r for r in load_json(batch / "surface_bank.json")}
    parents = {r["parent_id"]: r for r in load_json(batch / "source_manifest.json")}
    draft_files = [
        batch / name
        for name in config.get(
            "draft_files", ["terra_fit_drafts.json", "terra_evaluation_drafts.json"]
        )
    ]
    drafts = [d for p in draft_files for d in load_json(p)]
    if len(drafts) != config["candidate_budget"]:
        raise ValueError("Draft count differs from frozen preview budget.")
    expected_ids = set(config.get("candidate_ids", [f"preview_{i:03}" for i in range(1, 21)]))
    if len(expected_ids) != len(drafts) or {d["candidate_id"] for d in drafts} != expected_ids:
        raise ValueError("Unexpected or duplicate candidate IDs.")
    prior_records = []
    if config.get("prior_batch"):
        prior = batch / config["prior_batch"]
        for name, digest in load_json(batch / "prior_batch_fingerprints.json").items():
            if hashlib.sha256((prior / name).read_bytes()).hexdigest() != digest:
                raise ValueError("A prior frozen batch has changed.")
        prior_records = [
            json.loads(line) for line in (prior / "candidates.jsonl").read_text().splitlines()
        ]
    requests = (
        {r["candidate_id"]: r for r in load_json(batch / "generation_requests.json")}
        if (batch / "generation_requests.json").exists()
        else {}
    )
    revision_files = sorted(batch.glob("terra_*_revision_1.json")) if apply_revision else []
    if apply_editorial:
        revision_files += [batch / "terra_editorial_corrections.json"]
    revisions = [d for p in revision_files for d in load_json(p)]
    if len({r["candidate_id"] for r in revisions}) != len(revisions):
        raise ValueError("Duplicate revision IDs.")
    originals = {d["candidate_id"]: d for d in drafts}
    for revision in revisions:
        prior = originals[revision["candidate_id"]]
        if any(revision[k] != prior[k] for k in ["language", "domain_pool", "genre"]):
            raise ValueError("Revision changed frozen candidate allocation.")
        originals[revision["candidate_id"]] = revision
    drafts = list(originals.values())
    records = []
    seen = {normalized(r["text"]) for r in prior_records}
    group_counts = Counter(g for r in prior_records for g in set(r["parent_group_ids"]))
    for d in drafts:
        if requests:
            request = requests[d["candidate_id"]]
            for field in ["language", "domain_pool", "genre", "target_length_bucket"]:
                if request[field] != d[field]:
                    raise ValueError(f"Frozen request mismatch: {d['candidate_id']} {field}")
            if [s["resource_id"] for s in d["slots"]] != request["resource_ids"]:
                raise ValueError("Draft changed frozen resource allocation.")
        pool = d["domain_pool"]
        if d["template_family"] not in config["families"][pool]:
            raise ValueError("Family is not assigned to this domain pool.")
        selected = [resources[s["resource_id"]] for s in d["slots"]]
        if any(r["domain_pool"] != pool or r["language"] != d["language"] for r in selected):
            raise ValueError("Surface-resource pool or language mismatch.")
        for r in selected:
            parent = parents[r["parent_id"]]
            if parent["fold"] not in [1, 2] or parent["domain_pool"] != pool:
                raise ValueError("Protected or wrong-pool parent.")
        text, spans, locations, edits = assemble(d["draft"], d["slots"], resources)
        if d["intended_mentions"] != len(spans):
            raise ValueError(f"Mention count mismatch: {d['candidate_id']}")
        if not text.strip() or normalized(text) in seen:
            raise ValueError("Empty or duplicate text.")
        seen.add(normalized(text))
        groups = sorted({r["parent_group_id"] for r in selected})
        group_counts.update(groups)
        records.append(
            {
                **d,
                "text": text,
                "loc_markers": spans,
                "locations": locations,
                "parent_ids": sorted({r["parent_id"] for r in selected}),
                "parent_group_ids": groups,
                "allowed_cv_fold": 0,
                "surface_form_resource": [r["resource_id"] for r in selected],
                "stream": config["preview_stream"],
                "recipe_version": config["recipe_version"],
                "policy_version": config["policy_version"],
                "generator_version": config["generator"],
                "reasoning_effort": config["reasoning_effort"],
                "seed": config["seed"],
                "seed_note": "Seed governs manifests, not subagent sampling.",
                "review_status": "pending",
                "reviewer": "",
                "review_notes": "",
                "rejection_reason": "",
                "agent_review_status": "pending",
                "agent_review_notes": "",
                "length_characters": len(text),
                "actual_length_bucket": length_bucket(text),
                "assembly_map": edits,
            }
        )
    if max(group_counts.values(), default=0) > 3:
        raise ValueError("More than three candidates use one parent group.")
    records.sort(key=lambda r: r["candidate_id"])
    if apply_revision or apply_editorial:
        if not revisions:
            raise ValueError("No revision drafts exist.")
        if (batch / "candidates_round0.jsonl").exists():
            raise ValueError("A correction pass has already been applied to this batch.")
        for name in [
            "candidates.jsonl",
            "review.csv",
            "REVIEW.md",
            "attempt_ledger.json",
            "diagnostics.json",
        ]:
            path = batch / name
            if path.exists():
                path.rename(path.with_name(path.stem + "_round0" + path.suffix))
    (batch / "candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    # Export intended training schema for inspection, never mark pending rows gold.
    export_review_csv(batch, records)
    attempts = len(drafts) + len(revisions)
    cumulative_attempts = attempts + config.get("prior_attempts", 0)
    ledger = {
        "candidate_attempts": attempts,
        "unique_candidate_ids": len(drafts),
        "cumulative_attempts": cumulative_attempts,
        "cumulative_unique_candidate_ids": len(prior_records) + len(drafts),
        "superseded_drafts": len(revisions),
        "recipe_revisions_used": int(bool(revisions) and apply_revision),
        "editorial_replacement_drafts": len(revisions) if apply_editorial else 0,
        "accepted": 0,
        "pending_human_review": len(records),
        "rejected": 0,
        "stage_A_remaining_attempts": 150 - cumulative_attempts,
        "overall_remaining_attempts": 1000 - cumulative_attempts,
        "draft_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in draft_files + revision_files
        },
        "generation_reproducibility": "Immutable authored drafts; exact model replay is not guaranteed.",
    }
    (batch / "attempt_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    render_review(batch, records)
    return records


def export_review_csv(batch, records):
    fields = [
        "candidate_id",
        "text",
        "locations",
        "loc_markers",
        "language",
        "genre",
        "domain_pool",
        "review_status",
        "reviewer",
        "review_notes",
        "agent_review_status",
        "agent_review_notes",
    ]
    with (batch / "review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = {k: r[k] for k in fields}
            row["locations"] = repr(r["locations"])
            row["loc_markers"] = repr([tuple(s) for s in r["loc_markers"]])
            writer.writerow(row)


def render_review(batch, records):
    parts = [
        f"# Terra preview: {len(records)} synthetic candidates\n",
        "All posts describe invented events. **Pending human review; not gold training data.**\n",
        "Bold text marks every proposed location occurrence. Check the entire post for missing mentions.\n",
    ]
    for r in records:
        chunks, cursor = [], 0
        for a, b in r["loc_markers"]:
            chunks.extend([r["text"][cursor:a], "**" + r["text"][a:b] + "**"])
            cursor = b
        chunks.append(r["text"][cursor:])
        parts.extend(
            [
                f"## {r['candidate_id']} · {r['language']} · {r['genre']}\n",
                f"{len(r['text'])} characters · {len(r['loc_markers'])} occurrences · "
                f"agent review: {r['agent_review_status']}\n",
                "".join(chunks) + "\n",
                f"Locations: `{json.dumps(r['locations'], ensure_ascii=False)}`\n",
                f"Agent notes: {r['agent_review_notes'] or 'Awaiting parent-agent review.'}\n",
                "Human decision: pending.\n",
            ]
        )
    (batch / "REVIEW.md").write_text("\n".join(parts))


def diagnostics(records):
    count = len(records)
    return {
        "count": count,
        "languages": dict(Counter(r["language"] for r in records)),
        "genres": dict(Counter(r["genre"] for r in records)),
        "domain_pools": dict(Counter(r["domain_pool"] for r in records)),
        "length_buckets": dict(Counter(length_bucket(r["text"]) for r in records)),
        "median_characters": statistics.median(len(r["text"]) for r in records),
        "newline_fraction": sum("\n" in r["text"] for r in records) / count,
        "blank_line_fraction": sum("\n\n" in r["text"] for r in records) / count,
        "emoji_fraction": sum(
            bool(re.search("[\U0001f300-\U0001faff\u2600-\u27bf]", r["text"])) for r in records
        )
        / count,
        "mention_counts": dict(Counter(len(r["loc_markers"]) for r in records)),
        "tokenizer_validation": "not checked by this character-span assembly command",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    corrections = parser.add_mutually_exclusive_group()
    corrections.add_argument("--apply-revision", action="store_true")
    corrections.add_argument("--apply-editorial-corrections", action="store_true")
    args = parser.parse_args()
    if (args.batch / "candidates.jsonl").exists() and not (
        args.apply_revision or args.apply_editorial_corrections
    ):
        raise SystemExit(
            "Candidates already exist; preserve reviewed records rather than overwriting."
        )
    rows = build(
        args.batch,
        apply_revision=args.apply_revision,
        apply_editorial=args.apply_editorial_corrections,
    )
    result = diagnostics(rows)
    (args.batch / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
