# Terra preview report

This report records the first 20-post batch before further generation or model
fitting. See [SECOND_BATCH_REPORT.md](SECOND_BATCH_REPORT.md) for all 50 posts
and [AUC_REPORT.md](AUC_REPORT.md) for the subsequently completed comparison.

The first 20 original candidates were authored by `gpt-6.1-terra` subagents with
`high` reasoning, then reviewed in full by the parent Codex agent. This follows
the user's model and review preference. No Ollama or external generation API was
used. The user's explicit subagent choice supersedes the earlier local-model
recommendation for this writing step.

## Current outputs

- [Highlighted review](artifacts/preview_001/REVIEW.md): all text and proposed spans.
- [Review CSV](artifacts/preview_001/review.csv): original data-style columns and review fields.
- [Candidate JSONL](artifacts/preview_001/candidates.jsonl): occurrence spans, draft slots,
  assembly maps, lineage, versions, author model, and review notes.
- [Automated audit](artifacts/preview_001/audit_checks.json).
- [Parent review](artifacts/preview_001/parent_review.json).
- [Attempt accounting](artifacts/preview_001/attempt_ledger.json).

Artifacts remain local and ignored by Git. Human review statuses remain pending;
an agent review is explicitly recorded separately. None is silently marked gold.

## Volume and generation

The overall medium-volume pilot remains 600–800 accepted posts, at most 1,000
generation attempts including rejected/superseded drafts. This 20-post preview
is the first part of Stage A, not the completed medium dataset. There were 20
initial drafts and eight replacement drafts in one correction round: **28 total
attempts**, 20 current candidates, zero human-accepted posts. Stage A has 122
remaining attempts. Do not scale before its audit requirements are satisfied.

All 20 previews use the location-first stream to inspect the requested Terra
writing behavior. The later Stage A allocation still needs source-based assembly
to reach the approximately balanced stream mix. Six genres are represented;
18 posts are Ukrainian and two Russian. Sixteen belong to domain fit and four
to domain evaluation. No source group contributes to more than two current posts.
Only folds 1 and 2 supply location resources; fold 0 remains protected.

The median length is 277 characters, against 264 in the observed test texts.
Fifteen posts have blank lines and 14 contain emoji. This preview covers three
posts below 80 characters, six at 80–220, six at 221–500, four at 501–1,100,
and one above 1,100. Three posts have no locations; others cover single,
repeated, address, multiword-street, and list mentions. These counts describe a
small deliberately stratified preview, not proof of distribution matching.

## What the parent review changed

The parent inspected every complete draft and all spans, including negatives.
Terra revised eight posts while preserving the original drafts:

- Removed synthetic-example disclaimers from weather text; fictional provenance
  is recorded outside the training text.
- Replaced a multi-city mixture of street works with a coherent single worksite.
- Corrected Ukrainian euphony and a questionable vehicle choice.
- Reduced forced repeated-city wording.
- Replaced two topics that were too similar across domain pools.

Some repeated-address phrasing remains somewhat formal. This is recorded in the
per-post review notes rather than presented as perfect naturalness.

## Validation and limits

Four focused assembly tests pass, covering Unicode code-point offsets, emoji,
combining characters, repeated mentions, full address suffixes, negatives, and
missing/duplicated/unresolved slots. Every final span reconstructs its intended
surface form. All 34 surface-bank resources reconstruct their training-side
parent occurrence and retain provenance.

The audit found no exact/normalized overlap with source texts, test texts, or
labeled samples, and no suspicious near-duplicate flags under the documented
five-character-shingle thresholds. Internal candidate similarity checks also
returned no flags. These are heuristic checks, not proof against every semantic
paraphrase. Protected texts were used only by the exclusion checker; generators
received the style brief and eligible surface resources, not protected posts.

Manifests and recipe inputs were recorded before generation. Automated target
grouping joined 477 posts into 460 conservative groups with 21 overlap edges.
Human review of these repost relationships remains pending for the primary pilot.
The prior all-target aggregate inspection is disclosed; the split does not make
the target historically untouched.

Tokenizer-offset reconstruction, human review, the independent second review,
and the full medium dataset remain outstanding. No discriminator or extractor
was trained, no score was measured, and no competition submission was made.
Use the frozen paired AUC protocol only after its prerequisites are met. Record
the first score before adjustments; another recipe evaluation needs a separate
protocol that accounts for any already-observed holdout results.

## Reproduction

`prepare_preview.py` uses pandas/pyarrow and refuses to overwrite frozen inputs.
`preview_tools.py` uses only the standard library to assemble stored model drafts
and export reviews. `audit_preview.py` uses pandas/pyarrow for source integrity
and exclusion checks. The sibling assignment 3 environment supplied these read
dependencies; no packages or model weights were installed.

```sh
python3 -m unittest discover -s generation_pilot -p 'test_preview_tools.py' -v
../deep-learning-assignment3/.venv/bin/python generation_pilot/audit_preview.py
```

Do not rerun preparation over the current batch. To reconstruct assembly in a
separate directory, copy frozen JSON inputs and original draft files, run
`preview_tools.py --batch <directory>`, copy the revision files, then run it with
`--apply-revision`. Parent review notes are stored separately in
`parent_review.json`. Exact model replay is not guaranteed; original authored
drafts and their SHA-256 hashes are preserved.
