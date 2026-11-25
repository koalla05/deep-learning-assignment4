# Second Terra batch: 30 more posts

This is the generation-stage report. The requested exploratory comparison was
subsequently completed and is recorded in [AUC_REPORT.md](AUC_REPORT.md).

The user's requested 30 additional posts are generated and fully reviewed by the
parent agent. There are now **50 distinct posts**, with 61 total generation
attempts including 11 superseded drafts across both batches. Generation stops
at this requested count before the proposed AUC assessment.

## Files

- [All 50 posts, highlighted](artifacts/combined_050/REVIEW.md).
- [Combined CSV](artifacts/combined_050/review.csv), with `text`, `locations`,
  `loc_markers`, candidate IDs, language, genre, pool, and review status.
- [Combined JSONL](artifacts/combined_050/candidates.jsonl), including lineage,
  placeholder drafts, Unicode assembly maps, and per-post parent-review notes.
- [Combined provenance and accounting](artifacts/combined_050/provenance.json).
- [New 30 posts only](artifacts/preview_002/REVIEW.md).
- [Audit of all 50](artifacts/preview_002/audit_checks.json).

All text was authored by `gpt-6.1-terra` subagents with high reasoning. The parent
reviewed every new full text, all location occurrences, every negative post, and
the three corrected drafts. Human review remains explicitly pending. No generated
post is silently labeled human-accepted or gold.

## Counts and checks

The new batch has 27 Ukrainian and three Russian posts, 24 fit and six evaluation
posts, and all six genres. Combined totals are 45 Ukrainian / five Russian and
40 fit / ten evaluation. The combined median is 277 characters; 76% contain a
line break and 58% contain emoji. Seven posts have no annotated locations.

The three editorial replacements addressed a repeated lost-item topic, Ukrainian
euphony, and a near-duplicate transit caption across domain pools. Original drafts
and pre-correction outputs remain preserved. This is an editorial pass under the
same recipe, not a new recipe tuned against AUC. No AUC has been observed.

All spans reconstruct their exact intended strings. Surface resources remain
traceable to folds 1 and 2; no parent group contributes more than three posts
across both batches. The original batch and frozen target/source manifests remain
unchanged. Four focused assembly tests pass. The new batch is still location-first;
this 50-post collection does not complete the larger plan's balanced-stream pilot.

## Next AUC assessment

No classifier was trained or score measured in this step. With only ten synthetic
evaluation posts in the current grouped split, an AUC estimate will be unstable.
Treat the requested next assessment as exploratory and report uncertainty and
the exact eligible counts. It is outside the primary plan's minimum of 400
accepted posts and cannot establish the primary pilot's success rule.

Use text only as classifier input. Keep inherited source, target, and generation
family grouping fixed; do not fit and score on the same posts. Compare fresh
original-versus-target and synthetic-versus-target classifiers on the same target
evaluation groups, with comparable source-class counts. Record model/feature
choice and the exploratory protocol before fitting. Tokenizer reconstruction,
target repost review, and human label review remain pending for the full pilot.

Generated artifacts remain local and ignored by Git. No submission, extractor
training, new target-profile tuning, model download, or dependency installation
was performed.
