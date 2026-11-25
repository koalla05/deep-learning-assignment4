# Handoff to the implementation agent

There are 50 local Terra-authored posts and four completed exploratory XLM-R
fits. Read [AUC_REPORT.md](AUC_REPORT.md) for measured results and limitations;
[SECOND_BATCH_REPORT.md](SECOND_BATCH_REPORT.md) records the final batch.

## Task

Continue the bounded pilot in [PLAN.md](PLAN.md) only within the user's next
requested scope. The requested 50-post generation and exploratory AUC steps are
complete; this handoff does not request another generation or training run.

The user wants labeled synthetic training candidates that resemble the
competition Telegram posts more closely than the supplied short sentences.
The immediate experimental objective is a reproducible reduction in held-out
domain-classification ROC AUC, not maximum volume and not an extraction-F1 claim.

The user confirmed permission for synthetic supervision and aggregate target-text
adaptation, selected `gpt-6.1-terra` with high reasoning through subagents, and
requested parent-agent review. The user later authorized the exploratory AUC run
and a shared-code refactor of the existing notebook. Those instructions supersede
the original local-generator and notebook-isolation recommendations.

There have been 61 generation attempts, including 11 superseded drafts, and 50
current candidates. Parent-agent review is complete; human label review and the
independent second review remain pending. The 50 posts use only location-first
generation, and are not a completed version of the larger balanced-stream pilot.

## Important context

- Task: extract Ukrainian/Russian location mentions as exact text substrings.
- Model family for the assignment: `FacebookAI/xlm-roberta-base`.
- Existing adversarial validation: source sentences versus the 477 unlabeled
  competition posts, with domain labels 0 and 1 respectively.
- Reported AUC: 0.992 raw; 0.972 with formatting masked and exactly 32 tokens.
  These are historic results, not the baseline for the new comparison.
- Exploratory 50-post AUC: original/generated 0.9479/0.7574 raw and
  0.7312/0.8545 controlled. Both delta intervals include zero. The holdout has
  now been observed; do not treat another recipe's score on it as fresh evidence.
- Source CV comprises 63,388 rows in three frozen grouped folds.
- Most competition posts are Ukrainian. They are longer, use lists and footers,
  and cover local-news contexts that are uncommon in source sentences.
- The 26 supplied labeled posts reveal annotation-policy differences and
  possible omissions. They are qualitative examples, not a tuning set.
- Existing CV and leaderboard rankings agreed poorly. Lower domain AUC is a
  useful pilot result, but does not establish improved extraction.

## Before any further experiment

1. Read every document in this folder and the linked split policy. Inspect the
   real schema of `data/cv_rows.parquet`; do not assume unlisted provenance
   fields exist.
2. Reuse the recorded Terra generation preference and isolated classifier runtime
   where suitable. Record any newly selected model, dependencies or hardware.
3. Preserve the confirmed synthetic-data/aggregate-adaptation permissions. Ask
   only if a new action falls outside the scope already authorized by the user.
4. Resolve the reviewer workflow and ambiguous annotation rules with the user.
   Human review is a release gate, not something agent self-review can replace.
5. Freeze the parent restrictions, target groups, generator-family split,
   candidate budget, statistics profile, comparison sizes, and seeds before
   fitting models or inspecting new held-out AUC.

If a prerequisite is unavailable, implement the safe scaffolding and tests, then
report the precise blocker. Do not invent permissions or claim unreviewed labels
are gold.

## Implementation boundaries

- Keep pilot orchestration under `generation_pilot/` and shared classifier code
  under `evaluation/`. The notebook now uses that code; preserve its historical
  outputs and the frozen source folds.
- Reuse the supplied metric for optional sanity checks, without correcting its
  published normalization behavior.
- Keep original labels and offsets immutable. Store audited/generated labels
  separately with provenance.
- Never feed target posts, the 26 labeled examples, source-held-out rows, or
  their paraphrases to the generator or a training labeler.
- Terra subagents were explicitly authorized for writing. Do not upload raw
  competition posts or labels to a separate generator or compute service.
- No scraping, submissions, pushes or full extractor runs are requested. The
  user has authorized granular local commits of the polished code and reports.
- Stop after one audited recipe revision. A failed pilot is a valid outcome.

## Expected deliverables

Create executable code and focused tests for:

- Frozen manifests and parent/family grouping.
- Assembly-time character-span creation and preservation.
- Label validation, Unicode/tokenization checks, and duplicate exclusions.
- Candidate generation, rejection accounting, and provenance.
- Review export/import with explicit pending/accepted/rejected states.
- The paired domain-classification experiment and uncertainty reporting.

Write an actual-results report after execution, containing:

- Input hashes, selected IDs, seeds, model/configuration versions, and counts.
- Candidate/accepted/rejected counts by language and generation family.
- Review results: missing mentions, spurious mentions, boundaries, and fluency.
- Leakage checks and unresolved annotation-policy issues.
- Baseline and synthetic AUC per seed for raw and controlled conditions.
- Paired AUC differences and group-bootstrap intervals.
- Distribution diagnostics and a clear supported/inconclusive/failed conclusion.
- Limitations: historic target exposure and no evidence yet of extraction gain.

Future artifacts belong in `generation_pilot/artifacts/`, which is ignored.
Keep code, tests, and a compact report separate from raw text and model outputs.

## Completion condition

Complete the pilot only when the accepted labels have passed review, leakage
checks pass, and the frozen baseline-versus-synthetic protocol has run. If these
conditions cannot be met, stop with a blocker report rather than increasing
volume, relaxing checks, or repeatedly tuning against held-out posts.
