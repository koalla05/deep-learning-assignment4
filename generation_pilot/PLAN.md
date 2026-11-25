# Pilot plan

## 1. Freeze the task and permissions

Here "target" means `data/test.csv`, not source CV validation. Its `locations`
values are placeholders, not negative labels. Domain classification may use
target/source membership; location supervision must never use guessed target
labels.

Using aggregate statistics of target texts is target-informed adaptation.
Confirm permission and disclose it. The existing EDA and domain models have
already used these target texts: a new partition prevents additional operational
leakage, but cannot make the dataset historically untouched.

Do not access assignment 3's final labels for this pilot. Keep every `exclude_*`
row excluded. If additional source rows are needed, join the original rows to
`role_manifest_v2.parquet` and explicitly select permitted roles.

## 2. Freeze the split before generation

For this single pilot, fix source fold 0 as protected extraction validation and
use only parents from folds 1 and 2. This is not a three-fold extractor
experiment; accepted posts remain fold-0-training-only candidates.

Inspect and preserve parent `group_id`. Assign whole groups to domain-fit and
domain-evaluation pools. A candidate with multiple parents must use parents
from the same pool. All derivatives of a parent stay together.

Create a target manifest using exact/normalized duplicate groups and reviewed
near-duplicate/repost relationships. Use a fixed approximately 80/20 grouped
fit/evaluation partition. Preserve it across every experiment. The existing
notebook's random row split is not sufficient for this new protocol.

Split generator template families and any reusable footer/context banks into
domain-fit and domain-evaluation pools before generating. Do not share literal
templates, parent texts, or boilerplate banks across these pools. Genres can
appear in both through different families. This makes evaluation harder but
reduces memorization of generator artifacts.

Freeze the generation recipe using fit-side information only. Do not give the
generator target posts or distinctive target quotes, signatures, events, or
target-derived name lists. Use coarse existing EDA findings and, if permitted,
fit-side aggregate profiles; never evaluation-side text for iterative design.

Audit candidates against protected source and target texts for exact, normalized,
and suspicious near-duplicate overlap. Use protected texts only in the exclusion
checker, not as generation material. Shared common city names are not themselves
leakage; copied content, supervision, and group lineage are the concern.

## 3. Fix label policy and parent quality

Use occurrence-level half-open character spans: `[start, end)`.
Derive the string list from the final text and spans. Keep repeated occurrences,
even though the official metric later deduplicates strings.

Define explicit decisions for street prefixes, inflected forms, districts,
landmarks/buildings, organization names, footers, numbers, and adjacent names.
Refer to EXAMPLES.md for policy uncertainties.

Audit chosen source parents for complete annotations. Original labels are not
automatically correct: assignment 3 documented missing mentions and conflicting
labels. Never silently rewrite the source corpus.

Where policy is unclear, avoid that construction or quarantine it. Do not turn
an unexplained omission in one sample into a universal negative-label rule.

## 4. Generate in two bounded stages

### Stage A: quality pilot

Create at most 150 candidates, approximately balanced across these streams:

- Conservative assembly of audited, related source sentences into coherent posts.
- Location-first generation with placeholders and controlled surface forms.

Use 4-6 diverse genre families: transport, roadworks, municipal notices, local
announcements, measurement lists, and short community updates. Avoid copying
specific target events or channels.

Audit every Stage A candidate. Have a second human independently annotate at
least 30, stratified across languages, streams, and difficult boundaries.
Adjudicate disagreements and permit at most one recipe revision.

If the recipe changes, old candidates remain in the total attempt count and must
be regenerated or re-reviewed before acceptance. Do not mix incompatible label
policies or use old candidates merely to reach the desired count.

### Stage B: moderate dataset

Maximum 1,000 total attempts across both stages, including rejected posts.
Aim for 600-800 accepted posts; do not exceed 800 accepted posts in this pilot.
Use at most three candidates per parent group and avoid a few templates
dominating the dataset.

Starting allocation, finalized before results:

- Approximately 90% Ukrainian and 10% Russian; not an exact inferred prevalence.
- Approximately half from each stream.
- Multiple length buckets around the target median of 73 XLM-R tokens, with
  short alerts, longer paragraphs/lists, and a small long-post tail.
- Include verified location-free posts, varied mention counts, repeated names,
  streets/districts, and uncommon training-side surface forms.

Do not force quotas for uncertain label statistics from the 26 samples.
Do not simply repeat generic sentences until target length is reached.

## 5. Label construction and acceptance gates

For conservative edits, maintain an explicit assembly/edit map. For location-first
generation, insert approved surface forms into unique placeholders and calculate
offsets during assembly. Never calculate gold spans by naive unrestricted
substring search: repeated names and non-geographical uses make it ambiguous.

Operate on the final Unicode text. If normalization is performed, do it before
span construction, not afterward. Verify tokenizer offset mappings and
token-to-span reconstruction; emoji, combining characters, apostrophes,
adjacent names, and overlapping windows deserve focused tests.

Each candidate must pass:

1. Valid span bounds, intended substring equality, no unintended overlap, and
   complete placeholder resolution.
2. No exact/normalized protected-text overlap or unresolved suspicious
   near-duplicate match.
3. Correct language/grammar and plausible coherent content.
4. Complete annotation: no missed locations outside the planned placeholders.
5. Consistent boundary and context policy, including negative examples.
6. Provenance and review status present.

An independently implemented local dictionary/checker or permitted extractor
may flag additional mentions. Check an extractor's training lineage before
using it: a model trained on protected labels is not a clean reviewer.
Automatic agreement is not proof of gold-label quality.

Human review must cover the full accepted text and all spans, not just a sample
of preselected mentions. Double-review at least 50 accepted posts in total,
including the Stage A subset. Report disagreement and adjudication results;
do not claim a statistical error bound from a non-random audit.

Quarantine all unresolved candidates. Do not silently mark ambiguous spans as
outside tokens. Accepted posts require no unresolved annotation defects.

Suggested record fields:

`candidate_id`, `text`, `loc_markers`, `locations`, `language`, `stream`,
`genre`, `template_family`, `parent_ids`, `parent_group_ids`, `allowed_cv_fold`,
`domain_pool`, `surface_form_resource`, `recipe_version`, `policy_version`,
`generator_version`, `seed`, `review_status`, `review_notes`, `rejection_reason`.

## 6. Run a fair AUC comparison

### Primary comparison

Train fresh domain classifiers for:

- Original source sentences versus target posts.
- Accepted synthetic posts versus the same target posts.

Synthetic-only is intentional: this tests the generated set, not a potentially
diluted training mixture. Mixture studies and extractor training are later work.

Use the same target fit/evaluation groups, source-class counts, approximate
language mix, model, training settings, and seeds in both arms. Original source
rows and generated parents must respect the frozen restrictions. Use the
original arm's own group-held-out rows; do not evaluate on its fit rows.

Use the smaller feasible source-class count across the two arms, with roughly
80/20 domain fit/evaluation allocation. Lock sampling before any new AUC is
observed. If fewer than 400 accepted posts are available, stop at a feasibility
report rather than inflate the pool or silently change the protocol.

Run two conditions:

| Condition | Inputs | Purpose |
| --- | --- | --- |
| Raw | Original text, same 256-token limit as the existing notebook | Overall distribution gap |
| Controlled | URLs/emoji/line breaks masked; eligible texts truncated to exactly 32 tokens | Gap beyond obvious formatting/length cues |

For the controlled arm, re-balance original and synthetic source counts and
language composition after filtering to at least 32 tokens. Use the same eligible
target subset in both arms, with membership still fixed by the manifest.
Record what filtering removes; raw and controlled results describe different
subpopulations and cannot isolate one causal effect.

Use `FacebookAI/xlm-roberta-base`, learning rate 2e-5, two epochs, batch size 16,
and seeds 42, 43, 44. Use the final epoch, no held-out checkpoint selection or
hyperparameter tuning. This is 2 arms x 2 conditions x 3 seeds = 12 fits.
Document hardware/runtime before running the full bounded batch.

Do not reuse the existing trained discriminator as the main evaluator:
fresh classifiers need to learn cues that distinguish the new synthetic data.

### Statistics and diagnostics

Report per-seed AUC and
`delta = AUC(original vs target) - AUC(synthetic vs target)`.

Positive delta means reduced discriminability. Bootstrap whole domain-evaluation
groups, sharing the resampled target groups between arms. Resample original and
synthetic source groups separately unless an explicit matched parent-group
design was frozen. Compute the mean delta across seeds in each bootstrap sample.
Report seed variability separately: a fixed-prediction bootstrap is not a full
interval over retraining or future text populations.

Also report token lengths, language mix, line breaks, emoji, URL frequency,
mention counts, duplicate rates, and template diversity. Do not extract target
mention-count "gold" from pseudo-labels. Check how often 256-token truncation
occurs; the main AUC concerns text beginnings, not all content of long posts.

### Prespecified decision rule

Declare the pilot's AUC objective supported only if:

- Mean raw delta is at least 0.05 and its 95% group-bootstrap interval is above 0.
- Mean controlled delta is positive and its interval is above 0.
- Label and leakage gates pass.

The 0.05 threshold is a practical pilot threshold, not a known downstream
benefit or a promised attainable effect. If intervals cross zero, call the result
inconclusive. If only raw AUC improves, report a likely surface-level improvement,
not closure of the content gap.

Do not seek AUC exactly 0.5, flip sub-0.5 scores to manufacture success, or keep
changing recipes after inspecting held-out results. At most one frozen final
comparison is allowed. A further experiment needs a separately approved protocol
and an honest account of prior holdout exposure.

## 7. Stop and hand back evidence

Deliver the moderate reviewed candidate set locally, the reproducible code,
manifests, quality audit, AUC report, and recommendation.

Do not merge accepted candidates into `cv_rows.parquet`, submit to Kaggle, train
the extractor, or generate more data. For future three-fold extractor CV, regenerate
separate pools from each fold's training-side parents; the present pool is not
universally safe to reuse.

Even successful AUC reduction does not validate the generated annotation policy
or prove better extraction F1. The next study would need an independently labeled,
permitted real-post development set and a controlled real/synthetic mixture trial.
