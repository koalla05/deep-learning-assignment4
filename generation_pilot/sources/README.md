# Source index and evidence digest

Paths below point to original resources. No datasets are copied into this folder.
Read the linked material rather than treating this digest as an executable
specification. No external web research was performed for this handoff.

## Assignment 4: primary evidence

| Source | What to inspect |
| --- | --- |
| [Repository README](../../README.md) | Domain/label mismatch, metric, validation policy, historic AUC |
| [Adversarial notebook](../../adversarial_validation.ipynb) | Cells 4-6: sampling/split; 8: model settings; 10-11: controlled inputs/results |
| [EDA notebook](../../eda.ipynb) | Cells 2-4: XLM-R lengths; 7: CV/leaderboard comparison |
| [Data guide](../../data/README.md) | Actual CV columns and fold semantics |
| [CV rows](../../data/cv_rows.parquet) | `text`, `loc_markers`, `fold`, `source`, `group_id` |
| [Target posts](../../data/test.csv) | 477 posts; location labels hidden, `[]` is a placeholder |
| [Qualitative labeled sample](../../data/labeling_sample.csv) | 26 examples; annotation-policy review only |
| [Published metric copy](../../evaluation/metric.py) | Literal normalization behavior and location-set F1 |
| [Shared domain classifier](../../evaluation/domain_classifier.py) | Preprocessing, classifier configuration, training and held-out scores |
| [Exploratory runner](../run_adversarial.py) | Frozen groups, matched source counts, paired AUC and cluster bootstrap |
| [AUC report](../AUC_REPORT.md) | Completed 50-post experiment and its limitations |
| [Committed result summary](../results/auc_050_seed42.json) | Exact scores, intervals, counts, configuration and input hashes; no raw text |

Recorded XLM-R token counts:

| Texts | Count | Median | 95th percentile | Maximum | Over 512 |
| --- | --- | --- | --- | --- | --- |
| Ukrainian source CV | 31,432 | 27 | 54 | 85 | 0 |
| Russian source CV | 31,956 | 30 | 59 | 217 | 0 |
| Competition posts | 477 | 73 | 314 | 926 | 8 |

The adversarial notebook samples 954 source sentences, approximately 90/10
Ukrainian/Russian, and uses a random stratified 80/20 split. It drops repeated
source groups/texts, but does not establish a comprehensive target repost split.
Its model sees at most 256 tokens raw. The controlled condition masks URLs,
emoji, and line breaks, keeps texts with at least 32 tokens, and truncates to 32.

Reported AUC is 0.9923 raw on 287 held-out texts and 0.9716 controlled on 150.
These are single-seed historic comparisons, not a precise population estimate.
The exploratory 50-post experiment reran both arms under a common frozen grouped
protocol. Original/generated AUC was 0.9479/0.7574 raw and 0.7312/0.8545
controlled. Both paired delta intervals include zero, so the result is
inconclusive. These sample sizes differ from the historical notebook's run.

Language evidence is heuristic: 428 target posts contain Ukrainian-specific
letters, 34 Russian-specific letters, one both, and 16 neither. Do not treat
these counts as perfect language annotations.

The README reports mean post length about 395 characters versus 102-119 for
source sentences. Street-prefix words appear inside 21.9% of the qualitative
sample's location labels versus 2-4% in source data. Small-sample policy
differences and omissions remain unresolved.

Historic CV/leaderboard rank correlation: 0.20 across 14 submissions; -0.28 among
12 CRF submissions. Neither was significant. This motivates domain work but
does not establish that AUC reduction will improve extraction.

## Assignment 3: supporting context and mandatory exclusions

These links assume both repositories remain in the same parent directory.

| Source | What to inspect |
| --- | --- |
| [Assignment 3 README](../../../deep-learning-assignment3/README.md) | "Required split and exclusion policy for future experiments" |
| [Detailed EDA](../../../deep-learning-assignment3/eda/README.md) | "Data and labels", "What the errors show", "Unfamiliar names", domain-analysis limits |
| [Existing adaptation plan](../../../deep-learning-assignment3/eda/competition_like_validation_plan.md) | Rules separating domain scoring, extraction training, and qualitative samples |
| [Domain terminology](../../../deep-learning-assignment3/CONTEXT.md) | Mentions, markers, validation groups, domain discrimination |
| [Role manifest](../../../deep-learning-assignment3/eda/output/role_manifest_v2.parquet) | Join by `(source, row_id)`; explicit permitted-role selection |
| [CV manifest](../../../deep-learning-assignment3/eda/output/cv_manifest_v1.parquet) | Existing grouped CV; do not substitute a new raw-data split |

Verify manifest availability and provenance before using them. A missing or Git
LFS pointer-only file is a blocker, not permission to fall back to raw CSVs.

Key evidence:

- Exact/normalized duplicates and Russian document links require group-level
  splitting. The audit excluded conflicting duplicate labels rather than
  guessing the correct version.
- Every `exclude_*` role is forbidden for training, tokenizer fitting, and
  evaluation. Final rows remain reserved.
- Ukrainian flattened tags lost adjacent-name boundaries. Character-span BIO
  reconstruction raised the ceiling from 96.57% to 99.49% on retained rows:
  a reminder to test representation fidelity, not evidence of perfect labels.
- Some reviewed source examples contained unmarked geographical names.
- Earlier models recovered roughly 24-25% of unfamiliar Ukrainian names versus
  76-87% of names seen in labeled training. Controlled name/context diversity
  matters, but never source it from protected labels.
- Lists, streets/neighborhoods, and footers recur among qualitative errors.
- A domain classifier measures text-distribution differences, not annotation
  compatibility or extraction quality.

## Metric caveat

The published normalization contains a malformed punctuation expression. The
repository preserves the literal scorer. Do not "repair" it in the pilot or
normalize generated gold spans to match it. Keep character-span truth separate
from score-time normalization.

## Confirmed decisions and remaining review

- The user confirmed synthetic supervision and aggregate target-text adaptation
  are permitted, and chose Terra subagents at high reasoning for generation.
- Parent-agent review covers all 50 candidates; human label review and a second
  human reviewer are still outstanding. No source labels were overwritten.
- Four exploratory classifiers ran on the Mac MPS GPU. The isolated runtime,
  model revision and package versions are recorded in the AUC report and summary.
- Only folds 1 and 2 supplied source material. The actual parquet includes
  `source`, `row_id`, `doc_id`, `group_id`, `role_v2`, `text_key`, `has_location`,
  `fold`, `text` and serialized `loc_markers`; no extra gazetteer was used.
- The target repost grouping is automated and still needs human review for the
  primary pilot. The exploratory evaluation holdout is now exposed.

There is no independently annotated real-post development set in assignment 4.
The moderate pilot can support a domain-AUC claim, not a downstream F1 claim.
