# Exploratory adversarial validation of the 50 generated posts

Synthetic-versus-target ROC AUC is **0.7574 raw** and **0.8545 controlled**.
The matched original-source baseline is **0.9479 raw** and **0.7312 controlled**.
Lower AUC toward 0.5 means less separation by this particular trained classifier.
The raw result favors the synthetic texts; the controlled result does not.
Neither paired difference has a 95% interval excluding zero, so the comparison
is **inconclusive** at this sample size.

## Results

| Condition | Original source vs target | Generated vs target | Delta: original minus generated |
| --- | --- | --- | --- |
| Raw, at most 256 tokens | 0.9479 | 0.7574 | +0.1904 |
| URLs/emoji/newlines masked, exactly 32 tokens | 0.7312 | 0.8545 | −0.1233 |

Percentile 95% group-bootstrap intervals, 2,000 draws:

| Condition | Original AUC interval | Generated AUC interval | Delta interval |
| --- | --- | --- | --- |
| Raw | [0.8937, 0.9891] | [0.5079, 0.9714] | [−0.0262, +0.4467] |
| Controlled | [0.5083, 0.9150] | [0.7026, 0.9611] | [−0.3688, +0.1175] |

This is not evidence that location extraction improved. Raw results are
consistent with closer overall style, but the controlled comparison does not
support closing the content gap. Because the conditions use different eligible
subsets, their difference cannot isolate a causal formatting effect.

## What ran

The user explicitly requested this exploratory run before expanding the dataset.
It is separate from PLAN.md's primary experiment, which requires at least 400
accepted posts and three seeds. Domain labels use only source/target membership;
location annotations are not used. Human location-label review remains pending.

Four fresh `FacebookAI/xlm-roberta-base` classifiers were fine-tuned on the Mac's
MPS GPU, one for each condition and source arm. Settings follow the previous
notebook: learning rate 2e−5, two epochs, training batch 16, seed 42, fixed padding,
final epoch, no checkpoint selection or tuning. The model revision is pinned to
`e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`. Evaluation batch size is 16 rather
than the old notebook's 64, and AdamW is selected explicitly for MPS compatibility.
Four fits took about 108 seconds including model loading/evaluation, excluding setup
and bootstrap computation. No extractor was trained or competition submission made.

The existing grouped source, target, and generation-pool assignments were retained.
Only source folds 1 and 2 were eligible; fold 0 remained protected. Original-source
rows were sampled once with the same per-pool language counts as synthetic rows,
one row per source group, separately for each condition. Each arm used exactly
the same target rows. All eligible target rows were retained; the class imbalance
is substantial and is shared by both arms. This is a small-data experiment with
40 synthetic fit posts, not the earlier 954-source-sentence experiment.

| Condition | Fit source | Fit target | Held-out source | Held-out target |
| --- | --- | --- | --- | --- |
| Raw, each arm | 40 | 383 | 10 | 94 |
| Controlled, each arm | 34 | 306 | 8 | 73 |

The raw held-out target posts occupy 89 groups; the controlled ones occupy 68.
Synthetic shared-parent and shared-template dependencies form only five held-out
clusters in either condition. Original held-out source rows occupy 10 and eight
groups respectively. Bootstrap draws resample whole groups, pair the target
resamples across arms, and resample each arm's source groups independently.
The intervals condition on the fixed predictions; they omit retraining/seed
variation. Five synthetic clusters are too few for precise uncertainty estimates.

The historic notebook's 0.9923/0.9716 scores use different sample sizes and a random
row split. They are context, not the baseline for these deltas. A new grouped split
also cannot undo the target texts' previous exposure through EDA and generation
style statistics. Target near-duplicate groups remain automatically constructed,
without the human repost audit required by the full pilot.

## Implementation and verification

[evaluation/domain_classifier.py](../evaluation/domain_classifier.py) now contains
the shared classifier configuration, masking, controlled-input filtering, encoding,
training, and prediction. The existing notebook calls this module; its original
outputs remain labeled historical rather than presented as newly recomputed.

[run_adversarial.py](run_adversarial.py) owns experiment preparation, frozen
examples, matching source counts, split checks, score storage, and uncertainty
reporting. An immutable configuration keeps the same training settings available
to both callers; dataset preparation remains separate from model fitting.

Ten tests pass, covering Unicode span assembly, negatives, malformed slots,
preservation of existing reviews, conflicting correction modes, masking,
transitive dependency grouping, leakage rejection, and bootstrap ties/AUC
orientation. Formatting and lint checks pass. Notebook code cells compile and its JSON schema validates. The full
historical notebook was not rerun; the four new fits executed its shared training
and preprocessing implementation. Tokenization checks confirm every controlled
input has exactly 32 non-padding tokens. The old long-text counting warning is
harmless here: actual model inputs are truncated to 256 or 32 tokens.

## Artifacts and reproduction

- [Committed result summary](results/auc_050_seed42.json): available in a fresh
  checkout; scores, intervals, counts, configuration, versions and input hashes.
- [Frozen pre-fit protocol](artifacts/auc_050_seed42/protocol.json).
- [Exact results](artifacts/auc_050_seed42/results.json).
- [Raw synthetic predictions](artifacts/auc_050_seed42/raw_synthetic_predictions.json).
- [Controlled synthetic predictions](artifacts/auc_050_seed42/controlled_synthetic_predictions.json).
- [Tokenization checks](artifacts/auc_050_seed42/tokenization_checks.json).
- [Run log](artifacts/auc_050_seed42/run.log).
- [Pinned direct dependencies](requirements-adversarial.txt).

The protocol was frozen before any new AUC was observed. It records hashes of
the input datasets/manifests, sampled examples, model revision, configuration,
software versions, and hardware. Original and synthetic training/evaluation IDs
are saved in the corresponding `*_examples.jsonl` files. Artifacts, environment,
model cache, and predictions are local and ignored by Git. Their links require
the original local artifacts; only the compact result summary is committed.

The isolated runtime lives at `generation_pilot/artifacts/runtime`; dependencies
were installed there without modifying the sibling assignment's environment.
Some data-loading dependencies are reused through a read-only site-packages path;
the pinned requirements support installation into a separate environment.
Model weights were downloaded into `artifacts/model_cache`. Repository texts
were not uploaded to a generation or compute service.

```sh
generation_pilot/artifacts/runtime/bin/python -m unittest discover \
  -s generation_pilot -p 'test_*.py' -v

# Requires the local combined_050 candidates, preview_002 manifests and model cache.
# Preparation refuses to overwrite existing inputs. Inspect saved results first.
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false \
  generation_pilot/artifacts/runtime/bin/python generation_pilot/run_adversarial.py \
  prepare --output generation_pilot/artifacts/new_reproduction

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false \
  generation_pilot/artifacts/runtime/bin/python generation_pilot/run_adversarial.py \
  run --output generation_pilot/artifacts/new_reproduction
```

Reproduction is optional; no extra fits were run after seeing these results.
The now-observed holdout is recorded as exposed. Future recipe adjustments must
not be portrayed as fresh confirmation on this same holdout.
