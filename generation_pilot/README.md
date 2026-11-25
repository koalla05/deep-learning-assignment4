# Synthetic-data pilot

There are 50 Terra-authored posts, 61 generation attempts including superseded
drafts, and four completed exploratory XLM-R fits. Parent-agent review and
automated span/provenance checks are complete; human location-label review is
pending. Generated-versus-target AUC is 0.7574 raw and 0.8545 controlled. Both
paired difference intervals include zero, so the comparison is inconclusive.

Start with the [AUC report](AUC_REPORT.md) and
[committed results summary](results/auc_050_seed42.json). The generated posts,
full manifests, model cache and predictions are local under ignored `artifacts/`.
Their links in the batch reports work only in the checkout containing those files.

## Objective

Build an auditable, Telegram-like labeled training candidate set and test
whether it is less distinguishable from competition posts than the original
source sentences. Prioritize correct, complete labels and clean provenance.
Do not scale generation or claim improved location extraction from AUC alone.

## Sources and workflow

- [Source index](sources/README.md): original data, notebooks, metric and split policy.
- [Generation style](GENERATION_STYLE.md) and [examples](EXAMPLES.md): writing and span policy.
- [Plan](PLAN.md): the larger bounded pilot and its primary comparison protocol.
- [Handoff](HANDOFF.md): current authorization, outstanding work and execution limits.
- [First batch](PREVIEW_REPORT.md) and [second batch](SECOND_BATCH_REPORT.md): generation history.
- [AUC report](AUC_REPORT.md): completed exploratory comparison and uncertainty.

The user confirmed that synthetic supervision and aggregate target-text adaptation
are allowed, selected `gpt-6.1-terra` with high reasoning, and requested parent-agent
review and the 50-post exploratory assessment. The current collection is entirely
location-first; it does not complete the larger balanced-stream pilot. A second
human reviewer is not yet available. Assignment 3 links require its sibling checkout.

## Code and validation

`prepare_preview.py` freezes the initial source/target assignments and resource bank.
`preview_tools.py` assembles authored placeholder drafts and exports pending reviews;
it does not call a generation model. `audit_preview.py` checks spans, provenance and
overlap. `run_adversarial.py` prepares and runs the paired classifier experiment,
using shared training/preprocessing from `evaluation/domain_classifier.py`.

Install [requirements-adversarial.txt](requirements-adversarial.txt) into a separate
Python environment, then run tests from the repository root:

```sh
python -m unittest discover -s generation_pilot -p 'test_*.py' -v
```

The [AUC report](AUC_REPORT.md#artifacts-and-reproduction) explains reproduction
and which local input artifacts must be present. No model fitting is needed to
check the committed result summary or run the unit tests.

## Scope ceiling

- At most **1,000 total generation candidates**, including rejected candidates
  and pilot batches. No unlimited retries.
- Aim for **600-800 accepted posts**. Fewer are acceptable if quality requires it.
- No competition submissions, extractor training, external scraping, or uploads.
- At most **12 XLM-R domain-classifier fits** in the primary protocol.
- Keep generated data, checkpoints, and predictions local and ignored by Git.

The examples in this folder are documentation, not training examples. Future
generation or fitting should be a separately recorded experiment: the current
evaluation holdout has now been observed.
