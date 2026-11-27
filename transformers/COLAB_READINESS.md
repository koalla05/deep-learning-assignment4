# Colab readiness review: notebooks 03–05

**Implementation update:** the original findings below describe the reviewed baseline. Branch
`herman/transformers-training` now adds Drive weight/tokenizer exports, verified completion, stage-aware
checkpoint recovery, W&B ensemble logging, result/figure exports, explicit Kaggle score preservation,
automatic LR selection, configurable memory settings and a real-model GPU preflight. See the
[training guide](README.md) for the current workflow. Full-sized Colab GPU execution and a live
Drive/W&B round trip remain to be validated; cross-encoder alignment remains future ensemble work.

Reviewed on 2026-11-28 against commit `dd793d64f95b8108e79f3ed429618c8a8101ea3f`.

**Decision: the notebooks can start independently from a repository checkout, but are not yet ready
for runs whose weights, results and ensemble inputs must survive a Colab reset.** The main blocker is
that trained weights and tokenizers are never saved. This review records findings and proposed changes;
it does not change training code, the running optimisation experiment, or the selected recipe.

## Current experiment state

The recent changes added the shared `ner/` implementation and notebooks 00–06, then updated the data
recipe following notebook 01. Notebook 01's saved table reports two-stage training at `half_B` F1
**0.8723**, sentence CV **0.8970**, and gold26 **0.8052**. These are notebook outputs, not measurements
rerun during this review. `half_B` uses teacher pseudo-labels, not hidden Kaggle labels.

The current recipe is two epochs on sentences followed by two on synthetic posts, stage learning rates
`3e-5` / `2e-5`, AdamW, linear decay, warmup 0.1, weight decay 0.01, batch 32 and windows 256/64.
Notebook 02 compares 16 configurations, including the baseline. Its local file has no executed cells
or saved outputs; there are no local `transformers/runs` results. The user's live training session was
not inspected. **No optimisation or regularisation winner can be established from this checkout.**
`transformers/leaderboard.csv` contains only its header, so no transformer public Kaggle scores can be
reported from the repository either.

## What already works

- Each of 03–05 has its own setup, recipe construction, two learning-rate trials, three-fold workflow,
  fold ensemble and submission export. They do not need another notebook's Python variables.
- Shared code and all six required data files are tracked: sentence folds, test CSV, gold sample,
  pseudo-label parquet, cleaned synthetic parquet and adversarial-score parquet. Their loaders work.
- Verified counts: 63,388 sentences; 5,839 synthetic posts; 477 unique test IDs (239 A / 238 B); 26 gold
  posts. No sentence or synthetic group spans folds. No test repost group spans A/B under the existing
  shingle-based grouping rule. This does not establish independence from every possible semantic copy.
- The current fold-0 recipe trains on 42,259 sentences, then 3,942 synthetic posts, excluding fold 0
  in both stages. Folds 1 and 2 likewise exclude their own fold through `training_frame`.
- In ordinary Colab mode, the output root is `/content/drive/MyDrive/dl-a4-ner/runs`. Completed per-run
  exports include configuration, metrics, histories, token probabilities and a submission.
- All code cells in 02–05 compile. All code cells in 03–05 are currently unexecuted, so this is not an
  end-to-end Colab or GPU certification.

The notebooks are **repository-dependent**, rather than standalone single-file notebooks. That is a
reasonable design: a fresh Colab session should clone one frozen repository revision and run one
notebook top to bottom.

## Findings and suggested changes

### 1. Blocker: trained weights and tokenizer are discarded

**Evidence:** [`ner/train.py`](../ner/train.py), `run` and `_training_args`: `save_strategy="no"`, no
`save_model` / `save_pretrained` calls, followed by deletion of the model. `config.json` describes the
run but does not contain learned parameters. `probs.pkl` supports decoding existing evaluation texts,
not inference on new texts.

**Impact:** after each run returns, its trained model cannot be recovered from the exported artifacts.
This applies to notebook 02 as well as 03–05. Already completed runs would need retraining if their
weights are required; do not assume W&B or the base-model cache contains them. If the live session uses
this exact implementation, its remaining runs have the same preservation gap. A later Git change will
not automatically replace functions already imported into that session.

**Suggestion:** before deleting the model, export the final model and tokenizer into the run's Drive
directory using their Hugging Face save APIs. Add a separate, reload-only inference path that uses those
local artifacts and the saved window/decoding settings. Verify a reload on a small fixed set without
downloading the original base model. Keep final inference exports separate from resumable checkpoints.
The official [model documentation](https://huggingface.co/docs/transformers/main_classes/model)
describes local `save_pretrained` / `from_pretrained` round trips.

### 2. High: a partial export is considered complete; interrupted training cannot resume

**Evidence:** `run` writes `metrics.json` before histories and `probs.pkl`; `run_or_load` treats the
existence of `metrics.json` as completion. A temporary directory containing only matching configuration
and metrics was accepted as a finished run during this review. There are no training checkpoints or
`resume_from_checkpoint` calls.

**Impact:** a Drive interruption during export can leave a run that is skipped on restart but fails
during plotting or ensembling. A training interruption restarts the current run, including stage 1 if
stage 2 was in progress. It does not resume from the last epoch.

**Suggestion:** write a completion manifest last, after validating required artifacts and their checksums;
load only complete runs. Save periodic resumable checkpoints, including optimizer/scheduler/RNG state,
and persist the stage number. Publish files through temporary names where possible. Drive is not a
transactional artifact store: verify copied files before marking completion. See the official
[Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer) for checkpoint
and resume APIs, and [Colab FAQ](https://research.google.com/colaboratory/faq.html) for runtime and Drive
limitations.

### 3. High: clean-runtime dependencies and provenance are incomplete

**Evidence:** setup cell 2 in all three model notebooks installs only `wandb`, then imports shared code
requiring PyTorch, Transformers, Accelerate, NumPy, pandas, PyArrow and Matplotlib. There is no NER
dependency specification. Tokenizer dependencies such as SentencePiece/protobuf need verification,
particularly for mDeBERTa. Setup clones the moving default branch, or calls `git pull`; model loads do
not specify a revision. Saved run configs contain no Git SHA, data hashes or package versions.

**Suggestion:** provide one tested dependency file, record the resolved environment/GPU, pin the Git and
model/tokenizer revisions, and verify the six input files before GPU work. Use the environment of the
successful optimisation run as the starting point. Do not upgrade its dependencies mid-run. Preserve
input hashes and fold assignments with the exports. An internet connection and initial Drive/W&B
authentication are still needed; W&B can be disabled through the existing configuration.

### 4. High: memory settings do not establish that XLM-R large fits a T4

**Evidence:** notebook 03 reduces training to batch 8 with accumulation 4, but token-loss evaluation is
fixed at batch 64 (`_training_args`) and post-level inference also defaults to 64 (`evaluate_model`).
`token_probs` uses `no_grad` but no explicit autocast, and no gradient checkpointing option is configured.
Notebooks 04–05 retain training batch 32. No fresh GPU run was performed in this review.

**Suggestion:** expose separate train, loss-evaluation and inference batch sizes; start validation and
inference conservatively for the large encoder. Add optional gradient checkpointing. Run a tiny
two-stage check for each actual encoder on the intended GPU, including final evaluation and saving.
Check mDeBERTa for finite losses in the chosen precision; its notebook already notes the fp32 fallback.
Treat the notebooks' T4 runtime/fit estimates as estimates until measured.

### 5. High: heterogeneous ensembling is not implemented

**Evidence:** [`ner/cv.py`](../ner/cv.py) averages three folds with the same tokenizer.
[`average_probs`](../ner/predict.py) explicitly raises when offsets differ; a small synthetic example
reproduced that rejection. It also silently truncates unequal text counts through `zip` (reproduced).
Saved probabilities contain texts and offsets, but not stable row IDs or data hashes; fold averaging
does not check text identity/order between runs.

**Impact:** fold ensembles work in principle, but this is not yet an XLM-R + mDeBERTa + mBERT ensemble.
Different tokenizers cannot be combined by directly averaging their token arrays. Compatible offsets
alone also do not guarantee that two rows refer to the same post.

**Suggestion:** save stable IDs, text hashes, label order, tokenizer revision and offsets for every
prediction. Assert identical row sets/order and array lengths before averaging. Add character-level
alignment with an explicit special-token/whitespace policy, or a documented entity-voting method, for
cross-encoder ensembles. Keep OOF predictions separate from in-sample predictions. Token probabilities
already saved for `cv` and `synthetic` are useful OOF material, but need row/fold metadata and checks.

### 6. Medium: result tables and ensemble products are not all preserved on Drive

**Evidence:** notebooks 03–05 display learning-rate tables, `cv_table`, plots and ensemble metrics, but
only explicitly save the ensemble submission CSV. Averaged probabilities, ensemble metrics, selection
decision and plots are not exported. Public Kaggle scores are read from the repository CSV, not Drive.

**Suggestion:** export the LR comparison and fold table as CSV, metrics and decoding/ensemble settings
as JSON, figures as PNG, and the averaged probabilities plus row metadata. Maintain the Kaggle ledger on
Drive with model/run ID, submission filename/hash, submission ID, public/private split, score and date.
Record scores manually or through an explicitly requested Kaggle workflow; none are available here.
Preserve validation predictions/truth separately from predictions for hidden Kaggle test labels.

### 7. Medium: a top-to-bottom run always selects the first learning rate

**Evidence:** cell 9 in all three notebooks sets `BEST_LR = LEARNING_RATES[0]`, despite the preceding
instruction to select by `half_B`. There is no automatic selection or explicit stop before folds 1–2.
Run names contain only encoder prefix, LR and fold; updating the recipe will trigger the existing
different-config guard for old names rather than creating a new experiment family.

**Suggestion:** select deterministically from complete fold-0 results, record the criterion/tie rule,
and show the selected config before launching folds 1–2. Alternatively require an explicit selection
cell that fails until filled. Include a frozen recipe version or configuration hash in run names.
This also avoids colleagues writing different recipe versions into the same shared Drive folder.

### 8. Medium: selection scores need accurate interpretation

Notebook 01, notebook 02 and the model-specific LR trials all choose by `half_B`. It is therefore a
**development set with pseudo-labels**, not an untouched final test. Gold26 remains report-only.
The fold-0 sentence CV score is also reused during development, so the final three-fold mean is not a
nested-CV estimate of the entire selection procedure. The root README's statement that tuning happens
only on CV does not match this newer notebook workflow.

Notebook 06 tunes decoding on A, but its claim that reporting on B makes the final score honest needs
the same qualification: B has already influenced model and recipe selection. Keep the historical A/B
split, report the selection history, and use Kaggle's hidden-label evaluation as a separate result.
Do not pick ensemble weights by leaderboard trial and error. A fresh labeled post holdout would provide
a stronger independent estimate if one becomes available under the assignment rules.

## Proposed Drive artifact contract

All three notebooks should share the same exporter and loader. A run is complete only after validating
this bundle; use a unique recipe/configuration identifier in the directory name.

```text
MyDrive/dl-a4-ner/
  runs/<encoder>-<recipe-id>-lr<value>-fold<k>/
    config.json, provenance.json, complete.json
    model/                 # weights, model config, tokenizer files
    metrics.json, history.json, grad_flow.json
    predictions/           # IDs, text hashes, offsets, label order, probabilities
                           # held-out CV/synthetic, A, B and gold26 kept identifiable
    submission.csv
    checkpoints/           # resumable state and stage, bounded retention
  ensembles/<ensemble-id>/
    members.json, decoding.json, metrics.json
    cv.csv, lr_comparison.csv, figures/
    predictions/, submission.csv
  leaderboard.csv          # dated Kaggle score + submission/run identity
```

Keep the current probability format readable during migration. Large exports belong on Drive, with
small summaries/manifests optionally committed to Git. Update `.gitignore` when introducing model
directories: it currently ignores checkpoints and `probs.pkl`, but not new weight export directories.
`NER_SMOKE=1` currently redirects output to local `transformers/runs_smoke`, including on Colab; those
smoke outputs are intentionally not persistent unless explicitly copied.

## After optimisation finishes

1. Preserve notebook 02's outputs and full per-run bundles from Drive. Check which artifacts actually
   exist; completed runs made by the present code will not have trained weights.
2. Compare the baseline and completed trials using the saved configs and `half_B` development scores,
   with sentence/synthetic validation and gold26 as context. Check finite losses, precision/recall,
   stage curves and training cost. No winner is selected in this review.
3. Confirm the combined recipe in a new run before transferring it. One-factor improvements need not
   compose, and optimizer/LLRD trials also change LR. The 0.013 seed variation mentioned in notebook 02
   comes from a different Assignment 3 model; it is not a measured uncertainty threshold for these runs.
   Use paired post-level comparisons and repeat close contenders when compute permits.
4. Freeze `OPTIMIZATION` and the stage-2 LR/epochs together. Retain model-specific memory overrides and
   verify the intended stage-2 LR ratio. Notebook 03 currently forces effective batch 32 (8 × 4), so a
   winning batch-16 recipe would require adjusting accumulation as well. Freeze one Git revision for
   all three model notebooks.
5. Resolve weight export, completion checks, dependencies and memory configuration before the full
   03–05 runs. Validate one tiny two-stage run per encoder on Colab, reload its weights in a fresh
   runtime, regenerate its predictions, and verify the full Drive bundle.
6. Run LR selection and folds, save both fold and ensemble products, then build and validate the
   cross-encoder ensemble with explicit row alignment. Preserve Kaggle submissions and scores beside
   their exact model/decoding configuration.

## Review limits

Checks performed: source and recent commit review; compilation of all code cells in 02–05; actual
data-loader/fold checks; CPU construction of training arguments; temporary-file reproduction of the
incomplete-run bug; small-array reproductions of cross-tokenizer rejection and row-count truncation.
No training, model downloads, notebook execution, Kaggle submissions or Drive writes were performed.
The live Colab optimisation session was neither modified nor verified. Existing unrelated pilot-test
edits were left untouched.
