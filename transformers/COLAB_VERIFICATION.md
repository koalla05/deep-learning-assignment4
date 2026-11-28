# Colab verification — 2026-11-28

Verification used Colab CLI 0.7.4 and a separate Tesla T4 runtime, authenticated as the requested KSE
account. The starting checkout was `ef33ff5`; the corrected loader and its regression test are in
`beb3df8`; the AMP diagnostic fixes are in `19ba22b` and `357bef2`. No production training outputs or existing
uncommitted notebook edits were replaced.

## Recipe and inputs

Notebook 02's results are incorporated: two epochs on sentences followed by four on synthetic posts.
Notebooks 03–05 import the shared recipe and select their own learning rate on fold 0 before training
folds 1 and 2. XLM-R large uses microbatch 2 and accumulation 16 to retain effective batch 32.

All six repository inputs were present. Local checks found 63,388 sentences, 5,839 synthetic posts,
477 test posts and 26 gold examples. All training spans were within text bounds; test IDs were unique
and non-null. No sentence group or synthetic repost group crossed folds. Each stage excluded its
validation fold.

## Actual GPU probes

Each probe loaded the pinned pretrained encoder, completed two optimizer updates at the full
256-token window, and checked evaluation and inference batches with optimizer state resident.
Precision was FP16 autocast; trainable weights remained FP32. The runtime exposed 14.56 GiB VRAM.

| Encoder | Train microbatch × accumulation | Eval / inference batch | Peak allocated GiB | Peak reserved GiB | Result |
| --- | --- | --- | --- | --- | --- |
| XLM-R large | 2 × 16; gradient checkpointing | 4 / 4 | 10.42 | 11.21 | Passed |
| mDeBERTa v3 base | 32 × 1 | 8 / 8 | 11.65 | 13.11 | Passed after loader fix |
| mBERT cased | 32 × 1 | 8 / 8 | 5.31 | 6.04 | Passed |

mDeBERTa initially failed with `Attempting to unscale FP16 gradients`. A local half-precision tiny
checkpoint reproduced the incorrect parameter/gradient dtype. Explicit `dtype=torch.float32` in the
shared training loader fixed the regression and the actual mDeBERTa GPU probe. This follows the
[Transformers model-loading dtype interface](https://huggingface.co/docs/transformers/v5.0.0rc0/en/models).

These probes establish short-run fit, not complete-run throughput or protection against every future
allocation. mDeBERTa has the least reserved-memory headroom on this T4.

## Pipeline tests

All 15 regression tests passed locally and in a fresh CPU subprocess on Colab, from a clean checkout
of the exact pinned training revision. The earlier 12-test
suite also passed with CUDA visible. They execute
two-stage training with a tiny local encoder, checkpoint interruption/recovery, saved-model reload,
token-weighted accumulation, checksummed bundles, ensemble exports, submission identity and smoke
isolation. W&B failure handling is tested with simulated upload failures; live uploads were checked
separately as described below.

The notebook workflow test executes the actual code cells of 03–05 against a tiny local encoder and
fixture rows. Each notebook completes both LR trials, selection, three distinct fold runs, plotting,
ensemble export and saved-model reload. External setup is checked separately with the real Drive
mount and W&B credentials. The common training revision is pinned to
`357bef20057ade30b343478d214172fa22b06651` for overnight runs.

The first real-encoder runs exposed another issue: an early FP16 scaler overflow could leave NaNs in
gradient diagnostics and break strict JSON checkpoint writes. Diagnostics now record those values as
`null` with `finite: false`. They remain visible as missing observations rather than invented zero
gradients. A regression test reproduces the original checkpoint failure and verifies the saved bundle.

All three real encoders passed bounded two-stage training, evaluation on the real held-out post sets,
weight export/reload, completed-run reuse and ensemble export. Each submission contained 477 unique
test IDs. Training used 256 sentences and a quarter epoch of the top-quarter synthetic subset; the
full configured window and effective batch were retained. These runs check infrastructure and must
not be used to compare model scores. The actual notebook control-flow test covers both LR trials and
three folds; each real-encoder smoke export uses one member rather than retraining three large folds.

| Encoder | Drive model bundle | Saved-weight inference | Submission / ensemble |
| --- | --- | --- | --- |
| mBERT | Verified | Passed | 477 rows; verified export |
| mDeBERTa | Verified after recovery | Passed | 477 rows; verified export |
| XLM-R large | Verified | Passed | 477 rows; verified export |

mDeBERTa's first stage-2 attempt ran out of memory while the separate probe kernel still reserved
3.08 GiB of unused CUDA cache. Releasing that cache reduced its reservation to 0.04 GiB. The retry
restored the completed stage-1 snapshot and completed stage 2 and export. This was contention between
verification processes, not evidence that the standalone notebook needs a different recipe. Use one
training notebook per GPU runtime. The recovery retry took 293 seconds, excluding its earlier stage-1
session; mBERT took 249 seconds and XLM-R large 843 seconds for their bounded checks. These durations
include validation, exports and verification, and are not estimates for full training.

The same overflow case is also covered in Trainer's own log history: a non-finite `grad_norm` becomes
`null` before history and callbacks receive it. Training loss and finite norms remain unchanged.

## External integration status

- Drive: mounted under the requested KSE account; write/read, shared notebook setup, live tiny training
  exports and saved-model inference passed. Evidence is in
  `MyDrive/dl-a4-ner/verification/t4-20261128`. The Drive quota API returned 403, so remaining account
  quota is unverified; mounted filesystem free space is not a substitute.
- W&B: live two-stage training and metric readback passed in the
  [verification run](https://wandb.ai/models-kyiv-school-of-economics/dl-a4-ner/runs/4j1iqq5c).
  Both stage axes and final metrics were present. The supplied key's default entity is
  `models-kyiv-school-of-economics`, now the notebooks' default (environment override retained).
  Never put the token in source, notebook outputs or committed reports. CLI 0.7.4 records `exec --env`
  values as code in its history; verification injected the key through its client with logging and
  kernel history disabled. The key was not written to `/root/.netrc`.
  Ensemble metrics, figures and the evaluation artifact also uploaded in the
  [mBERT ensemble verification](https://wandb.ai/models-kyiv-school-of-economics/dl-a4-ner/runs/ckhp8z7m).
- Kaggle: the pipeline exports submissions and can record actual returned scores/IDs. Hidden-label
  validation still requires a real submission; no verification score is invented or submitted here.
- Cross-encoder ensemble: saved offsets/text identities support future alignment, but averaging token
  probabilities across different tokenizers is not implemented. Same-encoder fold averaging is tested.

## Runtime packages

Python 3.13.16; torch 2.14.1+cu130; transformers 5.0.0; accelerate 1.12.0; numpy 2.4.6;
pandas 2.2.3; pyarrow 23.0.1; matplotlib 3.10.0; wandb 0.28.1; sentencepiece 0.2.2;
protobuf 6.33.6. These are the actual Colab versions after installing `requirements-colab.txt`.

## Overnight launch

### Final launch audit and scheduling

The final source audit confirmed that all three notebooks compile, use the same verified training
revision, and retain their ensemble-export and saved-weight inference cells. The audited shared
training sources match that revision exactly. This audit uses the live Drive/W&B evidence above;
it does not claim a second live authentication check after the verification runtime was released.
Each new runtime must mount Drive and authenticate W&B again.

Each encoder performs four two-stage runs in sequence: two learning-rate trials on fold 0, followed
by folds 1 and 2 at the selected rate. The winning fold-0 run is reused in the three-fold ensemble.
Each run trains for two sentence epochs and four synthetic-post epochs, with its validation fold
excluded. That is 12 runs and 24 training stages across the three encoders. Standalone notebooks do
not start one another. The subsequent `06_train_sequence.ipynb` launcher adds an automatic queue:
existing training cells run in separate sequential child processes, with Drive logs/status and
stop-on-failure behavior. Startup authenticates Drive/W&B once. The final queue review found and fixed
an interruption bug: terminating the supervisor could leave its training worker alive. SIGTERM now
unwinds the active subprocess, records failure, and reaps the worker; notebook cancellation terminates
the whole queue process group. An OS file lock rejects duplicate queues for the same output root in
one runtime. It does not coordinate separate Colab runtimes; use one runtime for this launcher.

A real subprocess regression reproduced the orphan-worker bug before the fix and passed afterward.
The same test rejects a concurrent duplicate launch. A two-worker integration test runs the actual
notebook training cells with tiny local encoders/fixture data, through both LR trials, selected folds,
weight export, inference reload and ensemble generation: eight run bundles and two ensemble bundles
are checksum-verified, both workers exit, and inherited fixture credentials do not appear in logs.
Training, evaluation and export code is unchanged from the earlier T4 verification. This is a local
process-boundary check; the full-sized GPU queue has not been run end to end.
The final local regression suite passed all **20 tests** in 56.9 seconds, including the two real
workers and interruption/duplicate-launch checks. Lint, notebook code-cell compilation and diff
whitespace checks passed. The shared `ner/` training source still matches the T4-verified revision.

Recommend sequential encoders on the KSE account until its plan and compute-unit balance are known:
mBERT first for an early baseline, then mDeBERTa, then XLM-R large. Start each in a fresh runtime and
release the previous runtime. Parallel encoder training requires separate GPU runtimes, sufficient
account allowance and compute units, and additional Drive capacity. Never put two training notebooks
on the same GPU runtime or let two workers write the same run directory.

GPU memory is verified for all three encoders; full-duration capacity is not. Colab's
[official FAQ](https://research.google.com/colaboratory/faq.html) describes dynamic limits, generally
up to 12-hour runtimes, and up to 24 hours of continuous execution on Pro+ with sufficient compute
units. Concurrent GPU availability is not guaranteed. The account's current plan, remaining units and
Drive quota have not been established. Therefore completion of all three notebooks in one night is
not guaranteed. Use the first full run's stage timings and Colab's displayed unit consumption to
budget the remaining work; bounded smoke timings cannot establish that budget. Checkpoint recovery
limits lost work but does not extend Colab runtime limits.

For each encoder, retain all four run bundles (including the losing LR trial) and the selected
three-fold ensemble bundle. The ensemble submission is the preferred per-encoder Kaggle file;
individual fold submissions are also retained. Weights/config/tokenizer on Drive enable fresh-runtime
inference without Hub downloads. Saved probabilities, character offsets, text hashes and test IDs
enable later cross-encoder alignment. Cross-encoder voting/alignment remains separate implementation
work; direct averaging of the three tokenizers' probabilities is not supported. W&B stores curves,
metrics and ensemble evaluation artifacts; Drive is the authoritative weight store. Actual Kaggle
scores must be obtained by submission and then recorded with their submission IDs.

Notebooks 03–05 are ready to launch from `herman/transformers-training`, using a fresh GPU runtime for
each notebook and the pinned `REPO_REF`. Mount the requested KSE Drive, authenticate W&B, leave
`NER_SMOKE` unset and run top to bottom. The notebooks use the verified KSE W&B team by default.
Allow roughly 25 GiB of free Drive space per active XLM-R large run, in addition to completed exports;
account quota could not be checked through the CLI credentials. Full-length training and final Kaggle
scores remain the overnight experiment, not results of these smoke checks.
