# Training on Colab

Notebooks 03–05 use the same Drive export and W&B logging code. The optimisation recipe remains in
`ner/recipes.py`; it has not been changed while notebook 02 is training.

## Start a run

1. Publish branch `herman/transformers-training`, then open the desired notebook from that branch in
   Colab. The setup cell fetches `REPO_REF`; use a commit SHA to reproduce a particular version. A branch
   that exists only locally cannot be fetched by Colab.
2. Select a GPU runtime. Set `DRIVE_ROOT` to your Drive folder and `WANDB_ENTITY` to your team entity.
   A shortcut to a shared folder must exist at that same path for each teammate. Otherwise each person
   writes to their own My Drive. Set `WANDB_PROJECT = None` if logging is intentionally disabled.
3. Run setup, mount Drive and authenticate W&B. Setup installs
   [`requirements-colab.txt`](requirements-colab.txt), verifies input files and checks Drive writes.
   Restart the runtime when changing installed libraries or the training checkout.
4. Review the printed recipe and model revision. Run the memory preflight, then the LR trials and folds.
   LR selection uses `half_B` development F1, with lower LR breaking exact ties. The decision is saved.
5. Keep the final export and reload-inference cells enabled. Submit the exported `submission.csv` to
   Kaggle manually, then fill the score and numeric submission ID in the score-recording cell.

Set `os.environ["NER_SMOKE"] = "1"` before setup for a shorter end-to-end run. This writes to
`dl-a4-ner/runs_smoke` on Drive and disables W&B. It still loads the real encoder; the memory preflight
uses the full configured window/batches. Smoke results never substitute for measured full-run results.

## What survives a runtime reset

```text
dl-a4-ner/
  runs/<encoder>-<config-hash>-lr<value>-fold<k>/
    model/                 # final weights, model config, tokenizer
    config.json            # full settings, fold, pinned Hub revision
    provenance.json        # Git/source/input hashes, packages, precision, GPU
    metrics.json           # sentence/synthetic validation, half A/B, gold26
    history.json, grad_flow.json
    rows.json              # row identities, text hashes, reference locations; test IDs for A/B
    probs.pkl              # offsets and O/B-LOC/I-LOC probabilities, including OOF validation
    decoding.json, submission.csv
    wandb.json             # run URL/ID when logging is enabled
    complete.json          # checksums, published only after the full export
    work/stage-*/          # checkpoints + finished-stage snapshots while a run is incomplete
  runs/<experiment>-memory.json
  ensembles_smoke/          # smoke exports never overwrite full-run ensembles
  ensembles/<encoder>-<config-hash>/
    members.json, selection.json, decoding.json, metrics.json
    lr_comparison.csv, cv.csv, figures/*.png
    probs.pkl, submission.csv, complete.json
    wandb.json
    kaggle/<submission-id>-public.json
```

Kaggle records contain the actual score, timestamp and submission hash. Hidden Kaggle labels are not
available locally; `half_B` scores use teacher pseudo-labels. The W&B ensemble run logs metrics, figures
and a small evaluation artifact; trained weights remain on Drive.

W&B curves have separate stage step axes, so stage 2 does not overwrite stage 1's x coordinates.
Drive completion is published before final W&B uploads. An upload failure is recorded in
`logging.json` and does not invalidate the saved weights or ensemble. Retrying an upload is separate
from retraining; `ner.experiment.log_ensemble` can log an already-exported ensemble.

On restart, completed runs are checksum-verified before reuse. Incomplete runs resume from their last
checkpoint, saved every 250 optimizer steps by default. Completed stages are restored instead of
retrained. Optimizer, scheduler and RNG recovery use Trainer checkpoints; custom curves are stored
with them. Two checkpoint generations are retained, and recovery selects the newest checksummed one,
skipping a partial newer write. If none is valid, the stage restarts from its initial weights. Ordinary
seeded shuffling preserves row order on resume; length grouping is disabled. There is no claim of
bitwise GPU reproducibility or transactional recovery from a failed Drive write.

Each stage's checkpoints are removed after its stage snapshot is verified; stage snapshots are removed
after publishing the final inference bundle. Recovery files can consume several times the final model
size. Allow roughly **25 GiB of free Drive space per active XLM-R large run** for transient checkpoint
writes and fallback retention, in addition to completed runs. The mounted filesystem's free-space
number does not establish your account's Drive quota; check that quota in Drive before starting. A
damaged completed bundle raises an error rather than silently retraining or using partial results.
Experiment names hash settings, source, inputs and package versions. Use a new run name/configuration
after changing those. Completed ensembles are reused, and changed members/settings require a new
directory so previous submissions and Kaggle records stay intact. The code does not
coordinate two writers to the same run directory: assign distinct encoder/LR/fold jobs to teammates.

Load a finished run in a fresh runtime without downloading its original base model:

```python
from ner.inference import predict_saved

predictions, probabilities = predict_saved(
    "/content/drive/MyDrive/dl-a4-ner/runs/<run-name>",
    ["У Києві перекрили вулицю Хрещатик."],
    precision="auto",  # fp32 on CPU; bf16/fp16 on a supported GPU
)
```

Only unpickle `probs.pkl` files from your own runs. Saved row order, text hashes, labels and offsets
are checked before fold averaging. Different encoders still require character alignment or entity
voting for a future cross-model ensemble; token probabilities from different tokenizers cannot be
directly averaged. These exports retain the information needed for that work.

## XLM-R large: memory findings

The old notebook reduced training batch to 8 but still evaluated/inferred at 64, without explicit
inference autocast. The stage-1 Trainer and optimizer also stayed alive while stage 2's optimizer was
created. Those were concrete allocation risks, not a measured T4 failure.

The revised notebook starts at microbatch 2, gradient checkpointing, and evaluation/inference batch 4.
Accumulation preserves the recipe's effective batch (currently 32; 16 accumulation steps). Inference
uses the same resolved mixed precision as training. Stage optimizers are released before the next
stage or final inference. Changing evaluation batches does not change the intended training objective.
The loss now normalizes valid tokens across the accumulated batch, preserving token weighting when
microbatch sizes change. Gradient diagnostics reduce norms without materializing a squared embedding
gradient tensor. Native BF16 support is required; T4 therefore selects FP16. A resumed run retains its
recorded precision instead of switching scaler formats when the GPU changes.

For roughly 560M parameters, FP32 weights, gradients and two Adam moments alone need about **8.3 GiB**
(560M × 16 bytes), before activations, optimizer intermediates, CUDA workspace and allocator reserves.
Mixed-precision autocast does not remove those FP32 optimizer allocations. Model fit cannot be judged
from weight-file size alone.

`ner.preflight.memory_check` loads the actual pinned encoder, runs two full-window accumulated backward
and optimizer steps (the second with optimizer state resident), then both evaluation batch sizes. It saves
peak allocated/reserved VRAM, total VRAM, GPU and settings to Drive. Its probe model is discarded; no
preflight update enters an experiment. Check the report for headroom before proceeding. If it fails,
reduce microbatch/evaluation/inference batches and adjust accumulation to retain the effective batch.
Do not silently shorten windows or change optimizers just to make the probe pass.

This reduces known risks but **T4 fit and throughput remain unmeasured** in this local environment.
The preflight is a representative memory check, not a guarantee that every full-run allocation fits.
mDeBERTa also needs a GPU precision check; use `precision="fp32"` if mixed precision is non-finite.

The implementation follows the documented
[Trainer checkpoint/gradient-checkpointing settings](https://huggingface.co/docs/transformers/main_classes/trainer)
and [PyTorch autocast](https://docs.pytorch.org/docs/stable/amp.html).

## Local verification

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python -m unittest discover -s tests -v
```

The tests build a tiny BERT model/tokenizer locally. They exercise real two-stage Trainer execution,
checkpoint recovery during a later epoch, exact restored weights for a deterministic model, reload
inference, accumulation gradients, verified/fallback exports, smoke isolation, ensemble output, Kaggle
score records, failed W&B uploads and the memory probe. No model download, GPU training,
Drive authentication or W&B upload is needed. They do not establish that the three full-sized encoders
fit the selected Colab GPU.

Current local result: **11 tests pass**; notebook schemas, code-cell compilation and lint checks pass.
For resumed runs, `train_minutes` measures the completing session and excludes earlier interrupted
sessions; do not use it as a total GPU-cost estimate.

Notebook 02's running Python session keeps its already-imported implementation. These branch changes
do not retroactively preserve weights from its completed runs. Incorporate its eventual winning
settings only after reviewing its exported results; then freeze one revision for all model notebooks.
