# Training on Colab

Notebooks 03–05 use the same Drive export and W&B logging code. The optimisation results from
notebook 02 are incorporated in `ner/recipes.py`: two epochs on sentences, then four on synthetic
posts. AdamW and the other baseline settings remain unchanged. Each encoder still compares its own
learning rates before training the remaining folds.

## Start a run

For automatic sequential execution, open **`06_train_sequence.ipynb`** instead. Authenticate Drive
and W&B once and run all cells. Set `MODELS` to any ordered subset of `mbert`, `mdeberta`, `xlmr`;
the default runs all three. Each encoder executes the existing notebook's training cells in a fresh
child process, then exits before the next starts. Drive logs/status live under `sequence/` (or
`sequence_smoke/`). A failure stops the queue. Rerunning uses the existing bundle verification and
checkpoint recovery. There is no external polling or automatic allocation of replacement runtimes;
Colab lifetime and compute-unit limits still apply. Full overnight queue duration remains unmeasured.
Interrupting the launch cell terminates its process group, including the active training worker.
The queue rejects a duplicate launch against the same output root within one runtime. This lock does
not span separate Colab runtimes: run this launcher in only one runtime for these outputs.

1. Open the desired notebook from branch `herman/transformers-training` in Colab. Setup fetches the
   pinned training commit in `REPO_REF` (`357bef2`), shared by all three notebooks. Change it only for a
   deliberate new experiment; the selected commit must be published for Colab to fetch it.
2. Select a GPU runtime. Set `DRIVE_ROOT` to your Drive folder. `WANDB_ENTITY` defaults to the verified
   `models-kyiv-school-of-economics` team; an environment variable can override it.
   Use one training notebook per GPU runtime. A shortcut to a shared folder must exist at that same
   path for each teammate. Otherwise each person
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

All three encoders passed the full-window preflight on a Colab T4 on 2026-11-28. Peak allocated/reserved
memory was 10.42/11.21 GiB for XLM-R large, 11.65/13.11 GiB for mDeBERTa and 5.31/6.04 GiB for mBERT
(14.56 GiB available). See [the verification record](COLAB_VERIFICATION.md) for scope and remaining
integration checks. These short probes do not establish full-run throughput or guarantee that every
later allocation fits.

mDeBERTa's checkpoint loads FP16 weights by default under the pinned Transformers version. That caused
`Attempting to unscale FP16 gradients` on the T4. The shared loader now explicitly loads FP32 training
weights, while autocast still runs supported operations in FP16. Its corrected preflight passed; use
`precision="fp32"` if a later input produces non-finite mixed-precision results.

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

Current result: **15 tests pass locally and on Colab**; the suite includes regressions for loading an
FP16 checkpoint and preserving non-finite AMP diagnostics. The original 12-test suite also passed with
the T4 visible; the final rerun used a CPU subprocess while the real encoders used the GPU. The actual
code cells of all three notebooks also ran with a tiny encoder, including both LR trials, all three
folds, plots, ensemble export and saved-model reload. All three real encoders also completed bounded
two-stage training, Drive export, saved-weight reload and a 477-row submission. Live W&B training and
ensemble uploads passed. Notebook code-cell compilation and lint checks pass.
For resumed runs, `train_minutes` measures the completing session and excludes earlier interrupted
sessions; do not use it as a total GPU-cost estimate.

Notebook 02's saved results were merged from `main` in `ef33ff5`. Its historical runs used the earlier
implementation; the preservation changes do not retroactively create their model exports. Freeze one
Git revision for all new model notebooks.
