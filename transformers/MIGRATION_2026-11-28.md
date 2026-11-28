# Training data rescue — 2026-11-28

The stopped training runtime's artifacts were copied directly to the destination account's Drive:

`My Drive/dl-a4-ner-migrated-20261128/`

The rescue contains **330 files**, **17,323,786,119 bytes (16.13 GiB)**.
Every copied file passed destination read-back SHA-256 verification. All **17 completion
manifests** validated, and the Drive client successfully flushed all outstanding uploads and unmounted.
A fresh runtime subsequently reopened the destination Drive and confirmed all 330 file identities/sizes
and the checksum records. Both earlier runtimes had ended; the backup remained available from cloud
storage. No model-weight downloads to the laptop were needed for that check.

## Preserved artifacts

- Six completed production runs: four mBERT runs and two mDeBERTa learning-rate trials, including full
  weights, tokenizers, settings, provenance, metrics, histories and saved probabilities.
- The completed mBERT three-fold ensemble, membership/selection records, figures and submissions.
- The interrupted mDeBERTa fold-1 work directory, including optimizer, scheduler, RNG/scaler and custom
  history files. Checkpoints **1750 and 2000** passed their complete-bundle checksums. Checkpoint 2000
  is the latest verified recovery point; partial later saves are preserved but must not be selected.
- The earlier verification experiments and their evidence.
- VM-local W&B transaction records and output logs, including the failure output that was missing from
  the stale Drive log.
- The exact source checkout archive, installed-package inventory and morning audit archive in
  `vm-preservation/`.

The failed run's VM-local output confirms `OSError: [Errno 28] No space left on device` during checkpoint
saving. Removing only replaceable pip and pretrained-model download caches restored source reads before
rescue. Trained files were not removed. Credentials and replaceable download caches are excluded from
this backup.

## Verification records

The destination folder contains `source-inventory.json`, both copy progress records,
`rescue-verification.json` and `migration-complete.json`. The latter records SHA-256 hashes for the
backup files. The cloud copy-verification record was written before the final flush, so its status
captures that phase; the local controller's `final-verification.json` records the successful flush.

Small controller reports and logs are retained locally in the ignored
`transformers/artifacts/migration-20261128/` folder. Model weights were not downloaded to the laptop.
The previous approximately 54 MiB morning results archive omitted models; this rescue includes the
weights and full recovery state.

For recovery, use the unchanged source/input/package settings and the migrated `runs/` directory.
Completed runs can be reused and the interrupted fold can resume from its latest verified checkpoint.
The backup's original `sequence/` record is historical and may still say running; it does not establish
that a training process is active.
