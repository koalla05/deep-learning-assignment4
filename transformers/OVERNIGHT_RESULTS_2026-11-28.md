# Overnight training: morning verification

The queue started at 01:23 UTC on 2026-11-28 with training revision
`a358886abab06f47fe87f443441c07db938d177c` on the KSE account's T4. This is a results snapshot,
not a declaration that the queue has finished. At 07:40 UTC, the GPU was 97% utilized, the active
mDeBERTa log was one second old, and training had reached step 1,057/2,642 of the next fold's sentence
stage. No restart or new training job was launched during this audit.

## Completion and preservation

| Model | Status at the morning check | Weights / inference / submission |
| --- | --- | --- |
| mBERT | Both LR trials, selected folds 0–2 and ensemble completed | Four Drive model bundles verified, saved-weight reload passed in the notebook, ensemble submission verified |
| mDeBERTa v3 base | Both fold-0 LR trials completed; remaining fold training active | First trial's full bundle checksum-verified; second trial has a completion manifest, matching metrics checksum and successful W&B upload; final ensemble/reload still pending |
| XLM-R large | Queued; not started | No full-training results yet |

The queue was still running. Incomplete work is not classified as a failure. The inspected logs contain
no OOM or training exception. The audit did not stop the runtime, alter the training source, or change
the queue. Colab's lifetime/unit limits can still interrupt the remaining work; this runtime had
already used over six hours at the activity check. Full completion in the original runtime remains
uncertain.

Remote checksum verification covered all four completed mBERT run bundles, the first completed
mDeBERTa run, and the mBERT ensemble, including their model files. Every audited submission has the
expected 477 unique test IDs in the expected order. All six audited bundles recorded successful W&B
uploads. The second mDeBERTa trial completed after the full audit's directory snapshot; its metrics
were downloaded separately and checked against its completion manifest. Its entire weight file was
not rehashed in that snapshot.

## Scores

These are local F1 scores, not Kaggle leaderboard scores. Half A/B use teacher pseudo-labels;
half B is the development set used for LR selection. Gold26 is a small human-labeled reporting set.

| Run | Half A | Half B | Gold26 |
| --- | ---: | ---: | ---: |
| mBERT fold 0, LR 3e-5 | 0.9091 | 0.8690 | 0.8108 |
| mBERT fold 0, LR 5e-5 | 0.8952 | 0.8712 | 0.8267 |
| mBERT fold 1, LR 5e-5 | 0.8807 | 0.8719 | 0.8052 |
| mBERT fold 2, LR 5e-5 | 0.8886 | 0.8653 | 0.6944 |
| **mBERT three-fold ensemble** | **0.8905** | **0.8997** | **0.8158** |
| mDeBERTa fold 0, LR 3e-5 | 0.8923 | 0.8502 | 0.8052 |
| mDeBERTa fold 0, LR 5e-5 | 0.9054 | 0.8498 | 0.8267 |

mBERT selected LR 5e-5. Its selected folds have mean sentence CV F1 0.8978 (sample SD 0.0085), and
mean half-B F1 0.8694. Averaging their token probabilities improves half-B F1 to 0.8997. The mDeBERTa
trial scores favor 3e-5 by only 0.00045; this small gap is not evidence of a reliable LR advantage.
Compare its final ensemble once available rather than ranking its single fold against an ensemble.
Gold26 varies substantially across folds and contains only 26 posts, so avoid strong conclusions from
that score alone. No Kaggle submission was made or leaderboard score recorded by this audit.

## Local results and independent checks

Downloaded results live in the ignored local folder
`transformers/artifacts/overnight-20261128/`:

- `audit.json`: remote full-bundle checksums, metrics, upload records and submission identity checks.
- `completed-results.zip`: approximately 54 MiB; metrics, histories, probabilities, row identities,
  decoding settings, submissions, ensemble membership/selection, figures and logs. Large model files
  and recovery checkpoints remain on Drive.
- `results/`: extracted archive, with its original `runs/`, `ensembles/` and `sequence/` structure.
- `local-verification.json`: independent local verification results.
- `mdeberta-lr5e-05-*.json`: the second trial's later metrics, upload status and completion manifest.

Local verification passed archive CRC and every downloaded file's manifest checksum. Across more
than five million token probability rows, values were finite, bounded and normalized; character
offsets were within their texts. Re-decoding saved probabilities reproduced all six audited
submissions with **zero differing posts**. Re-averaging the selected mBERT folds reproduced the stored
ensemble probabilities exactly. This also confirms the local copies are usable for postprocessing
and later ensemble work. These extracted folders omit model files, so they are analysis copies rather
than complete inference bundles; load saved models from Drive.

Preferred completed Kaggle file:
`results/ensembles/05-mbert-81e05d78d61e/submission.csv` in the local download folder, or the equivalent
`MyDrive/dl-a4-ner/ensembles/05-mbert-81e05d78d61e/submission.csv` on Drive.

## W&B evidence

- [mBERT ensemble](https://wandb.ai/models-kyiv-school-of-economics/dl-a4-ner/runs/f1qkd6kc)
- [mBERT selected fold 0](https://wandb.ai/models-kyiv-school-of-economics/dl-a4-ner/runs/vb8udgh4)
- [mDeBERTa LR 3e-5 trial](https://wandb.ai/models-kyiv-school-of-economics/dl-a4-ner/runs/5fb7qzwz)
- [mDeBERTa LR 5e-5 trial](https://wandb.ai/models-kyiv-school-of-economics/dl-a4-ner/runs/c8502fkf)

Upload status comes from completed bundle records and run logs. W&B's API was not queried again in
this audit. The browser notebook-name warning in the CLI log affects automatic code capture; it did
not prevent training metrics or ensemble artifacts from uploading.
