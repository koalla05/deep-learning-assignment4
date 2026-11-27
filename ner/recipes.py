"""The current best recipe, shared by the notebooks.

Update this file after each experiment notebookЖ

1. after 01_data_mix: set DATA (and STAGE2 if the two-stage run wins);
2. after 02_optimization: set OPTIMIZATION;
3. the model notebooks (03-05) take both and only change what is model-specific.
"""

from .data import DataSpec

# 01_data_mix: two stages won on half_B (0.872). Stage 1 = all Kaggle sentences for OPTIMIZATION["epochs"],
# stage 2 = all synthetic posts for 2 epochs at lr 2e-5.
DATA = DataSpec(synthetic="none")
STAGE2 = {"stage2_data": DataSpec(sentences=0), "stage2_epochs": 2, "stage2_lr": 2e-5}

# 02_optimization decides this. Defaults: the usual fine-tuning setup for base-size encoders.
OPTIMIZATION = dict(
    optimizer="adamw",
    learning_rate=3e-5,
    scheduler="linear",
    warmup_ratio=0.1,
    weight_decay=0.01,
    layer_lr_decay=1.0,
    epochs=2,  # stage 1 (sentences); stage 2 is set in STAGE2
    batch_size=32,
    label_smoothing=0.0,
    max_length=256,
    stride=64,
    label_all_subtokens=True,
)
