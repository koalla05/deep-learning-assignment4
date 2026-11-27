"""Optimization curves and gradient flow of saved runs (see train.load_run)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_curves(runs, title=None):
    """Training loss, held-out losses and post-level F1 per epoch for one or several runs ({label: run})."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for label, run in runs.items():
        log = pd.DataFrame(run["history"]["trainer"])
        offset = _stage_offsets(log)
        if "loss" in log:  # missing when a run is shorter than one logging interval
            train = log.dropna(subset=["loss"])
            axes[0].plot(train.step + train.stage.map(offset), train.loss.rolling(5, min_periods=1).mean(), label=label)
        for col, style in [("eval_sentences_loss", "-"), ("eval_synthetic_loss", "--")]:
            if col in log:
                ev = log.dropna(subset=[col])
                axes[1].plot(ev.step + ev.stage.map(offset), ev[col], style, marker="o", label=f"{label}, {col.split('_')[1]}")
        post = pd.DataFrame(run["history"]["post_eval"])
        if len(post):
            post["x"] = np.arange(1, len(post) + 1)
            for col, style in [("half_B_f1", "-"), ("cv_f1", "--"), ("gold26_f1", ":")]:
                if col in post:
                    axes[2].plot(post.x, post[col], style, marker="o", label=f"{label}, {col[:-3]}")
    axes[0].set(title="training loss (smoothed)", xlabel="step")
    axes[1].set(title="held-out token loss", xlabel="step")
    axes[2].set(title="competition F1 after each epoch", xlabel="evaluation (epoch)")
    for ax in axes:
        ax.legend(fontsize=7)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def _stage_offsets(log):
    """Global step offset per training stage, so a second stage continues the x axis."""
    if "stage" not in log:
        log["stage"] = 1
    offsets, total = {}, 0
    for stage, part in log.groupby("stage"):
        offsets[stage] = total
        total += part.step.max()
    return offsets


def plot_grad_flow(run, title=None):
    """Heatmap of gradient norm per layer group (rows) over training steps (columns), log scale."""
    df = pd.DataFrame(run["grad_flow"]).set_index("step")
    layers = sorted(c for c in df.columns if c.startswith("layer_"))
    order = ["head"] + layers[::-1] + ["embeddings"]
    df = df[[c for c in order if c in df.columns]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]})
    im = axes[0].imshow(np.log10(df.T.values + 1e-12), aspect="auto", cmap="viridis")
    axes[0].set_yticks(range(len(df.columns)), df.columns, fontsize=7)
    ticks = np.linspace(0, len(df) - 1, min(8, len(df))).astype(int)
    axes[0].set_xticks(ticks, df.index[ticks])
    axes[0].set(xlabel="step", title="log10 gradient norm per layer")
    fig.colorbar(im, ax=axes[0])
    axes[1].barh(range(len(df.columns)), df.median().values)
    axes[1].set_yticks(range(len(df.columns)), df.columns, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set(xscale="log", title="median over training")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def results_table(runs, sets=("cv", "synthetic", "half_B", "half_A", "gold26"), leaderboard=None):
    """One row per run: F1 per evaluation set, training time and, if given, the public leaderboard score.

    leaderboard: path of a CSV with columns run, public_lb (transformers/leaderboard.csv), filled in by hand
    after each Kaggle submission.
    """
    rows = {}
    for label, run in runs.items():
        m = run["metrics"]
        rows[label] = {s: m[s]["f1"] for s in sets if s in m} | {"minutes": m.get("train_minutes")}
    table = pd.DataFrame(rows).T
    if leaderboard is not None:
        try:
            lb = pd.read_csv(leaderboard).set_index("run").public_lb
            table["public_lb"] = [lb.get(r["config"]["name"]) for r in runs.values()]
        except FileNotFoundError:
            pass
    return table
