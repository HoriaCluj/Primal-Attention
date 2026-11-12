"""Figures for the eta ablation. Reads results/ and writes the PNGs listed in the brief.

Bands and error bars are mean +/- one standard deviation over seeds. Each eta keeps one
colour and one line style in every figure; the numbers behind each figure are in results/.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4
STYLES = ["-", (0, (5, 2)), (0, (1, 1.5)), (0, (5, 1.5, 1, 1.5))]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 1.8, "figure.dpi": 150, "savefig.dpi": 200, "axes.axisbelow": True,
})


def style_for(etas):
    return {eta: (COLORS[i % len(COLORS)], STYLES[i % len(STYLES)]) for i, eta in enumerate(sorted(etas))}


def eta_label(eta):
    return "η = {:g}".format(eta)


def band(ax, x, group, color, style, label):
    """Plot mean over seeds with a +/- 1 std band. group is a pivot: rows=x, columns=seed."""
    mean, std = group.mean(axis=1), group.std(axis=1, ddof=1).fillna(0)
    ax.plot(x, mean, color=color, linestyle=style, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def cumulative(curves, layer, path, max_k=None, title=""):
    data = curves[curves.layer == layer]
    if data.empty:
        return
    etas = sorted(data.eta.unique())
    styles = style_for(etas)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ("c", "log10_rel_sigma")):
        for eta in etas:
            frame = data[(data.eta == eta) & (data.curve == name)]
            if max_k:
                frame = frame[frame.k <= max_k]
            pivot = frame.pivot_table(index="k", columns="seed", values="mean").sort_index()
            band(ax, pivot.index, pivot, styles[eta][0], styles[eta][1], eta_label(eta))
        ax.set_xlabel("Singular value index k (1-based)")
    axes[0].set_ylabel("c(k), cumulative raw singular values")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("{}: cumulative spectrum".format(title))
    axes[1].set_ylabel("log10(σ_k / σ_1)")
    axes[1].set_title("Decay on a log scale")
    axes[0].legend(loc="lower right")
    save(fig, path)


def seed_dots(ax, frame, column, etas, color=COLORS[0], offset=0.0, label=None):
    """One dot per seed plus the mean +/- std over seeds at each eta position."""
    for position, eta in enumerate(etas):
        values = frame.loc[frame.eta == eta, column].dropna().values
        if not len(values):
            continue
        x = position + offset
        ax.scatter([x] * len(values), values, s=22, color=color, alpha=0.55, linewidths=0, zorder=2)
        error = values.std(ddof=1) if len(values) > 1 else 0
        ax.errorbar(x, values.mean(), yerr=error, fmt="o", ms=7, color=color, mec=SURFACE, mew=1.2,
                    capsize=3, zorder=3, label=label if position == 0 else None)
    ax.set_xticks(range(len(etas)))
    ax.set_xticklabels(["{:g}".format(eta) for eta in etas])
    ax.set_xlabel("η")
    ax.set_xlim(-0.5, len(etas) - 0.5)


def run_means(per_head):
    numeric = [c for c in per_head.columns if c not in ("run_id", "eta", "seed", "layer", "seq", "head")]
    return per_head.groupby(["eta", "seed", "layer"])[numeric].mean().reset_index()


def ranks(means, path):
    etas = sorted(means.eta.unique())
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    panels = [("layer1", "stable_rank", "Primal kernel K: stable rank"),
              ("layer1", "effective_rank", "Primal kernel K: effective rank"),
              ("layer0", "stable_rank", "Softmax layer 0: stable rank"),
              ("layer0", "effective_rank", "Softmax layer 0: effective rank")]
    for ax, (layer, metric, title) in zip(axes.flat, panels):
        seed_dots(ax, means[means.layer == layer], metric, etas)
        ax.set_title(title)
        ax.set_ylabel(metric.replace("_", " "))
    save(fig, path)


def training(logs, path):
    etas = sorted(logs.eta.unique())
    styles = style_for(etas)
    per_run = logs.groupby(["eta", "seed", "epoch"]).agg(
        Jbar_sq=("Jbar_sq", "first"), J_abs=("J_abs_mean", "mean"), residual=("residual_aggregate", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    panels = [("Jbar_sq", "(head-mean J)²"), ("J_abs", "|J| per head, mean"), ("residual", "Aggregate KKT residual")]
    for ax, (column, label) in zip(axes, panels):
        for eta in etas:
            frame = per_run[per_run.eta == eta]
            pivot = frame.pivot_table(index="epoch", columns="seed", values=column).sort_index()
            band(ax, pivot.index, pivot, styles[eta][0], styles[eta][1], eta_label(eta))
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend(loc="best")
    save(fig, path)


def lambda_sigma(scatter, path):
    data = scatter[(scatter.eta > 0) & (~scatter.near_zero.astype(bool))]
    etas = sorted(data.eta.unique())
    if not etas:
        return
    fig, axes = plt.subplots(1, len(etas), figsize=(4.1 * len(etas), 4.1), sharex=True, sharey=True, squeeze=False)
    for ax, eta in zip(axes[0], etas):
        frame = data[data.eta == eta]
        ax.scatter(frame.predicted_sigma, frame.nearest_sigma, s=7, color=COLORS[0], alpha=0.3, linewidths=0)
        low = min(frame.predicted_sigma.min(), frame.nearest_sigma.min())
        high = max(frame.predicted_sigma.max(), frame.nearest_sigma.max())
        ax.plot([low, high], [low, high], color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(eta_label(eta))
        ax.set_xlabel("N / λ²  (predicted singular value)")
    axes[0][0].set_ylabel("Nearest singular value of K")
    save(fig, path)


def captured(means, path):
    # Both fractions sit within about 1e-6 of 1 because K's spectrum is so steep, so plot the shortfall.
    frame = means[means.layer == "layer1"].copy()
    frame["captured_shortfall"] = (1 - frame.captured).clip(lower=1e-16)
    frame["c2_shortfall"] = (1 - frame.c2_s).clip(lower=1e-16)
    etas = sorted(frame.eta.unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    seed_dots(ax, frame, "captured_shortfall", etas, COLORS[0], -0.08, "Learned directions: 1 − captured")
    seed_dots(ax, frame, "c2_shortfall", etas, COLORS[1], 0.08, "Best possible: 1 − c2(s)")
    ax.set_yscale("log")
    ax.set_ylabel("Fraction of K's energy missed")
    ax.set_title("Energy of K missed by the s directions")
    ax.legend(loc="best")
    save(fig, path)


def accuracy(summary, path):
    etas = sorted(summary.eta.unique())
    frame = summary.copy()
    frame["final"] = frame.accuracy_final * 100
    frame["best"] = frame.accuracy_best * 100
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    seed_dots(ax, frame, "final", etas, COLORS[0], -0.08, "Final epoch")
    seed_dots(ax, frame, "best", etas, COLORS[1], 0.08, "Best epoch (selected on test set)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Test accuracy by η")
    ax.legend(loc="lower right")
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--figures_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)
    read = lambda name: pd.read_csv(os.path.join(args.results_dir, name))
    out = lambda name: os.path.join(args.figures_dir, name)

    curves = read("curves.csv")
    means = run_means(read("metrics_per_head.csv"))
    cumulative(curves, "layer1", out("cumulative_vs_eta_layer1.png"), max_k=64, title="Primal kernel K (layer 1)")
    cumulative(curves, "layer0", out("cumulative_vs_eta_layer0.png"), title="Softmax attention (layer 0)")
    ranks(means, out("stable_effective_rank_vs_eta.png"))
    training(read("training_logs.csv"), out("J_and_residuals_over_training.png"))
    lambda_sigma(read("lambda_vs_sigma.csv"), out("lambda_vs_sigma.png"))
    captured(means, out("captured_vs_c2s.png"))
    accuracy(read("run_summary.csv"), out("accuracy_vs_eta.png"))
    print("wrote figures to {}".format(args.figures_dir))


if __name__ == "__main__":
    main()
