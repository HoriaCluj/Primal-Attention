"""Figures and summary table from figure1_spectra.py outputs (CPU, seconds; needs only numpy and matplotlib).

    python figure1_plots.py --spectra <dir with *.npz> --out analysis/imagenet_eta_ablation \
        [--logs analysis/imagenet_eta_ablation/results/raw_logs] [--curve c]
    python figure1_plots.py --logs <raw_logs> --out <dir>      # training curves only, no spectra needed

Writes, under --out:
  figures/fig1_<run>.png          the paper's layout for one Primal.+ run against DeiT-S: (a) layers 0-3, (b) 4-7,
                                  (c) 8-11 with each model's L[11] dashed, mean +- std over layers, images and heads;
                                  (d) heatmap of the run's mean curve per layer
  figures/per_layer_<curve>.png   one panel per layer, one curve per model (mean +- std over images and heads)
  figures/primal_candidates.png   the four Primal-layer matrices (K_full, E Lambda R^T, E Lambda^2/N R^T, softmax(qk^T/8))
  figures/ranks_by_layer.png      stable and effective rank per layer and model
  figures/training_curves.png     (with --logs) test top-1, mean |J| and KKT residual over epochs
  results/spectra_summary.csv     per model, layer and matrix: curve values and rank metrics (mean, std)
The Primal layer enters the eta comparison as K_full (--primal-matrix to change). --curve c is the paper's
normalized cumulative singular value; c2 uses squared singular values.
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ETA_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # fixed categorical order, by increasing eta
DEIT_STYLE = {"color": "#52514e", "linestyle": "--"}                    # neutral reference, dashed as second cue
DEIT_OFFICIAL_STYLE = {"color": "#8a8986", "linestyle": ":"}
CANDIDATES = [("K_full", "K = Φq F_Xᵀ F_X Φkᵀ", "#2a78d6", "-"), ("E_L_Rt", "E Λ Rᵀ", "#eb6834", "-"),
              ("E_L2N_Rt", "E (Λ²/N) Rᵀ", "#eb6834", "--"), ("softmax_qk", "softmax(q kᵀ/8)", "#1baf7a", "-")]
METRICS = ["stable_rank", "effective_rank", "rank_1e-10", "c_at_0"]


class Run:
    def __init__(self, path, primal_matrix):
        self.z = np.load(path)
        self.meta = json.loads(str(self.z["meta"]))
        self.name = self.meta["name"]
        self.primal_layer = self.meta["primal_layer"]
        self.primal_matrix = primal_matrix
        self.eta = self.meta["eta"]
        if self.meta["model"] == "primal_plus":
            self.label, self.style = "Primal.+ η = {:g}".format(self.eta), {}
        elif self.meta["model"] == "deit":
            self.label, self.style = "DeiT-S", dict(DEIT_STYLE)
        else:
            self.label, self.style = "DeiT-S (official, IN-1K)", dict(DEIT_OFFICIAL_STYLE)
        self.layers = sorted(self.meta["softmax_layers"] + ([self.primal_layer] if self.primal_layer is not None else []))

    def key(self, layer, matrix=None):
        if matrix is None:
            matrix = self.primal_matrix if layer == self.primal_layer else "softmax"
        return "L{:02d}.{}".format(layer, matrix)

    def curve(self, layers, curve, matrix=None):
        """Mean and std over images, heads and the given layers (pooled)."""
        s = q = n = 0
        for layer in layers:
            k = self.key(layer, matrix)
            s = s + self.z[k + "." + curve + ".sum"].sum(0)
            q = q + self.z[k + "." + curve + ".sumsq"].sum(0)
            n += int(self.z[k + ".n_images"]) * self.z[k + "." + curve + ".sum"].shape[0]
        mean = s / n
        return mean, np.sqrt(np.maximum(q / n - mean ** 2, 0))

    def metric(self, layer, name, matrix=None):
        return self.z["{}.{}.per_image".format(self.key(layer, matrix), name)]


def load_runs(spectra_dir, primal_matrix):
    runs = [Run(p, primal_matrix) for p in sorted(glob.glob(os.path.join(spectra_dir, "*.npz")))]
    primal = sorted([r for r in runs if r.eta is not None], key=lambda r: r.eta)
    for r, color in zip(primal, ETA_COLORS):
        r.style = {"color": color, "linestyle": "-"}
    if len(primal) > len(ETA_COLORS):
        raise SystemExit("more eta values than colors; facet instead")
    others = [r for r in runs if r.eta is None]
    return primal, others


def style_axes(ax, xlabel="Singular value index", ylabel=None, ylim=(0, 1.02)):
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


def band(ax, x, mean, std, style, label, dashed=False, alpha=0.15):
    st = dict(style)
    if dashed:
        st["linestyle"] = "--" if st.get("linestyle", "-") == "-" else ":"
    ax.plot(x, mean, linewidth=2, label=label, **st)
    if std is not None:
        ax.fill_between(x, np.clip(mean - std, 0, None), np.minimum(mean + std, 1.0), color=st["color"], alpha=alpha,
                        linewidth=0)


def fig1(run, reference, curve, path):
    """The paper's 4-panel layout for one model, against a reference (DeiT-S) if available."""
    ylabel = {"c": "Normalized cumulative\nsingular value", "c2": "Cumulative energy (σ²)"}[curve]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    for ax, layers in zip(axes[:3], groups):
        for model in [m for m in (reference, run) if m is not None]:
            mean, std = model.curve(layers, curve)
            x = np.arange(len(mean))
            band(ax, x, mean, std, model.style, model.label)
            if layers[-1] == 11:
                mean11, _ = model.curve([11], curve)
                band(ax, x, mean11, None, model.style, model.label + ", L[11]", dashed=True)
        ax.set_title("Layers {}".format(layers))
        style_axes(ax, ylabel=ylabel if layers[0] == 0 else None)
        ax.set_xticks([0, 50, 100, 150, 196])
        ax.legend(loc="lower right", frameon=False)
    heat = np.stack([run.curve([layer], curve)[0] for layer in range(12)])
    im = axes[3].imshow(heat, aspect="auto", origin="lower", cmap="Blues_r", vmin=0, vmax=1, interpolation="nearest")
    axes[3].set_yticks(range(12))
    axes[3].set_xticks([0, 50, 100, 150, 196])
    axes[3].set_xlabel("Singular value index")
    axes[3].set_ylabel("Layer index")
    axes[3].set_title(run.label)
    fig.colorbar(im, ax=axes[3])
    note = "L[11] of Primal.+ = {}; mean ± std over layers, {} images and heads".format(
        run.primal_matrix if run.primal_layer is not None else "softmax", run.meta["n_images"])
    fig.text(0.01, 0.005, note, fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_layer(models, curve, path):
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True, sharey=True)
    for layer, ax in enumerate(axes.flat):
        for model in models:
            mean, std = model.curve([layer], curve)
            band(ax, np.arange(len(mean)), mean, std, model.style, model.label, alpha=0.08)
        title = "Layer {}".format(layer)
        primal_here = [m for m in models if m.primal_layer == layer]
        if primal_here:
            title += " (Primal.+: {})".format(primal_here[0].primal_matrix)
        ax.set_title(title)
        style_axes(ax, xlabel="Singular value index" if layer >= 8 else "", ylabel=curve if layer % 4 == 0 else None)
    axes.flat[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def primal_candidates(primal, reference, curve, path):
    fig, axes = plt.subplots(1, len(primal), figsize=(5.2 * len(primal), 4.2), squeeze=False, sharey=True)
    for ax, run in zip(axes[0], primal):
        layer = run.primal_layer
        for matrix, label, color, ls in CANDIDATES:
            mean, std = run.curve([layer], curve, matrix)
            band(ax, np.arange(len(mean)), mean, std, {"color": color, "linestyle": ls}, label, alpha=0.08)
        if reference is not None:
            mean, _ = reference.curve([layer], curve)
            ax.plot(np.arange(len(mean)), mean, linewidth=1.5, label="DeiT-S L[{}]".format(layer), **reference.style)
        ax.set_title("{}: layer {}".format(run.label, layer))
        style_axes(ax, ylabel=curve if ax is axes[0][0] else None)
    axes[0][0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def ranks_by_layer(models, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, name, title in zip(axes, ["stable_rank", "effective_rank"],
                               ["Stable rank Σσ²/σ₁²", "Effective rank exp(entropy of σ/Σσ)"]):
        for model in models:
            values = [model.metric(layer, name) for layer in model.layers]
            ax.plot(model.layers, [v.mean() for v in values], marker="o", markersize=5, linewidth=2,
                    label=model.label, **model.style)
        ax.set_title(title)
        style_axes(ax, xlabel="Layer", ylim=None)
        ax.set_xticks(range(12))
    axes[1].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.005, "Mean over images and heads; the Primal layer (11) uses K_full, whose rank is at most 64.",
             fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def styles_from_names(names):
    """Style and label per run folder name: primal_eta<eta>_seed<s> or deit_seed<s>."""
    etas = sorted({float(n.split("_eta")[1].split("_seed")[0]) for n in names if "_eta" in n})
    out = {}
    for n in names:
        if "_eta" in n:
            eta = float(n.split("_eta")[1].split("_seed")[0])
            out[n] = ({"color": ETA_COLORS[etas.index(eta)], "linestyle": "-"}, "Primal.+ η = {:g}".format(eta))
        elif n.startswith("deit"):
            out[n] = (dict(DEIT_STYLE), "DeiT-S")
    return out


def training_curves(logs_dir, path):
    runs = sorted(d for d in glob.glob(os.path.join(logs_dir, "*")) if os.path.isfile(os.path.join(d, "log.txt")))
    if not runs:
        return
    styles = styles_from_names([os.path.basename(d) for d in runs])
    runs.sort(key=lambda d: (os.path.basename(d) in styles and styles[os.path.basename(d)][1] != "DeiT-S",
                             styles.get(os.path.basename(d), ({}, ""))[1]))   # DeiT-S first, then by eta
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
    for d in runs:
        name = os.path.basename(d)
        style, label = styles.get(name, ({"color": "#8a8986"}, name))
        rows = [json.loads(line) for line in open(os.path.join(d, "log.txt"))]
        axes[0].plot([r["epoch"] for r in rows], [r["test_acc1"] for r in rows], linewidth=1.5, label=label, **style)
        jpath = os.path.join(d, "J_log.csv")
        if os.path.isfile(jpath):
            by_epoch = {}
            for row in csv.DictReader(open(jpath)):
                by_epoch[(int(row["epoch"]), int(row["head"]))] = row          # last row per (epoch, head) wins
            epochs = sorted({e for e, _ in by_epoch})
            j_abs = [np.mean([abs(float(by_epoch[(e, h)]["J_mean"])) for (ee, h) in by_epoch if ee == e]) for e in epochs]
            resid = [np.mean([float(by_epoch[(e, h)]["residual_aggregate"]) for (ee, h) in by_epoch if ee == e]) for e in epochs]
            axes[1].plot(epochs, j_abs, linewidth=1.5, label=label, **style)
            axes[2].plot(epochs, resid, linewidth=1.5, label=label, **style)
    titles = ["Test top-1 (%)", "|J| per head, mean over heads (log)", "KKT residual (||G||²/||W||², log)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        style_axes(ax, xlabel="Epoch", ylim=None)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def summary_csv(models, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "model", "eta", "layer", "matrix", "n_images", "c_at_10", "c_at_63", "c2_at_10",
                    "index_c_0.9", "index_c_0.99"] + [m + s for m in METRICS for s in ("_mean", "_std")])
        for model in models:
            for layer in model.layers:
                matrices = ["softmax"] if layer != model.primal_layer else [c[0] for c in CANDIDATES]
                for matrix in matrices:
                    c, _ = model.curve([layer], "c", matrix)
                    c2, _ = model.curve([layer], "c2", matrix)
                    row = [model.name, model.meta["model"], model.eta, layer, matrix, model.meta["n_images"],
                           round(c[10], 5), round(c[63], 5), round(c2[10], 5),
                           int(np.argmax(c >= 0.9)), int(np.argmax(c >= 0.99))]
                    for m in METRICS:
                        v = model.metric(layer, m, matrix)
                        row += [round(float(v.mean()), 5), round(float(v.std()), 5)]
                    w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spectra", default="", help="folder with figure1_spectra.py outputs (*.npz)")
    parser.add_argument("--out", required=True, help="writes <out>/figures and <out>/results")
    parser.add_argument("--logs", default="", help="optional: folder of <run>/log.txt (+ J_log.csv)")
    parser.add_argument("--curve", default="c", choices=["c", "c2"])
    parser.add_argument("--primal-matrix", default="K_full", choices=[c[0] for c in CANDIDATES])
    cli = parser.parse_args()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.edgecolor": "#8a8986",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})
    figures, results = os.path.join(cli.out, "figures"), os.path.join(cli.out, "results")
    os.makedirs(figures, exist_ok=True)
    os.makedirs(results, exist_ok=True)

    primal, others = load_runs(cli.spectra, cli.primal_matrix) if cli.spectra else ([], [])
    reference = next((r for r in others if r.meta["model"] == "deit"), None)
    models = ([reference] if reference else []) + primal
    suffix = "" if cli.curve == "c" else "_c2"
    for run in primal:
        fig1(run, reference, cli.curve, os.path.join(figures, "fig1_{}{}.png".format(run.name, suffix)))
    for run in others:
        fig1(run, None, cli.curve, os.path.join(figures, "fig1_{}{}.png".format(run.name, suffix)))
    if models:
        per_layer(models, cli.curve, os.path.join(figures, "per_layer_{}.png".format(cli.curve)))
        ranks_by_layer(models, os.path.join(figures, "ranks_by_layer.png"))
    if primal:
        primal_candidates(primal, reference, cli.curve, os.path.join(figures, "primal_candidates{}.png".format(suffix)))
    if primal or others:
        summary_csv(primal + others, os.path.join(results, "spectra_summary.csv"))
    if cli.logs:
        training_curves(cli.logs, os.path.join(figures, "training_curves.png"))
    print("wrote", figures, "and", results)


if __name__ == "__main__":
    main()
