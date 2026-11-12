"""Compute spectra and diagnostics for every finished run; write the CSVs in results/.

Uses each run's final-epoch checkpoint (model_last.pth), so test-set checkpoint selection
cannot differ across eta. CSVs are appended run by run to keep memory use low.
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

from capture import CODE, Capture

os.chdir(CODE)

import torch  # noqa: E402

from data_loader.data import Normalizer, data_factory  # noqa: E402
from data_loader.dataset import ClassiregressionDataset, collate_superv  # noqa: E402
from models.model_factory import model_factory  # noqa: E402
from spectra import (LAMBDA_NEAR_ZERO, captured_energy, curves, eval_indices, feature_products,  # noqa: E402
                     ksvd_diagnostics, lambda_match, spectral_metrics)

BATCH = 8
SCATTER_SEQUENCES = 8


def load_data(run_dir, config):
    cls = data_factory[config["data_class"]]
    train = cls(config["data_dir"], pattern=config["pattern"], n_proc=-1, config=config)
    test = cls(config["data_dir"], pattern=config["val_pattern"], n_proc=-1, config=config)
    with open(os.path.join(run_dir, "normalization.pickle"), "rb") as f:
        norm = Normalizer(**pickle.load(f))
    test.feature_df.loc[test.all_IDs] = norm.normalize(test.feature_df.loc[test.all_IDs])
    return train, ClassiregressionDataset(test, test.all_IDs)


def analyse_run(run_dir, summary, train, dataset, device):
    with open(os.path.join(run_dir, "configuration.json")) as f:
        config = json.load(f)
    model = model_factory(config, train)
    state = torch.load(os.path.join(run_dir, "checkpoints", "model_last.pth"), map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    s = summary["s"]
    run_id = os.path.basename(run_dir)
    base = {"run_id": run_id, "eta": summary["eta"], "seed": summary["seed"]}
    indices = eval_indices(len(dataset))
    rows, scatter, sv_store = [], [], {"layer0": [], "layer1": []}

    for start in range(0, len(indices), BATCH):
        batch = indices[start:start + BATCH]
        X, _, masks, _ = collate_superv([dataset[i] for i in batch], max_len=model.max_len)
        with torch.no_grad(), Capture(model) as cap:
            model(X.to(device), masks.to(device))
            d = {k: v.double().cpu() for k, v in cap.d.items() if k not in ("x", "E", "R")}
            fx = cap.fx().double().cpu()  # float64 BEFORE any multiplication

        pq, pk = feature_products(d["phi_q"], d["phi_k"], fx)
        j, e, r, rho, aggregate = ksvd_diagnostics(pq, pk, d["We"], d["Wr"], d["lam"])
        kernel = pq @ pk.transpose(-1, -2)
        sv1 = torch.linalg.svdvals(kernel)
        sv0 = torch.linalg.svdvals(d["attn0"])
        sv_store["layer1"].append(sv1)
        sv_store["layer0"].append(sv0)

        metrics = {"layer1": spectral_metrics(sv1, s), "layer0": spectral_metrics(sv0, s)}
        captured = captured_energy(e, r, kernel)
        n_tokens = pq.shape[2]
        pred, matched, index, rel_error, flagged = lambda_match(d["lam"], sv1, n_tokens)
        relative = rel_error.masked_fill(flagged[None].expand_as(rel_error), float("nan"))

        for b, seq in enumerate(batch):
            for head in range(sv1.shape[1]):
                for layer, values in metrics.items():
                    row = dict(base, seq=seq, head=head, layer=layer)
                    row.update({k: v[b, head].item() for k, v in values.items()})
                    if layer == "layer1":
                        row.update(J=j[b, head].item(), rho_median=rho[b, head].median().item(),
                                   rho_mean=rho[b, head].mean().item(), residual_aggregate=aggregate[b, head].item(),
                                   captured=captured[b, head].item(),
                                   sigma_match_rel_error_median=np.nanmedian(relative[b, head].numpy()),
                                   lambda_near_zero=flagged[head].sum().item())
                    rows.append(row)
        if start < SCATTER_SEQUENCES:
            limit = max(0, min(SCATTER_SEQUENCES - start, len(batch)))
            for b in range(limit):
                for head in range(sv1.shape[1]):
                    for direction in range(s):
                        scatter.append(dict(base, seq=batch[b], head=head, direction=direction,
                                            lam=d["lam"][head, direction].item(),
                                            predicted_sigma=pred[b, head, direction].item(),
                                            nearest_sigma=matched[b, head, direction].item(),
                                            nearest_index=index[b, head, direction].item(),
                                            near_zero=bool(flagged[head, direction])))

    curve_rows = []
    for layer, parts in sv_store.items():
        for name, values in curves(torch.cat(parts).flatten(0, 1)).items():
            mean, std = values.mean(0), values.std(0)
            for k in range(values.shape[1]):
                curve_rows.append(dict(base, layer=layer, curve=name, k=k + 1, mean=mean[k].item(), std=std[k].item()))
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return pd.DataFrame(rows), pd.DataFrame(curve_rows), pd.DataFrame(scatter)


def append(df, path):
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))


def write_summaries(out_dir, run_dirs):
    per_head = pd.read_csv(os.path.join(out_dir, "metrics_per_head.csv"))
    ids = ["run_id", "eta", "seed", "layer", "seq", "head"]
    columns = [c for c in per_head.columns if c not in ids]
    run_means = per_head.groupby(["run_id", "eta", "seed", "layer"])[columns].mean().reset_index()
    run_stds = per_head.groupby(["run_id", "eta", "seed", "layer"])[columns].std().reset_index()
    records = []
    for (eta, layer), group in run_means.groupby(["eta", "layer"]):
        within = run_stds[(run_stds.eta == eta) & (run_stds.layer == layer)]
        for metric in columns:
            if group[metric].notna().any():
                records.append({"eta": eta, "layer": layer, "metric": metric, "mean": group[metric].mean(),
                                "std_over_seeds": group[metric].std(ddof=1), "n_seeds": group[metric].notna().sum(),
                                "mean_within_run_std": within[metric].mean()})
    pd.DataFrame(records).to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    logs = []
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "run_summary.json")) as f:
            summary = json.load(f)
        log = pd.read_csv(os.path.join(run_dir, "training_log.csv"))
        log.insert(0, "seed", summary["seed"])
        log.insert(0, "eta", summary["eta"])
        log.insert(0, "run_id", os.path.basename(run_dir))
        logs.append(log)
    pd.concat(logs).to_csv(os.path.join(out_dir, "training_logs.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    run_dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(os.path.abspath(args.runs_dir), "*", "run_summary.json")))
    if not run_dirs:
        raise SystemExit("No finished runs in {}".format(args.runs_dir))
    for name in ("metrics_per_head.csv", "curves.csv", "lambda_vs_sigma.csv"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(run_dirs[0], "configuration.json")) as f:
        train, dataset = load_data(run_dirs[0], json.load(f))

    summaries = []
    for number, run_dir in enumerate(run_dirs, 1):
        with open(os.path.join(run_dir, "run_summary.json")) as f:
            summary = json.load(f)
        summaries.append(summary)
        metrics, curve_df, scatter = analyse_run(run_dir, summary, train, dataset, device)
        append(metrics, os.path.join(out_dir, "metrics_per_head.csv"))
        append(curve_df, os.path.join(out_dir, "curves.csv"))
        append(scatter, os.path.join(out_dir, "lambda_vs_sigma.csv"))
        print("analysed {}/{}: {}".format(number, len(run_dirs), os.path.basename(run_dir)), flush=True)
    pd.DataFrame(summaries).to_csv(os.path.join(out_dir, "run_summary.csv"), index=False)
    write_summaries(out_dir, run_dirs)
    print("near-zero lambda threshold: {}".format(LAMBDA_NEAR_ZERO))
    print("wrote results to {}".format(out_dir))


if __name__ == "__main__":
    main()
