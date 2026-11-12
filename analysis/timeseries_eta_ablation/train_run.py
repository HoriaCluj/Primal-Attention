"""Train one sweep run with the repo's unmodified main.py logic.

main.main() has no callback, so two functions are wrapped at runtime to add logging:
`main.validate` (accuracy, J and residuals on fixed test sequences) and
`SupervisedRunner.train_epoch` (timing). The diagnostics draw no random numbers.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

from capture import CODE, Capture

os.chdir(CODE)

import torch  # noqa: E402

DATASETS = {
    # Values copied from bashes/run_primal_trans.sh; eta is the only value the sweep changes.
    "UWaveGestureLibrary": {"low_rank": 40, "rank_multi": 10, "optimizer": "RAdam"},
    "Handwriting": {"low_rank": 20, "rank_multi": 5, "optimizer": "RAdam"},
}
DIAGNOSTIC_SEQUENCES = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="UWaveGestureLibrary", choices=DATASETS)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs_dir", required=True)
    return parser.parse_args()


def repo_argv(a):
    d = DATASETS[a.dataset]
    return ["main.py",
            "--output_dir", a.runs_dir,
            "--comment", "eta ablation",
            "--name", "{}_seed{}".format(a.dataset, a.seed),
            "--records_file", os.path.join(a.runs_dir, "records_{}_eta{}_seed{}.xls".format(a.dataset, a.eta, a.seed)),
            "--data_dir", "Multivariate_ts/{}".format(a.dataset),
            "--data_class", "tsra", "--pattern", "TRAIN", "--val_pattern", "TEST",
            "--epochs", str(a.epochs), "--lr", "0.001", "--batch_size", "16",
            "--optimizer", d["optimizer"], "--pos_encoding", "learnable",
            "--task", "classification", "--key_metric", "accuracy", "--model", "primal_cos",
            "--low_rank", str(d["low_rank"]), "--eta", str(a.eta), "--rank_multi", str(d["rank_multi"]),
            "--seed", str(a.seed)]


def main():
    a = parse_args()
    a.runs_dir = os.path.abspath(a.runs_dir)
    os.makedirs(a.runs_dir, exist_ok=True)
    pattern = "{}_seed{}_primal_cos_eta{}_*/run_summary.json".format(a.dataset, a.seed, a.eta)
    if glob.glob(os.path.join(a.runs_dir, pattern)):
        print("already finished, skipping")
        return

    sys.argv = repo_argv(a)
    import main as repo_main
    import running
    from options import Options
    from spectra import eval_indices, feature_products, ksvd_diagnostics

    repo_args = Options().parse()
    repo_main.args = repo_args  # main.py reads this global, normally set under __main__
    config = running.setup(repo_args)
    run_dir = config["output_dir"]

    history, epoch_times = [], []
    log_path = os.path.join(run_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["epoch", "head", "J_mean", "J_abs_mean", "Jbar_sq", "rho_mean", "rho_median", "residual_aggregate",
                  "test_loss", "test_accuracy"])

    def log_diagnostics(evaluator, epoch, metrics):
        model, device = evaluator.model, evaluator.device
        dataset, collate = evaluator.dataloader.dataset, evaluator.dataloader.collate_fn
        indices = eval_indices(len(dataset))[:DIAGNOSTIC_SEQUENCES]
        model.eval()
        js, rhos, aggregates = [], [], []
        with torch.no_grad():
            for start in range(0, len(indices), 16):
                X, _, masks, _ = collate([dataset[i] for i in indices[start:start + 16]])
                with Capture(model, softmax=False) as cap:
                    model(X.to(device), masks.to(device))
                    d = cap.d
                    pq, pk = feature_products(d["phi_q"], d["phi_k"], cap.fx())
                    j, _, _, rho, aggregate = ksvd_diagnostics(pq, pk, d["We"], d["Wr"], d["lam"])
                js.append(j.cpu())
                rhos.append(rho.cpu())
                aggregates.append(aggregate.cpu())
        j, rho, aggregate = torch.cat(js), torch.cat(rhos), torch.cat(aggregates)
        jbar_sq = (j.mean(1) ** 2).mean().item()
        for head in range(j.shape[1]):
            log.writerow([epoch, head, j[:, head].mean().item(), j[:, head].abs().mean().item(), jbar_sq,
                          rho[:, head].mean().item(), rho[:, head].median().item(), aggregate[:, head].mean().item(),
                          metrics["loss"], metrics["accuracy"]])
        log_file.flush()

    original_validate = repo_main.validate

    def validate(val_evaluator, writer, cfg, best_metrics, best_value, epoch):
        output = original_validate(val_evaluator, writer, cfg, best_metrics, best_value, epoch)
        metrics = dict(output[0])
        history.append((epoch, metrics["accuracy"]))
        log_diagnostics(val_evaluator, epoch, metrics)
        return output

    repo_main.validate = validate

    original_train_epoch = running.SupervisedRunner.train_epoch

    def train_epoch(self, epoch_num=None):
        start = time.time()
        output = original_train_epoch(self, epoch_num)
        epoch_times.append(time.time() - start)
        return output

    running.SupervisedRunner.train_epoch = train_epoch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    repo_main.main(config)
    log_file.close()

    best_epoch, best_accuracy = max(history, key=lambda item: item[1])
    d = DATASETS[a.dataset]
    summary = {
        "dataset": a.dataset, "eta": a.eta, "seed": a.seed, "epochs": a.epochs,
        "s": d["low_rank"], "rank_multi": d["rank_multi"],
        "accuracy_final": history[-1][1], "accuracy_best": best_accuracy, "best_epoch": best_epoch,
        "seconds_per_epoch": sum(epoch_times) / len(epoch_times),
        "total_minutes": (time.time() - started) / 60,
        "peak_memory_MiB": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    # Written last: its presence is what marks a run as finished.
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
