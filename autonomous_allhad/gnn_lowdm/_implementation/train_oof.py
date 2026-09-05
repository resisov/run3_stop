from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import uproot
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ..data import GraphEvents, load_graph_events
    from ..model import GlobalOnlyClassifier, JetGraphClassifier
except ImportError:
    from data import GraphEvents, load_graph_events
    from model import GlobalOnlyClassifier, JetGraphClassifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensors(events: GraphEvents, indices: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(events.node_features[indices]),
        torch.from_numpy(events.node_mask[indices]),
        torch.from_numpy(events.node_eta[indices]),
        torch.from_numpy(events.node_phi[indices]),
        torch.from_numpy(events.global_features[indices]),
        torch.from_numpy(events.labels[indices]),
    )


def predict(
    model: nn.Module,
    dataset: TensorDataset,
    batch_size: int,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or torch.device("cpu")
    model.eval()
    output = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = tuple(item.to(device) for item in batch)
            logits = model(*batch[:-1])
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    positives = labels == 1
    npos = int(np.count_nonzero(positives))
    nneg = len(labels) - npos
    if not npos or not nneg:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return float((ranks[positives].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def roc_points(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)[::-1]
    y = np.asarray(labels, dtype=np.int8)[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    npos = max(int(tp[-1]), 1)
    nneg = max(int(fp[-1]), 1)
    return np.r_[0.0, fp / nneg], np.r_[0.0, tp / npos]


def score_bin_table(labels: np.ndarray, scores: np.ndarray) -> tuple[list[float], list[dict[str, float]]]:
    background = scores[labels == 0]
    edges = np.unique(np.r_[0.0, np.quantile(background, [0.50, 0.80, 0.95, 0.99]), 1.0])
    if len(edges) < 3:
        edges = np.linspace(0.0, 1.0, 6)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (scores >= low) & (scores < high if high < 1.0 else scores <= high)
        signal = int(np.count_nonzero(in_bin & (labels == 1)))
        background_count = int(np.count_nonzero(in_bin & (labels == 0)))
        rows.append({
            "low": float(low),
            "high": float(high),
            "signal_events": signal,
            "background_events": background_count,
            "s_over_sqrt_b": float(signal / math.sqrt(background_count)) if background_count else None,
        })
    return edges.tolist(), rows


def plot_outputs(
    output: Path,
    labels: np.ndarray,
    scores: np.ndarray,
    global_scores: np.ndarray,
    histories: dict[int, list[float]],
) -> None:
    bins = np.linspace(0.0, 1.0, 31)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.hist(scores[labels == 0], bins=bins, density=True, histtype="step", linewidth=2, label="Background MC (OOF)")
    ax.hist(scores[labels == 1], bins=bins, density=True, histtype="step", linewidth=2, label="Signal MC (OOF)")
    ax.set(xlabel="GNN score", ylabel="Normalized events", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "oof_score_distribution.png", dpi=160)
    plt.close(fig)

    fpr, tpr = roc_points(labels, scores)
    global_fpr, global_tpr = roc_points(labels, global_scores)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot(fpr, tpr, linewidth=2, label=f"OOF AUC = {binary_auc(labels, scores):.3f}")
    ax.plot(
        global_fpr,
        global_tpr,
        linewidth=2,
        linestyle=":",
        label=f"Global-only AUC = {binary_auc(labels, global_scores):.3f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5")
    ax.set(xlabel="Background efficiency", ylabel="Signal efficiency", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "oof_roc.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for fold, losses in histories.items():
        ax.plot(np.arange(1, len(losses) + 1), losses, label=f"fold {fold}")
    ax.set(xlabel="Epoch", ylabel="Training BCE")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "training_loss.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a deterministic out-of-fold corridor jet GNN.")
    parser.add_argument("--signal", nargs="+", required=True, type=Path)
    parser.add_argument("--background", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mstop", type=int, default=1050)
    parser.add_argument("--mlsp", type=int, default=900)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-jets", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=24680)
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    seed_everything(opts.seed)

    events = load_graph_events(
        opts.signal,
        opts.background,
        target_mstop=opts.mstop,
        target_mlsp=opts.mlsp,
        max_jets=opts.max_jets,
        folds=opts.folds,
    )
    labels = events.labels.astype(np.int8)
    oof = np.full(len(events), np.nan, dtype=np.float32)
    histories: dict[int, list[float]] = {}
    fold_summaries = []
    start_time = time.time()

    for fold in range(opts.folds):
        seed_everything(opts.seed + fold)
        train_indices = np.flatnonzero(events.fold != fold)
        test_indices = np.flatnonzero(events.fold == fold)
        train_data = tensors(events, train_indices)
        test_data = tensors(events, test_indices)
        generator = torch.Generator().manual_seed(opts.seed + fold)
        loader = DataLoader(
            train_data,
            batch_size=opts.batch_size,
            shuffle=True,
            generator=generator,
        )
        model = JetGraphClassifier(hidden=opts.hidden)
        optimizer = torch.optim.AdamW(model.parameters(), lr=opts.learning_rate, weight_decay=1.0e-4)
        criterion = nn.BCEWithLogitsLoss()
        losses = []
        for _epoch in range(opts.epochs):
            model.train()
            total = 0.0
            seen = 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(*batch[:-1])
                loss = criterion(logits, batch[-1])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch[-1])
                seen += len(batch[-1])
            losses.append(total / max(seen, 1))
        histories[fold] = losses
        oof[test_indices] = predict(model, test_data, opts.batch_size)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "fold": fold,
                "mstop": opts.mstop,
                "mlsp": opts.mlsp,
                "max_jets": opts.max_jets,
                "hidden": opts.hidden,
            },
            opts.output / f"model_fold{fold}.pt",
        )
        fold_summaries.append({
            "fold": fold,
            "train_events": int(len(train_indices)),
            "test_events": int(len(test_indices)),
            "test_signal": int(np.count_nonzero(labels[test_indices] == 1)),
            "test_background": int(np.count_nonzero(labels[test_indices] == 0)),
            "test_auc": binary_auc(labels[test_indices], oof[test_indices]),
            "final_train_bce": losses[-1],
        })

    if not np.all(np.isfinite(oof)):
        raise RuntimeError("out-of-fold inference left non-finite scores")
    global_oof = np.full(len(events), np.nan, dtype=np.float32)
    for fold in range(opts.folds):
        seed_everything(opts.seed + 1000 + fold)
        train_indices = np.flatnonzero(events.fold != fold)
        test_indices = np.flatnonzero(events.fold == fold)
        train_data = tensors(events, train_indices)
        test_data = tensors(events, test_indices)
        generator = torch.Generator().manual_seed(opts.seed + 1000 + fold)
        loader = DataLoader(
            train_data,
            batch_size=opts.batch_size,
            shuffle=True,
            generator=generator,
        )
        baseline = GlobalOnlyClassifier(hidden=opts.hidden)
        optimizer = torch.optim.AdamW(
            baseline.parameters(), lr=opts.learning_rate, weight_decay=1.0e-4
        )
        criterion = nn.BCEWithLogitsLoss()
        final_loss = float("nan")
        for _epoch in range(opts.epochs):
            baseline.train()
            total = 0.0
            seen = 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = baseline(*batch[:-1])
                loss = criterion(logits, batch[-1])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch[-1])
                seen += len(batch[-1])
            final_loss = total / max(seen, 1)
        global_oof[test_indices] = predict(baseline, test_data, opts.batch_size)
        fold_summaries[fold]["global_only_test_auc"] = binary_auc(
            labels[test_indices], global_oof[test_indices]
        )
        fold_summaries[fold]["global_only_final_train_bce"] = final_loss
    if not np.all(np.isfinite(global_oof)):
        raise RuntimeError("global-only out-of-fold inference left non-finite scores")
    edges, bin_rows = score_bin_table(labels, oof)
    plot_outputs(opts.output, labels, oof, global_oof, histories)
    with uproot.recreate(opts.output / "oof_scores.root") as root_file:
        root_file["Events"] = {
            "physical_dataset_id": events.physical_dataset_id,
            "run": events.run,
            "luminosityBlock": events.luminosity_block,
            "event": events.event,
            "is_signal": labels.astype(np.int32),
            "fold": events.fold.astype(np.int32),
            "gnn_score": oof,
            "global_only_score": global_oof,
            "mStop": events.mstop,
            "mLSP": events.mlsp,
            "gen_weight": events.gen_weight,
        }
    summary = {
        "schema_version": "lowdm_corridor_gnn_poc_v1",
        "status": "complete",
        "physics_scope": {
            "domain": "feature_lowdm_preselection && !feature_SR",
            "highdm_exclusivity": "conservative feature_SR veto; not yet the complete adopted High-dM search-bin population veto",
            "signal_mass": {"mStop": opts.mstop, "mLSP": opts.mlsp, "deltaM": opts.mstop - opts.mlsp},
            "classifier_axis": "out-of-fold sigmoid GNN score",
        },
        "training": {
            "folds": opts.folds,
            "epochs": opts.epochs,
            "batch_size": opts.batch_size,
            "max_jets": opts.max_jets,
            "hidden": opts.hidden,
            "learning_rate": opts.learning_rate,
            "seed": opts.seed,
            "loss_weighting": "unweighted proof-of-concept",
            "data_usage": "no collision data used for training",
        },
        "model": {
            "type": "dense permutation-invariant jet graph",
            "node_features": ["log_pt", "eta", "sin_phi", "cos_phi", "log_mass", "btag"],
            "edge_features": ["delta_eta", "sin_delta_phi", "cos_delta_phi", "delta_r_proxy"],
            "message_layers": 2,
            "global_control": "MLP using MET, HT, Njet, Nb, and mass-hypothesis features",
        },
        "provenance": {
            "signal_files": [
                {"path": str(path), "sha256": sha256(path)} for path in opts.signal
            ],
            "background_files": [
                {"path": str(path), "sha256": sha256(path)} for path in opts.background
            ],
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "uproot_version": uproot.__version__,
        },
        "events": {
            "total": len(events),
            "signal": int(np.count_nonzero(labels == 1)),
            "background": int(np.count_nonzero(labels == 0)),
        },
        "metrics": {
            "oof_auc": binary_auc(labels, oof),
            "global_only_oof_auc": binary_auc(labels, global_oof),
            "gnn_auc_gain_over_global": binary_auc(labels, oof) - binary_auc(labels, global_oof),
            "score_min": float(np.min(oof)),
            "score_median": float(np.median(oof)),
            "score_max": float(np.max(oof)),
        },
        "folds": fold_summaries,
        "score_bins": {"edges": edges, "unweighted_poc_yields": bin_rows},
        "runtime_seconds": time.time() - start_time,
        "artifacts": {
            "scores": "oof_scores.root",
            "score_plot": "oof_score_distribution.png",
            "roc_plot": "oof_roc.png",
            "loss_plot": "training_loss.png",
            "models": [f"model_fold{fold}.pt" for fold in range(opts.folds)],
        },
        "limitations": [
            "This is an unweighted small-sample classifier test, not an expected-limit result.",
            "Only nominal kinematics are scored; shape-systematic migrations are not yet evaluated.",
            "The background pilot uses a small TT-dominated shard rather than the complete process mixture.",
            "The intermediate schema supports a jet-level graph but has no PF candidates or MET covariance.",
        ],
    }
    (opts.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", **summary["events"], **summary["metrics"], "output": str(opts.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
