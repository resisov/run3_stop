from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import torch
import uproot
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ..data import GraphEvents, load_graph_events
    from ..model import (
        FlattenDNNClassifier,
        GlobalOnlyClassifier,
        JetGraphClassifier,
        JetTransformerClassifier,
    )
    from .train_oof import (
        binary_auc,
        predict,
        roc_points,
        score_bin_table,
        seed_everything,
        sha256,
        tensors,
    )
except ImportError:
    from data import GraphEvents, load_graph_events
    from model import (
        FlattenDNNClassifier,
        GlobalOnlyClassifier,
        JetGraphClassifier,
        JetTransformerClassifier,
    )
    from train_oof import (
        binary_auc,
        predict,
        roc_points,
        score_bin_table,
        seed_everything,
        sha256,
        tensors,
    )


ModelFactory = Callable[[], nn.Module]
PROCESS_NAMES = {
    282677918: "TT",
    997207294: "Zto2Nu",
    2093051268: "WtoLNu",
}


def normalized_physics_weights(
    events: GraphEvents,
    normalization: dict[str, object],
) -> np.ndarray:
    """Signed xsec weights, extrapolated from the selected source shards."""
    luminosity_pb = float(normalization["luminosity_pb"])
    by_id = normalization["by_physical_dataset_id"]
    output = np.asarray(events.gen_weight * events.sampling_weight, dtype=np.float64)
    background = events.labels == 0
    for physical_id in np.unique(events.physical_dataset_id[background]):
        record = by_id.get(str(int(physical_id)))
        if record is None or not record.get("normalization_complete"):
            raise RuntimeError(f"missing complete xsec normalization for physical_dataset_id={physical_id}")
        selected_sumw = float(record["selected_shard_sumw"])
        if selected_sumw == 0.0:
            raise RuntimeError(f"zero selected-shard sumw for physical_dataset_id={physical_id}")
        selected = background & (events.physical_dataset_id == physical_id)
        output[selected] *= float(record["xsec_pb"]) * luminosity_pb / selected_sumw
    return output


def cross_section_training_weights(
    events: GraphEvents,
    indices: np.ndarray,
    physics_weights: np.ndarray,
) -> np.ndarray:
    """Balance S/B while preserving the xsec-weighted background mixture."""
    labels = events.labels[indices].astype(np.int8)
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise RuntimeError("each training fold needs signal and background events")
    weights = np.abs(np.asarray(physics_weights[indices], dtype=np.float64))
    signal = labels == 1
    background = ~signal
    if weights[signal].sum() == 0.0 or weights[background].sum() == 0.0:
        raise RuntimeError("zero absolute physics weight in a training class")
    weights[signal] *= 0.5 / weights[signal].sum()
    weights[background] *= 0.5 / weights[background].sum()
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def weighted_tensors(
    events: GraphEvents,
    indices: np.ndarray,
    physics_weights: np.ndarray,
) -> torch.utils.data.TensorDataset:
    base = tensors(events, indices)
    weights = torch.from_numpy(cross_section_training_weights(events, indices, physics_weights))
    return torch.utils.data.TensorDataset(*base.tensors, weights)


def weighted_binary_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    positive_total = weights[labels == 1].sum()
    negative_total = weights[labels == 0].sum()
    if positive_total <= 0.0 or negative_total <= 0.0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    labels, scores, weights = labels[order], scores[order], weights[order]
    concordant = 0.0
    negative_below = 0.0
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[stop] == scores[start]:
            stop += 1
        tie_positive = weights[start:stop][labels[start:stop] == 1].sum()
        tie_negative = weights[start:stop][labels[start:stop] == 0].sum()
        concordant += tie_positive * (negative_below + 0.5 * tie_negative)
        negative_below += tie_negative
        start = stop
    return float(concordant / (positive_total * negative_total))


def paired_bootstrap_auc(
    labels: np.ndarray,
    results: dict[str, dict[str, object]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    names = ("dnn", "gnn", "transformer")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    values = {name: np.empty(replicates, dtype=float) for name in names}
    for replica in range(replicates):
        sample = np.concatenate(
            (
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            )
        )
        sampled_labels = labels[sample]
        for name in names:
            values[name][replica] = binary_auc(
                sampled_labels, np.asarray(results[name]["scores"])[sample]
            )

    output: dict[str, object] = {
        "replicates": replicates,
        "seed": seed,
        "auc": {},
        "paired_differences": {},
    }
    for name in names:
        low, high = np.quantile(values[name], [0.025, 0.975])
        output["auc"][name] = {
            "bootstrap_mean": float(np.mean(values[name])),
            "bootstrap_std": float(np.std(values[name], ddof=1)),
            "ci95": [float(low), float(high)],
        }
    for left, right in (("gnn", "dnn"), ("gnn", "transformer"), ("transformer", "dnn")):
        difference = values[left] - values[right]
        low, high = np.quantile(difference, [0.025, 0.975])
        output["paired_differences"][f"{left}_minus_{right}"] = {
            "mean": float(np.mean(difference)),
            "ci95": [float(low), float(high)],
            "fraction_positive": float(np.mean(difference > 0.0)),
        }
    return output


def train_candidate(
    *,
    name: str,
    factory: ModelFactory,
    seed_offset: int,
    events: GraphEvents,
    folds: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    output: Path,
    save_models: bool,
    device: torch.device,
    physics_weights: np.ndarray,
) -> dict[str, object]:
    scores = np.full(len(events), np.nan, dtype=np.float32)
    histories: dict[int, list[float]] = {}
    fold_summaries = []
    parameter_count: int | None = None
    labels = events.labels.astype(np.int8)

    for fold in range(folds):
        fold_seed = seed + seed_offset + fold
        seed_everything(fold_seed)
        train_indices = np.flatnonzero(events.fold != fold)
        test_indices = np.flatnonzero(events.fold == fold)
        train_data = weighted_tensors(events, train_indices, physics_weights)
        test_data = tensors(events, test_indices)
        loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(fold_seed),
        )
        model = factory().to(device)
        if parameter_count is None:
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1.0e-4
        )
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        losses = []
        for _epoch in range(epochs):
            model.train()
            total = 0.0
            seen = 0
            for batch in loader:
                batch = tuple(item.to(device) for item in batch)
                optimizer.zero_grad(set_to_none=True)
                logits = model(*batch[:-2])
                per_event_loss = criterion(logits, batch[-2])
                loss = (per_event_loss * batch[-1]).sum() / batch[-1].sum()
                loss.backward()
                optimizer.step()
                total += float((per_event_loss.detach() * batch[-1]).sum())
                seen += float(batch[-1].sum())
            losses.append(total / max(seen, 1))
        histories[fold] = losses
        scores[test_indices] = predict(model, test_data, batch_size, device)
        if save_models:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": name,
                    "fold": fold,
                    "seed": fold_seed,
                },
                output / f"{name}_fold{fold}.pt",
            )
        fold_summaries.append({
            "fold": fold,
            "train_events": int(len(train_indices)),
            "test_events": int(len(test_indices)),
            "test_signal": int(np.count_nonzero(labels[test_indices] == 1)),
            "test_background": int(np.count_nonzero(labels[test_indices] == 0)),
            "test_background_by_process": {
                PROCESS_NAMES.get(int(process_id), str(int(process_id))): int(np.count_nonzero(
                    (labels[test_indices] == 0)
                    & (events.process_id[test_indices] == process_id)
                ))
                for process_id in np.unique(events.process_id[test_indices][labels[test_indices] == 0])
            },
            "test_auc": binary_auc(labels[test_indices], scores[test_indices]),
            "test_auc_xsec_weighted": weighted_binary_auc(
                labels[test_indices],
                scores[test_indices],
                cross_section_training_weights(events, test_indices, physics_weights),
            ),
            "final_train_bce": losses[-1],
        })

    if not np.all(np.isfinite(scores)):
        raise RuntimeError(f"{name}: non-finite or missing out-of-fold scores")
    edges, bin_rows = score_bin_table(labels, scores)
    return {
        "scores": scores,
        "histories": histories,
        "folds": fold_summaries,
        "parameter_count": int(parameter_count or 0),
        "score_bins": {"edges": edges, "unweighted_poc_yields": bin_rows},
    }


def plot_comparison(
    output: Path,
    labels: np.ndarray,
    results: dict[str, dict[str, object]],
    analysis_weights: np.ndarray,
) -> None:
    candidates = ("dnn", "gnn", "transformer")
    bins = np.linspace(0.0, 1.0, 31)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), sharey=True)
    for axis, name in zip(axes, candidates):
        scores = np.asarray(results[name]["scores"])
        axis.hist(
            scores[labels == 0], bins=bins, density=True, histtype="step",
            linewidth=2, label="TT + Zto2Nu + WtoLNu",
            weights=analysis_weights[labels == 0],
        )
        axis.hist(
            scores[labels == 1], bins=bins, density=True, histtype="step",
            linewidth=2, label="Corridor signal",
            weights=analysis_weights[labels == 1],
        )
        axis.set(
            xlabel=f"{name.upper()} score",
            yscale="log",
            title=f"xsec-weighted OOF AUC = {weighted_binary_auc(labels, scores, analysis_weights):.3f}",
        )
    axes[0].set_ylabel("Normalized events")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "oof_score_distributions.png", dpi=160)
    plt.close(fig)

    styles = {
        "dnn": ("-", "DNN"),
        "gnn": ("-", "GNN"),
        "transformer": ("-", "Transformer"),
        "global_only": (":", "Global-only control"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    for name, (linestyle, label) in styles.items():
        scores = np.asarray(results[name]["scores"])
        fpr, tpr = roc_points(labels, scores)
        ax.plot(
            fpr, tpr, linewidth=2, linestyle=linestyle,
            label=f"{label}: {binary_auc(labels, scores):.3f}",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5")
    ax.set(
        xlabel="Background efficiency",
        ylabel="Corridor signal efficiency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "oof_roc_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name in candidates:
        histories = results[name]["histories"]
        stacked = np.asarray([histories[fold] for fold in sorted(histories)])
        epoch = np.arange(1, stacked.shape[1] + 1)
        ax.plot(epoch, stacked.mean(axis=0), linewidth=2, label=name.upper())
        ax.fill_between(epoch, stacked.min(axis=0), stacked.max(axis=0), alpha=0.12)
    ax.set(xlabel="Epoch", ylabel="Training BCE (fold mean and range)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_loss_comparison.png", dpi=160)
    plt.close(fig)

    aucs = [binary_auc(labels, np.asarray(results[name]["scores"])) for name in candidates]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    bars = ax.bar([name.upper() for name in candidates], aucs)
    ax.set(ylabel="Out-of-fold ROC AUC", ylim=(max(0.45, min(aucs) - 0.08), min(1.0, max(aucs) + 0.03)))
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in aucs], padding=3)
    fig.tight_layout()
    fig.savefig(output / "oof_auc_comparison.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare DNN, GNN, and Transformer OOF corridor classifiers.")
    parser.add_argument("--signal", nargs="+", required=True, type=Path)
    parser.add_argument("--background", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--background-label", default="TT + Zto2Nu + WtoLNu")
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument(
        "--selection-branch",
        choices=("feature_lowdm_preselection", "feature_lowdm_SR"),
        default="feature_lowdm_preselection",
    )
    parser.add_argument("--mstop", type=int, default=1050)
    parser.add_argument("--mlsp", type=int, default=900)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-jets", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    seed_everything(opts.seed)
    if opts.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(opts.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this process")
    start = time.time()

    events = load_graph_events(
        opts.signal,
        opts.background,
        target_mstop=opts.mstop,
        target_mlsp=opts.mlsp,
        max_jets=opts.max_jets,
        folds=opts.folds,
        selection_branch=opts.selection_branch,
    )
    normalization = json.loads(opts.normalization.read_text())
    if normalization.get("status") != "complete":
        raise RuntimeError(f"normalization is not complete: {opts.normalization}")
    physics_weights = normalized_physics_weights(events, normalization)
    analysis_weights = cross_section_training_weights(
        events, np.arange(len(events)), physics_weights
    )
    factories: dict[str, tuple[ModelFactory, int, bool]] = {
        "dnn": (
            lambda: FlattenDNNClassifier(max_jets=opts.max_jets, hidden=opts.hidden),
            0,
            True,
        ),
        "gnn": (lambda: JetGraphClassifier(hidden=opts.hidden), 1000, True),
        "transformer": (
            lambda: JetTransformerClassifier(hidden=opts.hidden, heads=4),
            2000,
            True,
        ),
        "global_only": (lambda: GlobalOnlyClassifier(hidden=opts.hidden), 3000, False),
    }
    results = {
        name: train_candidate(
            name=name,
            factory=factory,
            seed_offset=seed_offset,
            events=events,
            folds=opts.folds,
            epochs=opts.epochs,
            batch_size=opts.batch_size,
            learning_rate=opts.learning_rate,
            seed=opts.seed,
            output=opts.output,
            save_models=save_models,
            device=device,
            physics_weights=physics_weights,
        )
        for name, (factory, seed_offset, save_models) in factories.items()
    }
    labels = events.labels.astype(np.int8)
    plot_comparison(opts.output, labels, results, analysis_weights)

    with uproot.recreate(opts.output / "oof_scores.root") as root_file:
        root_file["Events"] = {
            "physical_dataset_id": events.physical_dataset_id,
            "run": events.run,
            "luminosityBlock": events.luminosity_block,
            "event": events.event,
            "is_signal": labels.astype(np.int32),
            "fold": events.fold.astype(np.int32),
            "dnn_score": np.asarray(results["dnn"]["scores"]),
            "gnn_score": np.asarray(results["gnn"]["scores"]),
            "transformer_score": np.asarray(results["transformer"]["scores"]),
            "global_only_score": np.asarray(results["global_only"]["scores"]),
            "mStop": events.mstop,
            "mLSP": events.mlsp,
            "gen_weight": events.gen_weight,
            "sampling_weight": events.sampling_weight,
            "signed_normalized_weight": physics_weights,
            "loss_weight": analysis_weights,
        }

    global_auc = binary_auc(labels, np.asarray(results["global_only"]["scores"]))
    metrics = {}
    for name in ("dnn", "gnn", "transformer", "global_only"):
        scores = np.asarray(results[name]["scores"])
        fold_aucs = [float(item["test_auc"]) for item in results[name]["folds"]]
        metrics[name] = {
            "oof_auc": binary_auc(labels, scores),
            "oof_auc_xsec_weighted": weighted_binary_auc(labels, scores, analysis_weights),
            "fold_auc_mean": float(np.mean(fold_aucs)),
            "fold_auc_std": float(np.std(fold_aucs)),
            "parameter_count": int(results[name]["parameter_count"]),
            "auc_gain_over_global": binary_auc(labels, scores) - global_auc,
            "score_min": float(np.min(scores)),
            "score_median": float(np.median(scores)),
            "score_max": float(np.max(scores)),
            "oof_auc_by_background_process": {
                PROCESS_NAMES.get(int(process_id), str(int(process_id))): binary_auc(
                    labels[(labels == 1) | ((labels == 0) & (events.process_id == process_id))],
                    scores[(labels == 1) | ((labels == 0) & (events.process_id == process_id))],
                )
                for process_id in np.unique(events.process_id[labels == 0])
            },
        }
    candidate_names = ("dnn", "gnn", "transformer")
    best = max(candidate_names, key=lambda name: metrics[name]["oof_auc_xsec_weighted"])
    bootstrap = paired_bootstrap_auc(
        labels,
        results,
        replicates=opts.bootstrap_replicates,
        seed=opts.seed + 9000,
    )
    summary = {
        "schema_version": "lowdm_corridor_model_comparison_v2",
        "status": "complete",
        "physics_scope": {
            "domain": f"{opts.selection_branch} && !feature_SR",
            "highdm_exclusivity": "conservative feature_SR veto",
            "signal_mass": {"mStop": opts.mstop, "mLSP": opts.mlsp, "deltaM": opts.mstop - opts.mlsp},
            "background": opts.background_label,
            "classifier_axis_candidates": ["dnn_score", "gnn_score", "transformer_score"],
        },
        "training": {
            "folds": opts.folds,
            "epochs": opts.epochs,
            "batch_size": opts.batch_size,
            "max_jets": opts.max_jets,
            "hidden": opts.hidden,
            "learning_rate": opts.learning_rate,
            "seed": opts.seed,
            "loss_weighting": "abs(genWeight*xsec*lumi/selected-source-sumw*sampling-correction); signal/background rescaled to 50/50 while retaining the xsec-weighted TT/Zto2Nu/WtoLNu mixture",
            "negative_weight_policy": "absolute normalized MC weights in BCE; signed normalized weights retained for yield evaluation",
            "data_usage": "no collision data used for training",
            "device": str(device),
        },
        "models": {
            "dnn": "pT-ordered padded jet features flattened with global/mass features",
            "gnn": "two dense edge-message layers with permutation-invariant pooling",
            "transformer": "two self-attention layers over jet tokens with a global/mass token",
            "global_only": "control MLP using MET, HT, Njet, Nb, and mass-hypothesis features",
        },
        "provenance": {
            "signal_files": [{"path": str(path), "sha256": sha256(path)} for path in opts.signal],
            "background_files": [{"path": str(path), "sha256": sha256(path)} for path in opts.background],
            "normalization": {"path": str(opts.normalization), "sha256": sha256(opts.normalization)},
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "uproot_version": uproot.__version__,
        },
        "events": {
            "total": len(events),
            "signal": int(np.count_nonzero(labels == 1)),
            "background": int(np.count_nonzero(labels == 0)),
            "background_by_process": {
                PROCESS_NAMES.get(int(process_id), str(int(process_id))): int(np.count_nonzero(
                    (labels == 0) & (events.process_id == process_id)
                ))
                for process_id in np.unique(events.process_id[labels == 0])
            },
            "signed_normalized_background_yield_by_process": {
                PROCESS_NAMES.get(int(process_id), str(int(process_id))): float(physics_weights[
                    (labels == 0) & (events.process_id == process_id)
                ].sum())
                for process_id in np.unique(events.process_id[labels == 0])
            },
            "fold_counts": np.bincount(events.fold, minlength=opts.folds).astype(int).tolist(),
        },
        "metrics": metrics,
        "best_small_sample_auc": {
            "model": best,
            "auc_xsec_weighted": metrics[best]["oof_auc_xsec_weighted"],
            "auc_unweighted": metrics[best]["oof_auc"],
        },
        "paired_bootstrap": bootstrap,
        "folds": {name: results[name]["folds"] for name in results},
        "score_bins": {name: results[name]["score_bins"] for name in candidate_names},
        "runtime_seconds": None,
        "artifacts": {
            "scores": "oof_scores.root",
            "score_plot": "oof_score_distributions.png",
            "roc_plot": "oof_roc_comparison.png",
            "auc_plot": "oof_auc_comparison.png",
            "loss_plot": "training_loss_comparison.png",
            "models": {
                name: [f"{name}_fold{fold}.pt" for fold in range(opts.folds)]
                for name in candidate_names
            },
        },
        "limitations": [
            "This is a process-balanced nominal-MC classifier comparison, not an expected-limit result.",
            "The supervised background set contains the three leading low-dM processes TT, Zto2Nu, and WtoLNu; smaller backgrounds are reserved for external evaluation.",
            "Only nominal kinematics are scored; shape-systematic score migrations are not yet evaluated.",
            "The feature_SR veto is more conservative than vetoing only the adopted High-dM search-bin population.",
            "Model selection requires full weighted yields, control regions, nuisance variations, and expected limits.",
        ],
    }
    if device.type == "mps":
        torch.mps.synchronize()
    summary["runtime_seconds"] = time.time() - start
    (opts.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "complete",
        "events": summary["events"],
        "auc": {name: metrics[name]["oof_auc"] for name in metrics},
        "best": summary["best_small_sample_auc"],
        "output": str(opts.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
