#!/usr/bin/env python3
"""Open the frozen diagonal-v3 70% test split exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.lines import Line2D

from ..model import PhysicsInformedJetGraphClassifier
from .plot_full_gnn_results import (
    PROCESS_COLORS,
    PROCESS_LABELS,
    PROCESS_ORDER,
    grouped_process,
)
from .plot_lowdm_category_nn_out import signed_stack
from .significance import CATEGORY_NAMES, diagonal_v3_category_ids, evaluate_binning
from .train_oof import predict, tensors
from .tune_full_gnn import (
    background_process_names,
    load_inputs,
    physics_weights,
    validation_metrics,
)


CATEGORY_LABELS = (
    r"$N_b=1$, $N_{\mathrm{ISR}}=0$",
    r"$N_b=1$, $N_{\mathrm{ISR}}\geq1$",
    r"$N_b\geq2$, $N_{\mathrm{ISR}}=0$",
    r"$N_b\geq2$, $N_{\mathrm{ISR}}\geq1$",
)
TOPOLOGY_NAMES = {1: "T2tt", 2: "T2bW", 3: "T2tb"}
BENCHMARKS = ((1, 600, 400), (1, 900, 700))
BENCHMARK_COLORS = ("#E60000", "#0057FF")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def flat_bins(categories: np.ndarray, scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    score_bin = np.searchsorted(edges, scores, side="right") - 1
    score_bin = np.clip(score_bin, 0, len(edges) - 2)
    return categories.astype(np.int64) * (len(edges) - 1) + score_bin


def histogram(
    coordinate: np.ndarray, weights: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.bincount(coordinate, weights=weights, minlength=bins)[:bins]
    sumw2 = np.bincount(
        coordinate, weights=np.square(weights), minlength=bins
    )[:bins]
    return values.astype(float), sumw2.astype(float)


def weighted_roc(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    labels = np.asarray(labels, dtype=bool)
    weights = np.abs(np.asarray(weights, dtype=float))
    if not np.any(labels) or not np.any(~labels):
        raise RuntimeError("ROC requires signal and background")
    order = np.argsort(scores, kind="stable")[::-1]
    ordered_label = labels[order]
    ordered_weight = weights[order]
    tpr = np.cumsum(ordered_weight * ordered_label) / np.sum(weights[labels])
    fpr = np.cumsum(ordered_weight * ~ordered_label) / np.sum(weights[~labels])
    tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.r_[0.0, fpr, 1.0]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def plot_roc(
    output: Path,
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    topology: np.ndarray,
    stem: str = "diagonal_v3_test_roc",
) -> dict[str, Any]:
    hep.style.use("CMS")
    curves: dict[str, dict[str, float]] = {}
    figure, axis = plt.subplots(figsize=(8.3, 7.4))
    definitions = [("All signal", labels, "black", "-")]
    for identifier, color in zip((1, 2, 3), ("red", "blue", "green"), strict=True):
        use = ~labels | (labels & (topology == identifier))
        definitions.append(
            (TOPOLOGY_NAMES[identifier], labels[use], color, "--", use)
        )
    for definition in definitions:
        if len(definition) == 4:
            name, local_label, color, style = definition
            local_score, local_weight = scores, weights
        else:
            name, local_label, color, style, use = definition
            local_score, local_weight = scores[use], weights[use]
        fpr, tpr, auc = weighted_roc(local_label, local_score, local_weight)
        axis.plot(fpr, tpr, color=color, linestyle=style, linewidth=2.2, label=f"{name}: AUC = {auc:.4f}")
        curves[name] = {"auc": auc}
    axis.plot((0, 1), (0, 1), color="0.5", linestyle=":", linewidth=1.3)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Background efficiency", fontsize=15)
    axis.set_ylabel("Signal efficiency", fontsize=15)
    axis.tick_params(labelsize=13)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=12, loc="lower right")
    hep.cms.label(
        llabel="Simulation",
        rlabel="(13.6 TeV)",
        ax=axis,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    artifacts = {}
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        artifacts[suffix] = {"path": str(path), "sha256": sha256(path)}
    plt.close(figure)
    return {"curves": curves, "artifacts": artifacts}


def plot_templates(
    output: Path,
    process_histograms: dict[str, np.ndarray],
    process_sumw2: dict[str, np.ndarray],
    signal_histograms: dict[str, np.ndarray],
    score_edges: np.ndarray,
    signal_scale: float,
) -> dict[str, Any]:
    import mplhep as hep

    hep.style.use("CMS")
    nscore = len(score_edges) - 1
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 12.0), sharex=True)
    panel_edges = np.arange(nscore + 1, dtype=float)
    handles = [
        Line2D([0], [0], color=PROCESS_COLORS[name], linewidth=7, label=PROCESS_LABELS[name])
        for name in PROCESS_ORDER
    ]
    for (topology, mstop, mlsp), color in zip(BENCHMARKS, BENCHMARK_COLORS, strict=True):
        key = f"{TOPOLOGY_NAMES[topology]}_{mstop}_{mlsp}"
        handles.append(
            Line2D(
                [0], [0], color=color, linestyle="--", linewidth=2.2,
                label=rf"T2tt $({mstop},{mlsp})$ GeV $\times{signal_scale:g}$",
            )
        )
    for category, axis in enumerate(axes.flat):
        start, stop = category * nscore, (category + 1) * nscore
        values = [process_histograms[name][start:stop] for name in PROCESS_ORDER]
        signed_stack(
            axis,
            values,
            panel_edges,
            [PROCESS_LABELS[name] for name in PROCESS_ORDER],
            [PROCESS_COLORS[name] for name in PROCESS_ORDER],
        )
        total = sum(values, np.zeros(nscore))
        total_sumw2 = sum(
            (process_sumw2[name][start:stop] for name in PROCESS_ORDER),
            np.zeros(nscore),
        )
        uncertainty = np.sqrt(total_sumw2)
        axis.fill_between(
            panel_edges,
            np.r_[total - uncertainty, total[-1] - uncertainty[-1]],
            np.r_[total + uncertainty, total[-1] + uncertainty[-1]],
            step="post",
            facecolor="0.82",
            edgecolor="0.2",
            hatch="////",
            linewidth=0.0,
            alpha=0.65,
        )
        axis.stairs(total, panel_edges, color="black", linewidth=1.35)
        maximum = float(np.max(total + uncertainty)) if len(total) else 1.0
        for (topology, mstop, mlsp), color in zip(BENCHMARKS, BENCHMARK_COLORS, strict=True):
            key = f"{TOPOLOGY_NAMES[topology]}_{mstop}_{mlsp}"
            signal = signal_histograms.get(key, np.zeros(4 * nscore))[start:stop]
            axis.stairs(
                signal_scale * signal,
                panel_edges,
                color=color,
                linestyle="--",
                linewidth=2.2,
            )
            maximum = max(maximum, float(np.max(signal_scale * signal)))
        with np.errstate(divide="ignore", invalid="ignore"):
            neff = np.square(total) / total_sumw2
            relative = np.sqrt(total_sumw2) / total
        finite_neff = neff[np.isfinite(neff)]
        finite_relative = relative[np.isfinite(relative)]
        axis.text(
            0.98,
            0.95,
            (
                rf"min $N_{{eff}}={np.min(finite_neff):.1f}$; "
                rf"max MC stat.=${100*np.max(finite_relative):.0f}\%$"
                if len(finite_neff) and len(finite_relative)
                else "MC-stat unavailable"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=11,
        )
        axis.set_title(CATEGORY_LABELS[category], fontsize=18)
        axis.set_yscale("symlog", linthresh=0.3, linscale=0.8)
        axis.set_ylim(-0.5, max(1.0, 2.4 * maximum))
        axis.set_xlim(0.0, nscore)
        axis.grid(axis="y", alpha=0.16)
        axis.tick_params(labelsize=12)
    tick_labels = [
        f"{left:.3f}–{right:.3f}"
        for left, right in zip(score_edges[:-1], score_edges[1:], strict=True)
    ]
    for axis in axes[1]:
        axis.set_xticks(np.arange(nscore) + 0.5)
        axis.set_xticklabels(tick_labels, rotation=18, ha="right")
        axis.set_xlabel("GNN output (absolute)", fontsize=15)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Expected events (70% test $\times1/0.70$)", fontsize=15)
    hep.cms.label("Work in progress", data=False, lumi=109.82, com=13.6, ax=axes[0, 0])
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.965),
        ncol=5,
        fontsize=10.5,
        frameon=False,
    )
    figure.text(
        0.99,
        0.012,
        "Independent 70% test; model and absolute score edges frozen on validation",
        ha="right",
        fontsize=11,
    )
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.11, top=0.87, hspace=0.27, wspace=0.15)
    artifacts = {}
    for suffix in ("png", "pdf"):
        path = output / f"diagonal_v3_test_gnnout_categories.{suffix}"
        figure.savefig(path, dpi=200)
        artifacts[suffix] = {"path": str(path), "sha256": sha256(path)}
    plt.close(figure)
    return artifacts


def write_templates(
    output: Path,
    process_histograms: dict[str, np.ndarray],
    process_sumw2: dict[str, np.ndarray],
    signal_histograms: dict[str, np.ndarray],
) -> dict[str, str]:
    path = output / "diagonal_v3_test_templates.root"
    bins = len(next(iter(process_histograms.values())))
    edges = np.arange(bins + 1, dtype=float)
    with uproot.recreate(path) as root_file:
        total = sum(process_histograms.values(), np.zeros(bins))
        total_sumw2 = sum(process_sumw2.values(), np.zeros(bins))
        root_file["background_total"] = (total, edges)
        root_file["background_sumw2"] = (total_sumw2, edges)
        for name in PROCESS_ORDER:
            root_file[f"background/{name}"] = (process_histograms[name], edges)
            root_file[f"background_sumw2_by_process/{name}"] = (
                process_sumw2[name], edges
            )
        for name, values in signal_histograms.items():
            root_file[f"signal/{name}"] = (values, edges)
    return {"path": str(path), "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument("--xsec", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--signal-scale", type=float, default=10.0)
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    summary_path = opts.output / "diagonal_v3_test_summary.json"
    selection_path = opts.result / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("test_touched"):
        recorded = Path(str(selection.get("test_summary", "")))
        if recorded == summary_path and summary_path.is_file():
            print(summary_path.read_text())
            return 0
        raise RuntimeError("test partition was already opened by another evaluation")
    if selection.get("status") != "frozen_before_test":
        raise RuntimeError("model and binning are not frozen before test")
    selection_hash_before_test = sha256(selection_path)

    opts.max_jets = 10
    opts.signal_topologies = ("T2tt", "T2bW", "T2tb")
    opts.delta_m_min, opts.delta_m_max = 150, 300
    opts.mstop_min, opts.mstop_max = 600, 1800
    opts.top_targeted_features = False
    opts.engineered_features_v2 = False
    opts.engineered_features_expanded = False
    opts.engineered_features_diagonal_v3 = True
    opts.require_highdm_exclusive = False
    opts.require_adopted_lowdm_bins = False
    opts.selection_branch = "feature_lowdm_diagonal_v3_SR"
    events, campaign, split = load_inputs(opts)
    train_indices = np.flatnonzero(split == 0)
    validation_indices = np.flatnonzero(split == 1)
    test_indices = np.flatnonzero(split == 2)
    if not len(test_indices):
        raise RuntimeError("deterministic split has no test events")
    campaign = dict(campaign)
    campaign["manifest"] = str(opts.campaign_manifest)
    manifest = json.loads(opts.campaign_manifest.read_text())
    weights = physics_weights(events, campaign, json.loads(opts.xsec.read_text()))
    process_names = background_process_names(events, manifest)

    checkpoint_path = opts.result / "trials" / selection["best_trial"] / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    requested_device = opts.device
    if requested_device == "mps" and not torch.backends.mps.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    model = PhysicsInformedJetGraphClassifier(
        node_features=events.node_features.shape[2],
        global_features=events.global_features.shape[1],
        hidden=int(config["hidden"]),
        message_layers=int(config["message_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    test_scores = predict(model, tensors(events, test_indices), opts.batch_size, device)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    labels = events.labels[test_indices].astype(bool)
    local_weights = weights[test_indices] / 0.70
    categories = diagonal_v3_category_ids(events, test_indices)
    score_edges = np.asarray(selection["score_edges"], dtype=float)
    coordinate = flat_bins(categories, test_scores, score_edges)
    bins = len(CATEGORY_NAMES) * (len(score_edges) - 1)
    local_process = np.asarray(
        [grouped_process(str(name)) for name in process_names[test_indices]],
        dtype=object,
    )
    unknown = sorted(set(local_process[~labels]) - set(PROCESS_ORDER))
    if unknown:
        raise RuntimeError(f"unmapped background processes: {unknown}")
    process_histograms: dict[str, np.ndarray] = {}
    process_sumw2: dict[str, np.ndarray] = {}
    for name in PROCESS_ORDER:
        use = ~labels & (local_process == name)
        process_histograms[name], process_sumw2[name] = histogram(
            coordinate[use], local_weights[use], bins
        )
    signal_histograms: dict[str, np.ndarray] = {}
    signal_rows = np.unique(
        np.stack(
            (
                events.signal_topology_id[test_indices][labels],
                events.mstop[test_indices][labels],
                events.mlsp[test_indices][labels],
            ),
            axis=1,
        ),
        axis=0,
    )
    for topology, mstop, mlsp in signal_rows:
        use = (
            labels
            & (events.signal_topology_id[test_indices] == topology)
            & (events.mstop[test_indices] == mstop)
            & (events.mlsp[test_indices] == mlsp)
        )
        key = f"{TOPOLOGY_NAMES[int(topology)]}_{int(mstop)}_{int(mlsp)}"
        signal_histograms[key], _ = histogram(
            coordinate[use], local_weights[use], bins
        )

    frozen_binning_test = evaluate_binning(
        events,
        test_indices,
        test_scores,
        weights,
        score_edges,
        subset_scale=1.0 / 0.70,
        min_background_neff=0.0,
        max_relative_mc_stat=float("inf"),
    )
    if frozen_binning_test is None:
        raise RuntimeError("frozen binning has nonpositive test-background yield")
    metrics = validation_metrics(
        events, test_indices, test_scores, weights, process_names
    )
    score_path = opts.output / "diagonal_v3_test_scores.npz"
    np.savez_compressed(
        score_path,
        test_indices=test_indices,
        scores=test_scores.astype(np.float32),
        labels=events.labels[test_indices],
        category=categories,
        weights=local_weights.astype(np.float64),
        signal_topology_id=events.signal_topology_id[test_indices],
        process_id=events.process_id[test_indices],
    )
    roc = plot_roc(
        opts.output,
        labels,
        test_scores,
        local_weights,
        events.signal_topology_id[test_indices],
    )
    plot_artifacts = plot_templates(
        opts.output,
        process_histograms,
        process_sumw2,
        signal_histograms,
        score_edges,
        opts.signal_scale,
    )
    root_artifact = write_templates(
        opts.output, process_histograms, process_sumw2, signal_histograms
    )
    payload = {
        "schema_version": "gnn_lowdm_diagonal_v3_test_v1",
        "status": "complete",
        "test_opened_after_validation_freeze": True,
        "test_opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_reuse_policy": "one-shot final evaluation; never tune on test",
        "selection_sha256_before_test": selection_hash_before_test,
        "device": str(device),
        "split": {
            "policy": "deterministic event-key 2:1:7",
            "train_events": int(len(train_indices)),
            "validation_events": int(len(validation_indices)),
            "test_events": int(len(test_indices)),
            "test_signal_events": int(np.count_nonzero(labels)),
            "test_background_events": int(np.count_nonzero(~labels)),
        },
        "normalization": {
            "luminosity_fb": float(manifest["normalization"]["luminosity_pb"]) / 1000.0,
            "test_fraction": 0.70,
            "test_weight_scale": 1.0 / 0.70,
        },
        "frozen_definition": {
            "categories": list(CATEGORY_NAMES),
            "score_edges_absolute": score_edges.tolist(),
            "bins": bins,
            "test_side_optimization": False,
        },
        "background_processes": list(PROCESS_ORDER),
        "other_present": False,
        "test_metrics": metrics,
        "frozen_binning_test_metrics": frozen_binning_test,
        "roc": roc,
        "artifacts": {
            "category_plot": plot_artifacts,
            "templates_root": root_artifact,
            "scores_npz": {"path": str(score_path), "sha256": sha256(score_path)},
        },
    }
    write_json(summary_path, payload)
    selection.update(
        status="test_evaluated",
        test_touched=True,
        test_summary=str(summary_path),
        test_summary_sha256=sha256(summary_path),
    )
    write_json(selection_path, selection)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
