#!/usr/bin/env python3
"""Test-only GradientSHAP attribution for the 40 diagonal-v3 global inputs.

The jet graph is held fixed for every explained event.  Only the global input
vector is interpolated between a test-background reference event and the event
being explained.  This makes the result a conditional attribution of the two
global paths (global context and the raw-global classifier bypass), rather than
an attribution of the complete hybrid GNN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from ..data import (
    DIAGONAL_V3_GLOBAL_FEATURE_NAMES,
    GraphEvents,
    _read_one,
    concatenate,
    event_hash,
    split_buckets_2_1_7,
)
from ..model import PhysicsInformedJetGraphClassifier


TOPOLOGY_NAMES = {0: "Background", 1: "T2tt", 2: "T2bW", 3: "T2tb"}
GROUPS = (
    ("Event kinematics", range(0, 6), "#0072B2"),
    ("b system", range(6, 16), "#D55E00"),
    ("Resolved system", range(16, 26), "#009E73"),
    ("Event shape", range(26, 29), "#CC79A7"),
    ("ISR and recoil", range(29, 40), "#E69F00"),
)
FEATURE_LABELS = (
    r"$p_{T}^{miss}$",
    r"$H_T$",
    r"$N_{jet}$",
    r"$N_b$",
    r"$p_T^{miss}/\sqrt{H_T}$",
    r"$\min\Delta\phi(j_{1\ldots4},p_T^{miss})$",
    r"$p_T(b_1)$",
    r"$p_T(b_2)$",
    r"has two medium-$b$ jets",
    r"$M_T(b_1,p_T^{miss})$",
    r"$\min M_T(b,p_T^{miss})$",
    r"$\max M_T(b,p_T^{miss})$",
    r"$m_{bb}$",
    r"$\Delta R_{bb}$",
    r"$\Delta\phi_{bb}$",
    r"$M_{CT}(bb)$",
    r"has $m_{jj}/m_{jjj}$ candidate",
    r"$m_{jj}$",
    r"$m_{jjj}$",
    r"$W$-mass residual",
    r"top-mass residual",
    r"resolved $\chi^2$",
    r"$p_T(W_{resolved})$",
    r"$p_T(t_{resolved})$",
    r"has $M_{T2}^{bb}$",
    r"$M_{T2}^{bb}$",
    r"transverse sphericity",
    r"centrality",
    r"$p_T^{miss}/(p_T^{miss}+H_T)$",
    r"$N_{ISR}$",
    r"has ISR",
    r"$p_T(ISR)$",
    r"$\eta(ISR)$",
    r"$\Delta\phi(ISR,p_T^{miss})$",
    r"ISR-subjet max $b$ score",
    r"$p_T^{miss}/p_T(ISR)$",
    r"scalar recoil balance",
    r"vector recoil balance",
    r"$p_{T,\parallel}^{miss}/p_T(ISR)$",
    r"$H_T/p_T(ISR)$",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def cache_paths(cache: Path, kind: str) -> list[Path]:
    output = []
    for path in sorted(cache.glob(f"{kind}_cache_*.root")):
        sidecar_path = path.with_suffix(".json")
        if not sidecar_path.is_file():
            raise RuntimeError(f"missing cache sidecar: {sidecar_path}")
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar.get("status") != "complete":
            raise RuntimeError(f"incomplete cache shard: {path}")
        if int(sidecar.get("events_selected", -1)) > 0:
            output.append(path)
    return output


def hash_values(events: GraphEvents) -> np.ndarray:
    return event_hash(
        events.physical_dataset_id,
        events.run,
        events.luminosity_block,
        events.event,
    )


def load_test_sample(
    cache: Path,
    background_events: int,
    signal_events_per_topology: int,
    reference_events: int,
) -> tuple[GraphEvents, GraphEvents, dict[str, object]]:
    """Stream cache shards and retain deterministic lowest-hash test events."""
    requested = {
        0: background_events + reference_events,
        1: signal_events_per_topology,
        2: signal_events_per_topology,
        3: signal_events_per_topology,
    }
    candidates: dict[int, list[GraphEvents]] = {key: [] for key in requested}
    paths = [*cache_paths(cache, "signal"), *cache_paths(cache, "mc")]
    if not paths:
        raise RuntimeError("no complete cache shards found")

    for number, path in enumerate(paths, start=1):
        part = _read_one(
            path,
            target_mstop=None,
            target_mlsp=None,
            max_jets=10,
            folds=5,
            require_highdm_exclusive=False,
            selection_branch="feature_lowdm_diagonal_v3_SR",
            include_mass_features=False,
            top_targeted_features=False,
            engineered_features_v2=False,
            engineered_features_expanded=False,
            engineered_features_diagonal_v3=True,
            signal_topology_ids=(1, 2, 3),
            signal_delta_m_min=150,
            signal_delta_m_max=300,
            signal_mstop_min=600,
            signal_mstop_max=1800,
            allow_empty=True,
        )
        if part is None:
            continue
        split = split_buckets_2_1_7(
            part.physical_dataset_id,
            part.run,
            part.luminosity_block,
            part.event,
        )
        part = part.take(np.flatnonzero(split == 2))
        if not len(part):
            continue
        strata = np.where(
            part.labels.astype(bool), part.signal_topology_id, 0
        ).astype(np.int8)
        hashes = hash_values(part)
        for stratum, keep_count in requested.items():
            indices = np.flatnonzero(strata == stratum)
            if not len(indices):
                continue
            order = np.argsort(hashes[indices], kind="stable")[:keep_count]
            candidates[stratum].append(part.take(indices[order]))
        if number % 10 == 0 or number == len(paths):
            print(f"sample scan: {number}/{len(paths)} shards", flush=True)

    selected: dict[int, GraphEvents] = {}
    counts: dict[str, int] = {}
    for stratum, keep_count in requested.items():
        if not candidates[stratum]:
            raise RuntimeError(f"no test events for stratum {stratum}")
        merged = concatenate(candidates[stratum])
        order = np.argsort(hash_values(merged), kind="stable")[:keep_count]
        selected[stratum] = merged.take(order)
        counts[TOPOLOGY_NAMES[stratum]] = int(len(order))
        if len(order) < keep_count:
            raise RuntimeError(
                f"requested {keep_count} {TOPOLOGY_NAMES[stratum]} events, "
                f"found {len(order)}"
            )

    background = selected[0]
    reference = background.take(np.arange(reference_events, dtype=np.int64))
    explained_background = background.take(
        np.arange(reference_events, reference_events + background_events, dtype=np.int64)
    )
    explained = concatenate(
        [explained_background, selected[1], selected[2], selected[3]]
    )
    provenance = {
        "cache_shards_scanned": len(paths),
        "candidate_counts_before_reference_split": counts,
        "reference_background_events": int(len(reference)),
        "explained_counts": {
            "Background": int(len(explained_background)),
            "T2tt": int(len(selected[1])),
            "T2bW": int(len(selected[2])),
            "T2tb": int(len(selected[3])),
        },
        "partition": "test (70%) only",
        "sampling": "deterministic lowest event-hash within each class",
    }
    return explained, reference, provenance


def build_model(checkpoint: dict[str, object]) -> PhysicsInformedJetGraphClassifier:
    config = checkpoint["config"]
    model = PhysicsInformedJetGraphClassifier(
        node_features=6,
        global_features=len(DIAGONAL_V3_GLOBAL_FEATURE_NAMES),
        hidden=int(config["hidden"]),
        message_layers=int(config["message_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def conditional_gradient_shap(
    model: PhysicsInformedJetGraphClassifier,
    events: GraphEvents,
    references: GraphEvents,
    *,
    draws: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Expected-gradients approximation with the per-event jet graph fixed."""
    rng = np.random.default_rng(seed)
    all_attributions = []
    all_completeness_residuals = []
    all_logit_differences = []
    reference_globals = np.asarray(references.global_features, dtype=np.float32)

    for start in range(0, len(events), batch_size):
        stop = min(start + batch_size, len(events))
        nodes = torch.from_numpy(events.node_features[start:stop])
        mask = torch.from_numpy(events.node_mask[start:stop])
        eta = torch.from_numpy(events.node_eta[start:stop])
        phi = torch.from_numpy(events.node_phi[start:stop])
        target = torch.from_numpy(events.global_features[start:stop])
        batch = stop - start
        attribution_sum = torch.zeros_like(target)
        baseline_logit_sum = torch.zeros(batch, dtype=torch.float32)

        with torch.no_grad():
            target_logits = model(nodes, mask, eta, phi, target)

        for _ in range(draws):
            reference_index = rng.integers(0, len(reference_globals), size=batch)
            baseline = torch.from_numpy(reference_globals[reference_index])
            alpha = torch.from_numpy(
                rng.random((batch, 1), dtype=np.float32)
            )
            interpolated = (
                baseline + alpha * (target - baseline)
            ).detach().requires_grad_(True)
            logits = model(nodes, mask, eta, phi, interpolated)
            gradient = torch.autograd.grad(logits.sum(), interpolated)[0]
            attribution_sum += gradient.detach() * (target - baseline)
            with torch.no_grad():
                baseline_logit_sum += model(nodes, mask, eta, phi, baseline)

        attribution = attribution_sum / float(draws)
        mean_baseline_logit = baseline_logit_sum / float(draws)
        logit_difference = target_logits - mean_baseline_logit
        residual = attribution.sum(dim=1) - logit_difference
        all_attributions.append(attribution.numpy())
        all_completeness_residuals.append(residual.numpy())
        all_logit_differences.append(logit_difference.numpy())
        print(f"GradientSHAP: {stop}/{len(events)} events", flush=True)

    attributions = np.concatenate(all_attributions).astype(np.float32)
    residuals = np.concatenate(all_completeness_residuals)
    differences = np.concatenate(all_logit_differences)
    nonzero = np.abs(differences) > 1.0e-6
    diagnostics = {
        "mean_absolute_completeness_residual_logit": float(
            np.mean(np.abs(residuals))
        ),
        "median_absolute_completeness_residual_logit": float(
            np.median(np.abs(residuals))
        ),
        "mean_absolute_logit_difference": float(np.mean(np.abs(differences))),
        "relative_mean_absolute_residual": float(
            np.mean(np.abs(residuals[nonzero]))
            / max(np.mean(np.abs(differences[nonzero])), 1.0e-12)
        ),
        "correlation_sum_shap_vs_logit_difference": float(
            np.corrcoef(attributions.sum(axis=1), differences)[0, 1]
        ),
    }
    return attributions, diagnostics


def group_color(feature_index: int) -> str:
    for _, indices, color in GROUPS:
        if feature_index in indices:
            return color
    raise KeyError(feature_index)


def display_feature_view(
    attributions: np.ndarray, feature_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Combine the two exactly duplicated two-b/MT2-availability bits.

    Their event-level SHAP values must be summed to preserve local additivity;
    ranking the two identical inputs separately would overstate their physical
    importance and make their arbitrary credit split look meaningful.
    """
    display_attributions = []
    display_values = []
    display_names = []
    display_labels = []
    display_colors = []
    for index, (name, label) in enumerate(
        zip(DIAGONAL_V3_GLOBAL_FEATURE_NAMES, FEATURE_LABELS, strict=True)
    ):
        if index == 24:
            continue
        if index == 8:
            if not np.array_equal(feature_values[:, 8], feature_values[:, 24]):
                raise RuntimeError(
                    "has_two_medium_b and has_mt2_bb are no longer identical; "
                    "remove their presentation grouping"
                )
            display_attributions.append(attributions[:, 8] + attributions[:, 24])
            display_values.append(feature_values[:, 8])
            display_names.append("two_b_and_mt2_availability")
            display_labels.append(r"has two medium-$b$ jets")
            display_colors.append(group_color(8))
        else:
            display_attributions.append(attributions[:, index])
            display_values.append(feature_values[:, index])
            display_names.append(name)
            display_labels.append(label)
            display_colors.append(group_color(index))
    return (
        np.stack(display_attributions, axis=1),
        np.stack(display_values, axis=1),
        display_names,
        display_labels,
        display_colors,
    )


def save_figure(figure: plt.Figure, output: Path) -> dict[str, object]:
    artifacts = {}
    for suffix in ("png", "pdf"):
        path = output.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=220, bbox_inches="tight")
        artifacts[suffix] = {"path": str(path), "sha256": sha256(path)}
    plt.close(figure)
    return artifacts


def plot_importance(
    output: Path,
    attributions: np.ndarray,
    labels: list[str],
    colors: list[str],
) -> tuple[dict[str, object], np.ndarray]:
    hep.style.use("CMS")
    importance = np.mean(np.abs(attributions), axis=0)
    order = np.argsort(importance)
    figure, axis = plt.subplots(figsize=(9.2, 12.5))
    axis.barh(
        np.arange(len(order)),
        importance[order],
        color=[colors[index] for index in order],
        edgecolor="black",
        linewidth=0.45,
    )
    axis.set_yticks(np.arange(len(order)), [labels[index] for index in order])
    axis.yaxis.set_minor_locator(NullLocator())
    axis.set_xlabel(r"Mean $|\mathrm{GradientSHAP}|$ on GNN logit")
    axis.tick_params(axis="y", labelsize=11)
    axis.grid(axis="x", alpha=0.20)
    handles = [
        Line2D([0], [0], color=color, linewidth=7, label=name)
        for name, _, color in GROUPS
    ]
    axis.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)
    hep.cms.label(llabel="Simulation", rlabel="(13.6 TeV)", ax=axis)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return save_figure(figure, output), order


def plot_beeswarm(
    output: Path,
    attributions: np.ndarray,
    feature_values: np.ndarray,
    order: np.ndarray,
    labels: list[str],
    *,
    top: int,
    seed: int,
) -> dict[str, object]:
    hep.style.use("CMS")
    selected = order[-top:]
    figure, axis = plt.subplots(figsize=(12.0, 10.0))
    rng = np.random.default_rng(seed)
    color_values = []
    scatter = None
    for row, feature in enumerate(selected):
        values = feature_values[:, feature]
        low, high = np.quantile(values, (0.05, 0.95))
        if high > low:
            colors = np.clip((values - low) / (high - low), 0.0, 1.0)
        else:
            colors = np.full(len(values), 0.5)
        jitter = rng.normal(0.0, 0.105, size=len(values))
        jitter = np.clip(jitter, -0.31, 0.31)
        scatter = axis.scatter(
            attributions[:, feature],
            row + jitter,
            c=colors,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=8.0,
            alpha=0.52,
            linewidths=0.0,
            rasterized=True,
        )
        color_values.append(colors)
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.set_xlim(-7.5, 7.5)
    axis.set_yticks(np.arange(len(selected)), [labels[index] for index in selected])
    axis.yaxis.set_minor_locator(NullLocator())
    axis.set_xlabel("GradientSHAP value on GNN logit")
    axis.tick_params(axis="y", labelsize=12)
    axis.grid(axis="x", alpha=0.18)
    assert scatter is not None
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.025, aspect=35)
    colorbar.set_label("Feature value", fontsize=12)
    colorbar.set_ticks((0.0, 1.0), labels=("Low", "High"))
    hep.cms.label(llabel="Simulation", rlabel="(13.6 TeV)", ax=axis)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return save_figure(figure, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--background-events", type=int, default=512)
    parser.add_argument("--signal-events-per-topology", type=int, default=256)
    parser.add_argument("--reference-events", type=int, default=256)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9022026)
    opts = parser.parse_args()
    if min(
        opts.background_events,
        opts.signal_events_per_topology,
        opts.reference_events,
        opts.draws,
        opts.batch_size,
        opts.top,
    ) < 1:
        raise ValueError("all count arguments must be positive")
    if opts.top > len(DIAGONAL_V3_GLOBAL_FEATURE_NAMES):
        raise ValueError("--top exceeds the number of global features")

    output = opts.output_dir or (opts.result_dir / "global_shap")
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((opts.result_dir / "selection.json").read_text())
    checkpoint_path = (
        opts.result_dir
        / "trials"
        / str(selection["best_trial"])
        / "best_model.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["global_feature_names"]) != tuple(
        DIAGONAL_V3_GLOBAL_FEATURE_NAMES
    ):
        raise RuntimeError("checkpoint global-feature schema mismatch")

    started = time.time()
    events, references, sampling = load_test_sample(
        opts.cache,
        opts.background_events,
        opts.signal_events_per_topology,
        opts.reference_events,
    )
    model = build_model(checkpoint)
    attributions, diagnostics = conditional_gradient_shap(
        model,
        events,
        references,
        draws=opts.draws,
        batch_size=opts.batch_size,
        seed=opts.seed,
    )
    (
        display_attributions,
        display_values,
        display_names,
        display_labels,
        display_colors,
    ) = display_feature_view(attributions, events.global_features)
    importance_artifacts, order = plot_importance(
        output / "diagonal_v3_global_gradientshap_importance",
        display_attributions,
        display_labels,
        display_colors,
    )
    beeswarm_artifacts = plot_beeswarm(
        output / "diagonal_v3_global_gradientshap_beeswarm",
        display_attributions,
        display_values,
        order,
        display_labels,
        top=opts.top,
        seed=opts.seed + 1,
    )

    npz_path = output / "diagonal_v3_global_gradientshap_values.npz"
    np.savez_compressed(
        npz_path,
        shap_values=attributions,
        feature_values=events.global_features,
        labels=events.labels,
        signal_topology_id=events.signal_topology_id,
        feature_names=np.asarray(DIAGONAL_V3_GLOBAL_FEATURE_NAMES),
        event_hash=hash_values(events),
    )
    raw_importance = np.mean(np.abs(attributions), axis=0)
    raw_order = np.argsort(raw_importance)
    display_importance = np.mean(np.abs(display_attributions), axis=0)
    audit = {
        "status": "complete",
        "schema_version": "lowdm_diagonal_v3_global_gradientshap_v1",
        "method": (
            "Expected gradients / GradientSHAP on model logit; global inputs "
            "vary over test-background references while the event jet graph is fixed"
        ),
        "scope": "40 global features only; no jet-node or edge attribution",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "trial": str(selection["best_trial"]),
            "epoch": int(checkpoint["epoch"]),
        },
        "sampling": sampling,
        "draws_per_event": int(opts.draws),
        "seed": int(opts.seed),
        "diagnostics": diagnostics,
        "display_grouping": {
            "raw_input_count": len(DIAGONAL_V3_GLOBAL_FEATURE_NAMES),
            "displayed_logical_feature_count": len(display_names),
            "combined_raw_features": ["has_two_medium_b", "has_mt2_bb"],
            "reason": (
                "The two inputs are exactly identical in the evaluated tensor. "
                "Their event-level SHAP values are summed to preserve additivity "
                "and avoid ranking an arbitrary credit split."
            ),
        },
        "feature_ranking": [
            {
                "rank": rank,
                "name": display_names[index],
                "label": display_labels[index],
                "mean_abs_shap_logit": float(display_importance[index]),
                "mean_shap_logit": float(
                    np.mean(display_attributions[:, index])
                ),
            }
            for rank, index in enumerate(order[::-1], start=1)
        ],
        "raw_40_feature_ranking": [
            {
                "rank": rank,
                "name": DIAGONAL_V3_GLOBAL_FEATURE_NAMES[index],
                "label": FEATURE_LABELS[index],
                "mean_abs_shap_logit": float(raw_importance[index]),
                "mean_shap_logit": float(np.mean(attributions[:, index])),
            }
            for rank, index in enumerate(raw_order[::-1], start=1)
        ],
        "artifacts": {
            "importance": importance_artifacts,
            "beeswarm": beeswarm_artifacts,
            "values_npz": {"path": str(npz_path), "sha256": sha256(npz_path)},
        },
        "runtime_seconds": time.time() - started,
        "interpretation_warning": (
            "Attribution describes model response, not causal physics impact or "
            "expected-limit improvement. Correlated features share attribution."
        ),
    }
    audit_path = output / "diagonal_v3_global_gradientshap_audit.json"
    write_json(audit_path, audit)
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "top_features": audit["feature_ranking"][:10],
        "diagnostics": diagnostics,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
