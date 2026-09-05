#!/usr/bin/env python3
"""Train the MET/sqrt(HT)-inclusive diagonal GNN with S/sqrt(B) guidance."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import uproot
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

from ..data import DIAGONAL_V3_GLOBAL_FEATURE_NAMES
from ..model import PhysicsInformedJetGraphClassifier
from .physics_informed_v2 import physics_informed_loss_weights
from .plot_hyperparameter_losses import save_plot as save_hyperparameter_plot
from .significance import (
    binning_selection_key,
    diagonal_v3_category_ids,
    evaluate_binning,
    optimize_validation_binning,
    soft_s_over_sqrt_b_loss,
    weighted_quantiles,
)
from .train_oof import seed_everything, sha256, tensors
from .tune_full_gnn import (
    background_process_names,
    load_inputs,
    physics_weights,
    validation_metrics,
    write_json,
)


DEFAULT_TRIALS = (
    {
        "name": "h48_l2_bce_control",
        "hidden": 48,
        "message_layers": 2,
        "dropout": 0.25,
        "learning_rate": 4.0e-4,
        "weight_decay": 1.0e-3,
        "batch_size": 2048,
        "significance_weight": 0.0,
        "significance_temperature": 0.35,
    },
    {
        "name": "h48_l2_sig005",
        "hidden": 48,
        "message_layers": 2,
        "dropout": 0.25,
        "learning_rate": 4.0e-4,
        "weight_decay": 1.0e-3,
        "batch_size": 2048,
        "significance_weight": 0.05,
        "significance_temperature": 0.35,
    },
    {
        "name": "h48_l2_sig010",
        "hidden": 48,
        "message_layers": 2,
        "dropout": 0.25,
        "learning_rate": 4.0e-4,
        "weight_decay": 1.0e-3,
        "batch_size": 2048,
        "significance_weight": 0.10,
        "significance_temperature": 0.35,
    },
    {
        "name": "h64_l2_sig010",
        "hidden": 64,
        "message_layers": 2,
        "dropout": 0.20,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-3,
        "batch_size": 1536,
        "significance_weight": 0.10,
        "significance_temperature": 0.35,
    },
)


def load_trial_configs(path: Path | None) -> list[dict[str, Any]]:
    rows = [dict(row) for row in DEFAULT_TRIALS]
    if path is not None:
        rows = [dict(row) for row in json.loads(path.read_text())["trials"]]
    required = {
        "name",
        "hidden",
        "message_layers",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "significance_weight",
        "significance_temperature",
    }
    names = []
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise RuntimeError(f"trial {row.get('name')} is missing {missing}")
        names.append(str(row["name"]))
    if len(names) != len(set(names)):
        raise RuntimeError("trial names are not unique")
    return rows


def make_dataset(events, indices: np.ndarray, weights: np.ndarray) -> TensorDataset:
    base = tensors(events, indices)
    categories = diagonal_v3_category_ids(events, indices).astype(np.int64)
    return TensorDataset(
        *base.tensors,
        torch.from_numpy(weights),
        torch.from_numpy(categories),
    )


def predict(
    model: nn.Module,
    dataset: TensorDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = tuple(item.to(device) for item in batch)
            output.append(torch.sigmoid(model(*batch[:5])).cpu().numpy())
    return np.concatenate(output)


def evaluate_bce(
    model: nn.Module,
    dataset: TensorDataset,
    batch_size: int,
    device: torch.device,
) -> float:
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    numerator = denominator = 0.0
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = tuple(item.to(device) for item in batch)
            values = criterion(model(*batch[:5]), batch[5])
            numerator += float(torch.sum(values * batch[6]).cpu())
            denominator += float(torch.sum(batch[6]).cpu())
    return numerator / max(denominator, 1.0)


def fast_validation_significance(
    events,
    indices: np.ndarray,
    scores: np.ndarray,
    analysis_weights: np.ndarray,
) -> dict[str, Any]:
    labels = events.labels[indices].astype(bool)
    thresholds = weighted_quantiles(
        scores[~labels],
        np.abs(analysis_weights[indices][~labels]),
        (0.70, 0.90, 0.98),
    )
    result = evaluate_binning(
        events,
        indices,
        scores,
        analysis_weights,
        (0.0, *thresholds, 1.0),
        subset_scale=10.0,
        min_background_neff=1.0,
        max_relative_mc_stat=1.0,
    )
    if result is None:
        return {
            "valid": False,
            "p10_gain": -1.0,
            "median_gain": -1.0,
            "p10_s_over_sqrt_b": -1.0,
            "median_s_over_sqrt_b": -1.0,
        }
    result["valid"] = True
    return result


def epoch_selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    significance = row["validation_significance"]
    return (
        float(significance["p10_gain"]),
        float(significance["median_gain"]),
        float(significance["p10_s_over_sqrt_b"]),
        float(row["minimum_mass_auc"]),
        float(row["macro_mass_auc"]),
        -float(row["validation_weighted_bce"]),
    )


def rng_payload(device: torch.device) -> dict[str, Any]:
    return {
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "mps_rng_state": (
            torch.mps.get_rng_state() if device.type == "mps" else None
        ),
    }


def restore_rng(payload: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    if device.type == "mps" and payload.get("mps_rng_state") is not None:
        torch.mps.set_rng_state(payload["mps_rng_state"])


def save_training_plot(path: Path, history: list[dict[str, Any]]) -> None:
    hep.style.use("CMS")
    epochs = np.asarray([row["epoch"] for row in history])
    panels = (
        (
            "a_bce",
            "Weighted BCE",
            None,
            (
                ([row["training_weighted_bce"] for row in history], "blue", "-", "o", "Train BCE"),
                ([row["validation_weighted_bce"] for row in history], "red", "--", "^", "Validation BCE"),
            ),
        ),
        (
            "b_accuracy",
            "Weighted accuracy at score = 0.5",
            (0.0, 1.02),
            (
                ([row["training_weighted_accuracy"] for row in history], "blue", "-", "o", "Train accuracy"),
                ([row["validation_weighted_accuracy"] for row in history], "red", "--", "^", "Validation accuracy"),
            ),
        ),
        (
            "c_significance",
            r"Gain over inclusive $S/\sqrt{B}$",
            None,
            (
                ([row["validation_significance"]["p10_gain"] for row in history], "blue", "-", "s", r"Validation p10 $S/\sqrt{B}$ gain"),
                ([row["validation_significance"]["median_gain"] for row in history], "red", "--", "D", r"Validation median $S/\sqrt{B}$ gain"),
            ),
        ),
    )
    for suffix, ylabel, ylim, series in panels:
        figure, axis = plt.subplots(figsize=(8.3, 7.2))
        for values, color, linestyle, marker, legend_label in series:
            axis.plot(
                epochs,
                values,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=7,
                linewidth=2.0,
                label=legend_label,
            )
        axis.set_xlabel("Epoch", fontsize=15)
        axis.set_ylabel(ylabel, fontsize=15)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.tick_params(labelsize=13)
        axis.grid(alpha=0.2)
        axis.xaxis.get_major_locator().set_params(integer=True)
        axis.legend(frameon=False, fontsize=13)
        hep.cms.label(
            llabel="Simulation",
            rlabel="(13.6 TeV)",
            ax=axis,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        output = path.with_name(f"{path.name}_{suffix}")
        figure.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)


def train_trial(
    opts: argparse.Namespace,
    config: dict[str, Any],
    events,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    analysis_weights: np.ndarray,
    process_names: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    trial_dir = opts.output / "trials" / str(config["name"])
    trial_dir.mkdir(parents=True, exist_ok=True)
    summary_path = trial_dir / "summary.json"
    if opts.resume and summary_path.is_file():
        previous_summary = json.loads(summary_path.read_text())
        if (
            previous_summary.get("status") == "complete"
            and previous_summary.get("config") == config
            and int(previous_summary.get("epochs_target", -1)) == opts.epochs
        ):
            return previous_summary
    training_weights = physics_informed_loss_weights(
        events, train_indices, analysis_weights
    )
    validation_weights = physics_informed_loss_weights(
        events, validation_indices, analysis_weights
    )
    train_data = make_dataset(events, train_indices, training_weights)
    validation_data = make_dataset(events, validation_indices, validation_weights)
    seed = opts.seed + sum(ord(character) for character in str(config["name"]))
    seed_everything(seed)
    model = PhysicsInformedJetGraphClassifier(
        node_features=events.node_features.shape[2],
        global_features=events.global_features.shape[1],
        hidden=int(config["hidden"]),
        message_layers=int(config["message_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=4.0e-5
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    epochs_without_improvement = 0
    start_epoch = 0
    latest_path = trial_dir / "latest.pt"
    if opts.resume and latest_path.is_file():
        latest = torch.load(latest_path, map_location=device, weights_only=False)
        if latest.get("config") != config:
            raise RuntimeError(f"resume config mismatch for {config['name']}")
        if tuple(latest.get("global_feature_names", ())) != tuple(
            DIAGONAL_V3_GLOBAL_FEATURE_NAMES
        ):
            raise RuntimeError(f"resume feature schema mismatch for {config['name']}")
        if latest.get("selection_branch") != opts.selection_branch:
            raise RuntimeError(f"resume selection mismatch for {config['name']}")
        model.load_state_dict(latest["state_dict"])
        optimizer.load_state_dict(latest["optimizer"])
        scheduler.load_state_dict(latest["scheduler"])
        history = list(latest["history"])
        best = latest["best"]
        start_epoch = int(latest["epoch"])
        epochs_without_improvement = int(
            latest.get("epochs_without_improvement", 0)
        )
        restore_rng(latest, device)
    started = time.time()
    for epoch in range(start_epoch, opts.epochs):
        loader = DataLoader(
            train_data,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train()
        bce_numerator = weight_denominator = correct_numerator = 0.0
        significance_sum = 0.0
        for batch in loader:
            batch = tuple(item.to(device) for item in batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch[:5])
            bce_values = criterion(logits, batch[5])
            bce = torch.sum(bce_values * batch[6]) / torch.sum(batch[6])
            significance = soft_s_over_sqrt_b_loss(
                logits,
                batch[5],
                batch[6],
                category_ids=batch[7],
                temperature=float(config["significance_temperature"]),
            )
            loss = bce + float(config["significance_weight"]) * significance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            bce_numerator += float(torch.sum(bce_values.detach() * batch[6]).cpu())
            weight_denominator += float(torch.sum(batch[6]).cpu())
            correct_numerator += float(
                torch.sum(
                    ((logits.detach() >= 0.0) == (batch[5] >= 0.5)) * batch[6]
                ).cpu()
            )
            significance_sum += float(significance.detach().cpu())

        validation_bce = evaluate_bce(
            model, validation_data, int(config["batch_size"]), device
        )
        validation_scores = predict(
            model, validation_data, int(config["batch_size"]), device
        )
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_weighted_bce": bce_numerator
            / max(weight_denominator, 1.0),
            "training_weighted_accuracy": correct_numerator
            / max(weight_denominator, 1.0),
            "training_soft_significance_loss": significance_sum
            / max(len(loader), 1),
            "validation_weighted_bce": validation_bce,
            "validation_weighted_accuracy": float(
                np.sum(
                    validation_weights
                    * (
                        (validation_scores >= 0.5)
                        == (events.labels[validation_indices] >= 0.5)
                    )
                )
                / max(float(np.sum(validation_weights)), 1.0)
            ),
            "validation_significance": fast_validation_significance(
                events,
                validation_indices,
                validation_scores,
                analysis_weights,
            ),
        }
        row.update(
            validation_metrics(
                events,
                validation_indices,
                validation_scores,
                analysis_weights,
                process_names,
            )
        )
        history.append(row)
        improved = best is None or epoch_selection_key(row) > epoch_selection_key(best)
        if improved:
            best = row
            epochs_without_improvement = 0
            torch.save(
                {
                    "schema_version": "lowdm_diagonal_v3_significance_checkpoint_v1",
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "config": config,
                    "global_feature_names": DIAGONAL_V3_GLOBAL_FEATURE_NAMES,
                    "selection_branch": opts.selection_branch,
                    "history": history,
                    "best": best,
                },
                trial_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
        scheduler.step(validation_bce)
        torch.save(
            {
                "schema_version": "lowdm_diagonal_v3_significance_checkpoint_v1",
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "config": config,
                "global_feature_names": DIAGONAL_V3_GLOBAL_FEATURE_NAMES,
                "selection_branch": opts.selection_branch,
                "history": history,
                "best": best,
                "epochs_without_improvement": epochs_without_improvement,
                **rng_payload(device),
            },
            latest_path,
        )
        write_json(
            trial_dir / "training_history.json",
            {"config": config, "history": history},
        )
        save_training_plot(trial_dir / "training", history)
        save_hyperparameter_plot(
            opts.output,
            opts.output / "all_candidates_train_validation_loss",
        )
        print(
            json.dumps(
                {
                    "trial": config["name"],
                    "epoch": epoch + 1,
                    "train_bce": row["training_weighted_bce"],
                    "validation_bce": validation_bce,
                    "validation_p10_gain": row["validation_significance"][
                        "p10_gain"
                    ],
                    "validation_median_gain": row["validation_significance"][
                        "median_gain"
                    ],
                    "improved": improved,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if device.type == "mps":
            torch.mps.empty_cache()
        if (
            epoch + 1 >= opts.minimum_epochs
            and epochs_without_improvement >= opts.patience
        ):
            break
    assert best is not None
    summary = {
        "status": "complete",
        "schema_version": "lowdm_diagonal_v3_significance_trial_v1",
        "config": config,
        "epochs_run": len(history),
        "epochs_target": opts.epochs,
        "best": best,
        "selection_key": epoch_selection_key(best),
        "runtime_seconds": time.time() - started,
        "test_touched": False,
    }
    write_json(trial_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument("--xsec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials-json", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--trial", action="append")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--minimum-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-train-events", type=int)
    parser.add_argument("--max-validation-events", type=int)
    parser.add_argument("--max-cache-shards-per-kind", type=int)
    parser.add_argument(
        "--selection-branch", default="feature_lowdm_diagonal_v3_SR"
    )
    parser.add_argument("--seed", type=int, default=831_2026)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    opts = parser.parse_args()
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
    opts.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "mps"
        if opts.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if opts.device == "auto" else opts.device
    )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    seed_everything(opts.seed)
    events, campaign, split = load_inputs(opts)
    if events.global_features.shape[1] != len(DIAGONAL_V3_GLOBAL_FEATURE_NAMES):
        raise RuntimeError("diagonal-v3 global feature schema mismatch")
    train_indices = np.flatnonzero(split == 0)
    validation_indices = np.flatnonzero(split == 1)
    test_indices = np.flatnonzero(split == 2)
    if opts.max_train_events is not None:
        train_indices = train_indices[: opts.max_train_events]
    if opts.max_validation_events is not None:
        validation_indices = validation_indices[: opts.max_validation_events]
    campaign = dict(campaign)
    campaign["manifest"] = str(opts.campaign_manifest)
    manifest = json.loads(opts.campaign_manifest.read_text())
    analysis_weights = physics_weights(
        events, campaign, json.loads(opts.xsec.read_text())
    )
    process_names = background_process_names(events, manifest)
    trial_configs = load_trial_configs(opts.trials_json)
    if opts.trial is not None:
        requested = set(opts.trial)
        known = {str(row["name"]) for row in trial_configs}
        if not requested <= known:
            raise RuntimeError(f"unknown trials: {sorted(requested - known)}")
        trial_configs = [row for row in trial_configs if row["name"] in requested]

    split_audit = {
        "train": {
            "events": int(len(train_indices)),
            "signal": int(events.labels[train_indices].sum()),
        },
        "validation": {
            "events": int(len(validation_indices)),
            "signal": int(events.labels[validation_indices].sum()),
        },
        "test": {
            "events": int(len(test_indices)),
            "signal": int(events.labels[test_indices].sum()),
        },
    }
    audit = {
        "status": "validation_tuning",
        "schema_version": "lowdm_diagonal_v3_significance_campaign_v1",
        "device": device.type,
        "selection": (
            "Low-dM preselection && Nt=0 && NW=0 && Nres=0 && Nb>=1; "
            "no MET/sqrt(HT), NISR, ISR-dphi, or !feature_SR requirement"
        ),
        "global_feature_names": DIAGONAL_V3_GLOBAL_FEATURE_NAMES,
        "split_2_1_7": split_audit,
        "test_sealed": True,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
        "trials": [row["name"] for row in trial_configs],
        "epochs_target": opts.epochs,
        "provenance": {
            "cache": str(opts.cache),
            "campaign_manifest_sha256": sha256(opts.campaign_manifest),
            "xsec_sha256": sha256(opts.xsec),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "uproot": uproot.__version__,
        },
    }
    if opts.baseline_checkpoint is not None:
        if not opts.baseline_checkpoint.is_file():
            raise FileNotFoundError(opts.baseline_checkpoint)
        audit["preserved_baseline_checkpoint"] = {
            "path": str(opts.baseline_checkpoint),
            "sha256": sha256(opts.baseline_checkpoint),
            "policy": "read-only reference; never overwritten or warm-started",
        }
    write_json(opts.output / "experiment_audit.json", audit)
    summaries = [
        train_trial(
            opts,
            config,
            events,
            train_indices,
            validation_indices,
            analysis_weights,
            process_names,
            device,
        )
        for config in trial_configs
    ]
    ranked = sorted(
        summaries,
        key=lambda summary: tuple(summary["selection_key"]),
        reverse=True,
    )
    best_summary = ranked[0]
    best_name = str(best_summary["config"]["name"])
    checkpoint = torch.load(
        opts.output / "trials" / best_name / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = best_summary["config"]
    model = PhysicsInformedJetGraphClassifier(
        node_features=events.node_features.shape[2],
        global_features=events.global_features.shape[1],
        hidden=int(config["hidden"]),
        message_layers=int(config["message_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    validation_weights = physics_informed_loss_weights(
        events, validation_indices, analysis_weights
    )
    validation_data = make_dataset(events, validation_indices, validation_weights)
    validation_scores = predict(
        model, validation_data, int(config["batch_size"]), device
    )
    binning = optimize_validation_binning(
        events,
        validation_indices,
        validation_scores,
        analysis_weights,
    )
    write_json(opts.output / "validation_binning.json", binning)
    selection = {
        "status": "frozen_before_test",
        "schema_version": "lowdm_diagonal_v3_selection_v1",
        "best_trial": best_name,
        "best_epoch": int(checkpoint["epoch"]),
        "model_selection_key": best_summary["selection_key"],
        "score_edges": binning["best"]["edges"],
        "score_bins_per_category": binning["best"]["score_bins_per_category"],
        "categories": (
            "Nb1_NISR0",
            "Nb1_NISR1plus",
            "Nb2plus_NISR0",
            "Nb2plus_NISR1plus",
        ),
        "binning_selection_key": binning_selection_key(binning["best"]),
        "test_touched": False,
    }
    write_json(opts.output / "selection.json", selection)
    write_json(
        opts.output / "validation_tuning_summary.json",
        {
            "status": "complete_validation_only",
            "best_trial": best_name,
            "ranked_trials": [row["config"]["name"] for row in ranked],
            "trial_summaries": summaries,
            "selection": selection,
            "test_touched": False,
        },
    )
    audit.update(
        status="complete_validation_only",
        best_trial=best_name,
        best_epoch=int(checkpoint["epoch"]),
        test_sealed=True,
    )
    write_json(opts.output / "experiment_audit.json", audit)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
