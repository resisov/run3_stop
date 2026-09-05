"""Rebuild supplementary training, ROC, and global-SHAP figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from .analyze_diagonal_v3_global_shap import (
    FEATURE_LABELS,
    display_feature_view,
    plot_beeswarm,
    plot_importance,
)


PACKAGE = Path(__file__).resolve().parent.parent
MODEL = PACKAGE / "models/diagonal_v3_h48_l3_sig010"
RESULT = PACKAGE / "results/diagonal_v3_significance_full_20260831"
DEFAULT_OUTPUT = RESULT / "supplementary"


def _selected_epoch() -> int:
    configuration = json.loads((PACKAGE / "config.json").read_text())
    return int(configuration["model"]["selected_epoch"])


def _save(figure: plt.Figure, output: Path) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for suffix in ("png", "pdf"):
        path = output.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=220, bbox_inches="tight")
        artifacts[suffix] = str(path)
    plt.close(figure)
    return artifacts


def _cms(axis: plt.Axes) -> None:
    hep.cms.label(llabel="Simulation", rlabel="(13.6 TeV)", ax=axis)


def _line_plot(
    epochs: np.ndarray,
    curves: tuple[tuple[np.ndarray, str, str, str, str], ...],
    ylabel: str,
    output: Path,
    selected_epoch: int,
) -> dict[str, str]:
    hep.style.use("CMS")
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    for values, label, color, style, marker in curves:
        axis.plot(
            epochs,
            values,
            label=label,
            color=color,
            linestyle=style,
            marker=marker,
            markersize=7,
            linewidth=2.2,
        )
    axis.axvline(selected_epoch, color="0.35", linestyle=":", linewidth=1.6)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_xticks(epochs)
    axis.tick_params(labelsize=14)
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, fontsize=14)
    _cms(axis)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return _save(figure, output)


def training_curves(history_path: Path, output: Path) -> dict[str, object]:
    payload = json.loads(history_path.read_text())
    rows = payload["history"]
    epochs = np.asarray([row["epoch"] for row in rows], dtype=int)
    selected_epoch = _selected_epoch()
    artifacts = {
        "loss": _line_plot(
            epochs,
            (
                (np.asarray([row["training_weighted_bce"] for row in rows]), "Train", "blue", "-", "o"),
                (np.asarray([row["validation_weighted_bce"] for row in rows]), "Validation", "red", "--", "s"),
            ),
            "Weighted binary cross-entropy",
            output / "best_model_train_validation_loss",
            selected_epoch,
        ),
        "accuracy": _line_plot(
            epochs,
            (
                (np.asarray([row["training_weighted_accuracy"] for row in rows]), "Train", "blue", "-", "o"),
                (np.asarray([row["validation_weighted_accuracy"] for row in rows]), "Validation", "red", "--", "s"),
            ),
            "Weighted accuracy",
            output / "best_model_train_validation_accuracy",
            selected_epoch,
        ),
        "auc": _line_plot(
            epochs,
            (
                (np.asarray([row["macro_mass_auc"] for row in rows]), "Validation: all backgrounds", "blue", "-", "o"),
                (np.asarray([row["macro_top_background_auc"] for row in rows]), "Validation: top background", "red", "--", "s"),
            ),
            "Mass-point-averaged AUC",
            output / "best_model_validation_auc",
            selected_epoch,
        ),
    }
    return {"source": str(history_path), "selected_epoch": selected_epoch, "artifacts": artifacts}


def _weighted_roc(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    labels = np.asarray(labels, dtype=bool)
    weights = np.abs(np.asarray(weights, dtype=float))
    if not np.any(labels) or not np.any(~labels):
        raise ValueError("ROC needs both signal and background events")
    order = np.argsort(scores, kind="stable")[::-1]
    labels = labels[order]
    weights = weights[order]
    tpr = np.cumsum(weights * labels) / np.sum(weights[labels])
    fpr = np.cumsum(weights * ~labels) / np.sum(weights[~labels])
    tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.r_[0.0, fpr, 1.0]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def roc_curve(scores_path: Path, output: Path) -> dict[str, object]:
    with np.load(scores_path, allow_pickle=False) as source:
        scores = np.asarray(source["scores"], dtype=float)
        labels = np.asarray(source["labels"], dtype=bool)
        required = {"weights", "signal_topology_id"}
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError(
                f"{scores_path} is a legacy score archive missing {missing}; "
                "rerun 'python -m autonomous_allhad.gnn_lowdm.eval test' "
                "before rebuilding the weighted topology ROC"
            )
        weights = np.asarray(source["weights"], dtype=float)
        topology = np.asarray(source["signal_topology_id"], dtype=int)

    hep.style.use("CMS")
    figure, axis = plt.subplots(figsize=(8.3, 7.4))
    definitions = [("All signal", np.ones(len(scores), dtype=bool), "black", "-")]
    for identifier, name, color in (
        (1, "T2tt", "red"),
        (2, "T2bW", "blue"),
        (3, "T2tb", "green"),
    ):
        definitions.append((name, ~labels | (topology == identifier), color, "--"))
    curves = {}
    for name, use, color, style in definitions:
        fpr, tpr, auc = _weighted_roc(labels[use], scores[use], weights[use])
        axis.plot(fpr, tpr, color=color, linestyle=style, linewidth=2.2, label=f"{name}: AUC = {auc:.4f}")
        curves[name] = auc
    axis.plot((0, 1), (0, 1), color="0.5", linestyle=":", linewidth=1.3)
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.02), xlabel="Background efficiency", ylabel="Signal efficiency")
    axis.tick_params(labelsize=13)
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, fontsize=12, loc="lower right")
    _cms(axis)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    artifacts = _save(figure, output / "independent_test_roc_auc")
    return {
        "source": str(scores_path),
        "weighting": "absolute event weights",
        "auc": curves,
        "artifacts": artifacts,
    }


def shap_plots(values_path: Path, output: Path, top: int, seed: int) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    with np.load(values_path, allow_pickle=False) as source:
        attributions = np.asarray(source["shap_values"], dtype=float)
        values = np.asarray(source["feature_values"], dtype=float)
        names = [str(item) for item in source["feature_names"]]
    if len(names) != len(FEATURE_LABELS):
        raise ValueError("SHAP feature schema does not match the frozen 40-feature model")
    display_shap, display_values, display_names, display_labels, colors = display_feature_view(attributions, values)
    importance, order = plot_importance(output / "global_shap_importance", display_shap, display_labels, colors)
    beeswarm = plot_beeswarm(
        output / "global_shap_beeswarm",
        display_shap,
        display_values,
        order,
        display_labels,
        top=top,
        seed=seed,
    )
    return {
        "source": str(values_path),
        "scope": "40 global inputs only",
        "display_features": display_names,
        "importance": importance,
        "beeswarm": beeswarm,
    }


def _write_manifest(output: Path, section: str, payload: dict[str, object]) -> int:
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / f"{section}_plot_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "manifest": str(manifest)}, sort_keys=True))
    return 0


def main_training_curves() -> int:
    parser = argparse.ArgumentParser(description="Plot best-model train/validation histories.")
    parser.add_argument("--history", type=Path, default=MODEL / "training_history.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return _write_manifest(args.output, "training", training_curves(args.history, args.output))


def main_roc() -> int:
    parser = argparse.ArgumentParser(description="Plot independent-test ROC and AUC.")
    parser.add_argument("--scores", type=Path, default=RESULT / "test/diagonal_v3_test_scores.npz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return _write_manifest(args.output, "roc", roc_curve(args.scores, args.output))


def main_shap() -> int:
    parser = argparse.ArgumentParser(description="Plot global-feature SHAP summaries.")
    parser.add_argument("--values", type=Path, default=RESULT / "global_shap/diagonal_v3_global_gradientshap_values.npz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9022026)
    args = parser.parse_args()
    return _write_manifest(args.output, "shap", shap_plots(args.values, args.output, args.top, args.seed))
