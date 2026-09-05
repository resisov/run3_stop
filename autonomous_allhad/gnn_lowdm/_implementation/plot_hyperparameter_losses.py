#!/usr/bin/env python3
"""Plot train/validation loss curves for every hyperparameter candidate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.lines import Line2D


DISPLAY_LABELS = {
    "original_core_h48_l2_rank010": (
        r"epochs $12$, lr $4\times10^{-4}$, dropout $0.25$, WD $10^{-3}$, "
        r"rank $0.10$, hard $0.05$"
    ),
    "hp00_baseline": (
        r"lr $4\times10^{-4}$, dropout $0.25$, WD $10^{-3}$, "
        r"rank $0.10$, hard $0.05$"
    ),
    "hp01_lr2p5e4": (
        r"lr $2.5\times10^{-4}$, dropout $0.25$, WD $10^{-3}$, "
        r"rank $0.10$, hard $0.05$"
    ),
    "hp02_lr6e4": (
        r"lr $6\times10^{-4}$, dropout $0.25$, WD $10^{-3}$, "
        r"rank $0.10$, hard $0.05$"
    ),
    "hp03_drop020_wd5e4": (
        r"lr $4\times10^{-4}$, dropout $0.20$, WD $5\times10^{-4}$, "
        r"rank $0.10$, hard $0.05$"
    ),
    "hp04_drop030_wd2e3": (
        r"lr $4\times10^{-4}$, dropout $0.30$, WD $2\times10^{-3}$, "
        r"rank $0.10$, hard $0.05$"
    ),
    "hp05_pair005": (
        r"lr $4\times10^{-4}$, dropout $0.25$, WD $10^{-3}$, "
        r"rank $0.05$, hard $0.05$"
    ),
    "hp06_pair020": (
        r"lr $4\times10^{-4}$, dropout $0.25$, WD $10^{-3}$, "
        r"rank $0.20$, hard $0.05$"
    ),
    "hp07_hard002": (
        r"lr $4\times10^{-4}$, dropout $0.25$, WD $10^{-3}$, "
        r"rank $0.10$, hard $0.02$"
    ),
    "hp08_hard010": (
        r"lr $4\times10^{-4}$, dropout $0.25$, WD $10^{-3}$, "
        r"rank $0.10$, hard $0.10$"
    ),
}


def load_histories(
    campaign: Path,
    reference_history: Path | None = None,
    reference_label: str = "original_core_h48_l2_rank010",
) -> dict[str, dict[str, object]]:
    histories: dict[str, dict[str, object]] = {}
    if reference_history is not None:
        payload = json.loads(reference_history.read_text())
        history = payload.get("history", [])
        if history:
            histories[reference_label] = {"history": history, "config": None}
    for path in sorted((campaign / "trials").glob("*/training_history.json")):
        payload = json.loads(path.read_text())
        history = payload.get("history", [])
        if history:
            histories[path.parent.name] = {
                "history": history,
                "config": payload.get("config"),
            }
    return histories


def scientific(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(f"{value:e}".split("e")[1])
    coefficient = value / (10.0**exponent)
    return rf"{coefficient:g}\times10^{{{exponent}}}"


def candidate_label(name: str, config: object) -> str:
    if not isinstance(config, dict):
        return DISPLAY_LABELS.get(name, name)
    return (
        rf"hidden ${int(config['hidden'])}$, layers ${int(config['message_layers'])}$, "
        rf"lr ${scientific(float(config['learning_rate']))}$, "
        rf"dropout ${float(config['dropout']):g}$, "
        rf"WD ${scientific(float(config['weight_decay']))}$, "
        rf"$\lambda_{{S/\sqrt{{B}}}}={float(config['significance_weight']):g}$, "
        rf"$T={float(config['significance_temperature']):g}$"
    )


def save_plot(
    campaign: Path,
    output: Path,
    reference_history: Path | None = None,
    reference_label: str = "original_core_h48_l2_rank010",
) -> int:
    hep.style.use("CMS")
    histories = load_histories(campaign, reference_history, reference_label)
    if not histories:
        return 0

    fig, axis = plt.subplots(figsize=(15.5, 7.2))
    colors = (
        "#0057FF",  # vivid blue
        "#E60000",  # vivid red
        "#00A000",  # vivid green
        "#E000E0",  # magenta
        "#00AACC",  # cyan
        "#FF7000",  # orange
        "#6A00A8",  # purple
        "#111111",  # black
        "#D4A000",  # golden yellow
        "#7A3E00",  # brown
    )
    markers = ("o", "s", "^", "v", "D", "P", "X", "*", "h", "<")
    candidate_handles: list[Line2D] = []
    for index, (name, payload) in enumerate(histories.items()):
        history = payload["history"]
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        display_label = candidate_label(name, payload.get("config"))
        epochs = [int(row["epoch"]) for row in history]
        train = [float(row["training_weighted_bce"]) for row in history]
        validation = [float(row["validation_weighted_bce"]) for row in history]
        axis.plot(
            epochs,
            train,
            color=color,
            linestyle="-",
            marker=marker,
            markersize=7.0,
            linewidth=1.7,
        )
        axis.plot(
            epochs,
            validation,
            color=color,
            linestyle="--",
            marker=marker,
            markersize=7.0,
            linewidth=1.7,
        )
        candidate_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                markersize=7.0,
                lw=2,
                label=display_label,
            )
        )

    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            lw=1.8,
            label="Train",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            lw=1.8,
            label="Validation",
        ),
    ]
    axis.set_xlabel("Epoch", fontsize=15)
    axis.set_ylabel("Weighted binary cross entropy", fontsize=15)
    axis.tick_params(axis="both", labelsize=13)
    axis.grid(alpha=0.22)
    axis.xaxis.get_major_locator().set_params(integer=True)
    first_legend = axis.legend(
        handles=candidate_handles,
        frameon=False,
        fontsize=13,
        ncol=1,
        loc="upper right",
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=style_handles,
        frameon=False,
        fontsize=13,
        loc="lower left",
    )
    hep.cms.label(
        llabel="Simulation",
        rlabel="(13.6 TeV)",
        ax=axis,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return len(histories)


def campaign_complete(campaign: Path) -> bool:
    path = campaign / "validation_tuning_summary.json"
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("status") == "complete"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-history", type=Path)
    parser.add_argument(
        "--reference-label", default="original_core_h48_l2_rank010"
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    opts = parser.parse_args()
    output = opts.output or opts.campaign / "all_candidates_train_validation_loss"

    while True:
        count = save_plot(
            opts.campaign,
            output,
            opts.reference_history,
            opts.reference_label,
        )
        print(
            json.dumps(
                {
                    "campaign": str(opts.campaign),
                    "candidates_plotted": count,
                    "complete": campaign_complete(opts.campaign),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not opts.watch or campaign_complete(opts.campaign):
            break
        time.sleep(max(opts.interval_seconds, 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
