#!/usr/bin/env python3
"""Plot train/validation loss curves for every hyperparameter candidate."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


COLORS = ("#0057FF", "#E60000", "#00A000", "#E000E0", "#00AACC",
          "#FF7000", "#6A00A8", "#111111", "#D4A000", "#7A3E00")
MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*", "h", "<")


def archived_histories(pdf: Path, campaign: Path) -> tuple[dict, dict]:
    """Recover only displayed epoch coordinates; never manufacture raw logs."""
    from .supplementary_plots import file_sha256, pdf_polylines

    paths, text = pdf_polylines(pdf)
    summary_path = campaign / "validation_tuning_summary.json"
    summary = json.loads(summary_path.read_text())
    trials = {row["config"]["name"]: row for row in summary["trial_summaries"]}
    if len(trials) != 8:
        raise ValueError("Archived recovery expects the frozen eight-candidate comparison")
    x_text, remaining = text.split("Epoch", 1)
    y_text = remaining.split("Weighted binary cross entropy", 1)[0]
    ticks = [np.array([float(x) for x in re.findall(r"\d+(?:\.\d+)?", chunk)])
             for chunk in (x_text, y_text)]
    affine = []
    for dimension, values in enumerate(ticks):
        positions = sorted({p["vertices"][0][dimension] for p in paths
                            if len(p["vertices"]) == 2
                            and np.allclose(p["color"], (0.6901960784,) * 3)
                            and p["vertices"][0][dimension] == p["vertices"][1][dimension]})
        if len(positions) != len(values) or len(values) < 2:
            raise ValueError("Cannot identify archived linear-axis ticks")
        scale = (values[-1] - values[0]) / (positions[-1] - positions[0])
        offset = values[0] - scale * positions[0]
        if np.max(np.abs(scale * np.array(positions) + offset - values)) > 1e-7:
            raise ValueError("Archived axis ticks are not consistent with a linear transform")
        affine.append((scale, offset))

    histories, checks = {}, {}
    for index, name in enumerate(sorted(trials)):
        trial = trials[name]
        selected = [p for p in paths if len(p["vertices"]) > 10
                    and np.allclose(p["color"], to_rgb(COLORS[index]), atol=1e-7)]
        if len(selected) != 2 or sorted(bool(p["dash"]) for p in selected) != [False, True]:
            raise ValueError(f"Expected train-solid/validation-dashed pair: {name}")
        rows = {}
        for path in selected:
            xy = np.asarray(path["vertices"])
            epochs = xy[:, 0] * affine[0][0] + affine[0][1]
            expected = np.arange(1, int(trial["epochs_run"]) + 1)
            if len(epochs) != len(expected) or np.max(np.abs(epochs - expected)) > 1e-7:
                raise ValueError(f"Epoch count/positions disagree with frozen summary: {name}")
            values = xy[:, 1] * affine[1][0] + affine[1][1]
            key = "validation_weighted_bce" if path["dash"] else "training_weighted_bce"
            for epoch, value in zip(expected, values):
                rows.setdefault(int(epoch), {"epoch": int(epoch)})[key] = float(value)
        history = [rows[epoch] for epoch in sorted(rows)]
        best = trial["best"]
        best_error = max(abs(rows[int(best["epoch"])][key] - float(best[key]))
                         for key in ("training_weighted_bce", "validation_weighted_bce"))
        if best_error > 1e-8:
            raise ValueError(f"Color-to-candidate mapping fails frozen best-epoch comparison: {name}")
        checks[name] = {"epochs": len(history), "color": COLORS[index],
                        "marker": MARKERS[index], "best_epoch": int(best["epoch"]),
                        "max_best_epoch_bce_difference": best_error}
        original = campaign / "trials" / name / "training_history.json"
        if original.exists():
            raw = json.loads(original.read_text())["history"]
            if len(raw) != len(history):
                raise ValueError(f"Remaining original history length mismatch: {name}")
            error = max(abs(a[key] - b[key]) for a, b in zip(history, raw)
                        for key in ("training_weighted_bce", "validation_weighted_bce"))
            if error > 1e-8 or any(a["epoch"] != b["epoch"] for a, b in zip(history, raw)):
                raise ValueError(f"Recovered curve differs from surviving raw history: {name}")
            checks[name]["full_history_comparison"] = {
                "path": str(original.resolve()), "sha256": file_sha256(original),
                "max_bce_difference": error}
        histories[name] = {"config": trial["config"], "history": history}
    return histories, {
        "source_pdf": str(pdf.resolve()), "source_pdf_sha256": file_sha256(pdf),
        "source_summary": str(summary_path.resolve()), "source_summary_sha256": file_sha256(summary_path),
        "method": "Archived PDF vector vertices transformed by its printed linear-axis ticks; no interpolation, smoothing, retraining, or performance re-evaluation",
        "limitation": "Display coordinates only, not restored raw training logs. Seven candidates have no surviving per-epoch raw history; their color/config mapping is checked against frozen best-epoch values.",
        "quantity": "Weighted BCE component, not the combined BCE plus significance training objective",
        "validation": checks,
    }


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
    *,
    histories: dict | None = None,
    render_metadata: dict | None = None,
) -> int:
    hep.style.use("CMS")
    plt.rcParams["path.simplify"] = False
    if histories is None:
        histories = load_histories(campaign, reference_history, reference_label)
    if not histories:
        return 0

    fig, axis = plt.subplots(figsize=(15.5, 7.2))
    candidate_handles: list[Line2D] = []
    for index, (name, payload) in enumerate(histories.items()):
        history = payload["history"]
        color = COLORS[index % len(COLORS)]
        marker = MARKERS[index % len(MARKERS)]
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
    if render_metadata is not None:
        render_metadata.update({"xlim": list(axis.get_xlim()), "ylim": list(axis.get_ylim())})
    plt.close(fig)
    return len(histories)


def replot_archive(pdf: Path, campaign: Path, output: Path) -> dict:
    from .supplementary_plots import file_sha256, pdf_polylines

    histories, manifest = archived_histories(pdf, campaign)
    metadata = {}
    save_plot(campaign, output, histories=histories, render_metadata=metadata)
    paths, _ = pdf_polylines(output.with_suffix(".pdf"))
    original_markers = _pdf_marker_forms(pdf)
    output_markers = _pdf_marker_forms(output.with_suffix(".pdf"))
    for index, (name, payload) in enumerate(histories.items()):
        selected = [p for p in paths if len(p["vertices"]) > 10
                    and np.allclose(p["color"], to_rgb(COLORS[index]), atol=1e-7)]
        if len(selected) != 2:
            raise ValueError(f"Output curve count mismatch: {name}")
        errors = []
        for path in selected:
            x, y, width, height = path["clip"]
            xy = (np.asarray(path["vertices"]) - (x, y)) / (width, height)
            xy = xy * (np.diff(metadata["xlim"])[0], np.diff(metadata["ylim"])[0])
            xy += (metadata["xlim"][0], metadata["ylim"][0])
            key = "validation_weighted_bce" if path["dash"] else "training_weighted_bce"
            expected = np.array([(r["epoch"], r[key]) for r in payload["history"]])
            if xy.shape != expected.shape:
                raise ValueError(f"Output vertex count mismatch: {name}")
            error = np.max(np.abs(xy - expected), axis=0)
            if error[0] > 1e-7 or error[1] > 1e-8:
                raise ValueError(f"Output vector round-trip failed: {name}: {error}")
            errors.append(error.tolist())
        manifest["validation"][name]["output_vector_max_differences_epoch_bce"] = errors
        color_key = tuple(round(x, 7) for x in to_rgb(COLORS[index]))
        before = original_markers.get(color_key)
        after = output_markers.get(color_key)
        if (before is None or before != after
                or sum(before.values()) != 2 * len(payload["history"])):
            raise ValueError(f"Marker geometry, size, color or count changed: {name}")
        manifest["validation"][name]["marker_geometry_sha256_and_count"] = before
    coordinates = output.with_suffix(".coordinates.json")
    coordinates.write_text(json.dumps(histories, indent=2) + "\n")
    manifest["curve_data"] = {"path": str(coordinates.resolve()), "sha256": file_sha256(coordinates)}
    manifest["artifacts"] = {suffix: {"path": str(output.with_suffix('.' + suffix).resolve()),
                                      "sha256": file_sha256(output.with_suffix('.' + suffix))}
                             for suffix in ("pdf", "png")}
    manifest["render"] = metadata
    manifest["renderer"] = {"path": str(Path(__file__).resolve()),
                            "sha256": file_sha256(Path(__file__)),
                            "matplotlib": matplotlib.__version__, "mplhep": hep.__version__}
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _pdf_marker_forms(pdf: Path) -> dict:
    """Hash drawn Matplotlib marker forms by stroke color, preserving size."""
    import hashlib
    from pypdf import PdfReader

    page = PdfReader(pdf).pages[0]
    objects = page["/Resources"]["/XObject"]
    color, stack, result = (0.0, 0.0, 0.0), [], {}
    for operands, operator in page.get_contents().operations:
        if operator == b"q":
            stack.append(color)
        elif operator == b"Q":
            color = stack.pop()
        elif operator == b"RG":
            color = tuple(round(float(x), 7) for x in operands)
        elif operator == b"G":
            color = (round(float(operands[0]), 7),) * 3
        elif operator == b"Do" and re.fullmatch(r"/M\d+", str(operands[0])):
            digest = hashlib.sha256(objects[operands[0]].get_object().get_data()).hexdigest()
            counts = result.setdefault(color, {})
            counts[digest] = counts.get(digest, 0) + 1
    return result


def campaign_complete(campaign: Path) -> bool:
    path = campaign / "validation_tuning_summary.json"
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("status") == "complete"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-pdf", type=Path, help="Explicit archived-PDF recovery when candidate raw histories are unavailable")
    parser.add_argument("--reference-history", type=Path)
    parser.add_argument(
        "--reference-label", default="original_core_h48_l2_rank010"
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    opts = parser.parse_args()
    output = opts.output or opts.campaign / "all_candidates_train_validation_loss"

    if opts.source_pdf:
        if opts.watch or opts.reference_history:
            parser.error("Archived-PDF recovery cannot be combined with watch/reference history")
        if opts.source_pdf.resolve() == output.with_suffix(".pdf").resolve():
            parser.error("Archived source PDF must not be overwritten")
        manifest = replot_archive(opts.source_pdf, opts.campaign, output)
        print(json.dumps({"candidates_plotted": len(manifest["validation"]),
                          "manifest": str(output.with_suffix('.manifest.json'))}))
        return 0

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
