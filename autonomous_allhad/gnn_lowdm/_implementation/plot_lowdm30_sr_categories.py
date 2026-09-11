#!/usr/bin/env python3
"""Plot the adopted 30-bin Low-dM SR with off-shell T2bW benchmarks."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 1.4
import matplotlib.pyplot as plt


LUMINOSITY_FB = 109.82
HIGHDM_MAIN_YLIM = (0.03, 1819951.9758889896)
CATEGORIES = (
    "SR_Nb1_NISR0",
    "SR_Nb1_NISR1",
    "SR_Nb1_NISR2plus",
    "SR_Nb2plus_NISR0",
    "SR_Nb2plus_NISR1",
    "SR_Nb2plus_NISR2plus",
)
CATEGORY_LABELS = {
    "SR_Nb1_NISR0": r"$N_b=1$, $N_{\mathrm{ISR}}=0$",
    "SR_Nb1_NISR1": r"$N_b=1$, $N_{\mathrm{ISR}}=1$",
    "SR_Nb1_NISR2plus": r"$N_b=1$, $N_{\mathrm{ISR}}\geq2$",
    "SR_Nb2plus_NISR0": r"$N_b\geq2$, $N_{\mathrm{ISR}}=0$",
    "SR_Nb2plus_NISR1": r"$N_b\geq2$, $N_{\mathrm{ISR}}=1$",
    "SR_Nb2plus_NISR2plus": r"$N_b\geq2$, $N_{\mathrm{ISR}}\geq2$",
}
PROCESS_ORDER = ("VV", "Top", "DY", "Photon", "W", "Zinv", "QCD")
PROCESS_LABELS = {
    "VV": "VV+VVV",
    "Top": "Top",
    "DY": "DY",
    "Photon": "Photon+jet",
    "W": r"$W\to\ell\nu$",
    "Zinv": r"$Z\to\nu\nu$",
    "QCD": "QCD Multijet",
}
PROCESS_COLORS = {
    "VV": "#6F7661",
    "Top": "#7A9FC2",
    "DY": "#35B6B4",
    "Photon": "#8E3B9E",
    "W": "#D9C6A5",
    "Zinv": "#E6A84F",
    "QCD": "#C995A2",
}
BENCHMARKS = (
    (
        "signal_mStop800_mLSP650",
        r"$m_{\tilde{t}}=800$ GeV, $m_{\tilde{\chi}^{0}_{1}}=650$ GeV",
        "#E60000",
    ),
    (
        "signal_mStop1000_mLSP850",
        r"$m_{\tilde{t}}=1000$ GeV, $m_{\tilde{\chi}^{0}_{1}}=850$ GeV",
        "#0057FF",
    ),
)


def signed_stack(axis, values, edges, labels, colors) -> None:
    positive_bottom = np.zeros(len(edges) - 1)
    negative_bottom = np.zeros(len(edges) - 1)
    for current, label, color in zip(values, labels, colors):
        positive = np.clip(current, 0.0, None)
        negative = np.clip(current, None, 0.0)
        axis.stairs(
            positive_bottom + positive,
            edges,
            baseline=positive_bottom,
            fill=True,
            color=color,
            edgecolor="black",
            linewidth=0.65,
            label=label,
        )
        axis.stairs(
            negative_bottom + negative,
            edges,
            baseline=negative_bottom,
            fill=True,
            color=color,
            edgecolor="black",
            linewidth=0.65,
        )
        positive_bottom += positive
        negative_bottom += negative


def score_interval(low: float, high: float, final: bool) -> str:
    return f"[{low:.3g}, {high:.3g}{']' if final else ')'}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--templates",
        type=Path,
        nargs="+",
        help="one or more same-binning yearly template ROOT files to sum",
    )
    inputs.add_argument("--payloads", type=Path, nargs="+",
                        help="canonical SR model JSON exports; no local ROOT input")
    parser.add_argument("--binning", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--luminosity-fb", type=float, default=LUMINOSITY_FB)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    process_values = {process: [] for process in PROCESS_ORDER}
    process_variances = {process: [] for process in PROCESS_ORDER}
    signal_values = {key: [] for key, _, _ in BENCHMARKS}
    binning = json.loads(args.binning.read_text())["best_by_scheme"]["nb2_nisr3"]
    expected_labels = [category.removeprefix("SR_") for category in CATEGORIES]
    if binning["category_labels"] != expected_labels:
        raise RuntimeError("validation-frozen category order does not match template")
    raw_edges = {
        "SR_" + category: np.asarray(edges, dtype=float)
        for category, edges in binning["edges_by_category"].items()
    }
    exports = [json.loads(path.read_text()) for path in args.payloads or []]
    if exports:
        if any(item.get("schema_version") != "canonical_sr_plot_payload_v1"
               or item.get("sr_observations_included") is not False
               or item.get("prediction_stage") != "prefit"
               or item.get("rate_initials_applied") is not True
               or item.get("uncertainty") != "mc_statistical_only"
               or item.get("topology") != "T2bW" for item in exports):
            raise RuntimeError("invalid blinded T2bW SR plot payload")
        mapping = {"VV": "VV_VVV", "Top": "Top", "DY": "DY", "Photon": "PhotonJet",
                   "W": "WtoLNu", "Zinv": "Zto2Nu", "QCD": "QCD"}
        for category in CATEGORIES:
            yearly = []
            for item in exports:
                rows = [row for row in item["bin_map"]["lowdm"] if "SR_" + row["category"] == category]
                if [row["score_bin"] for row in rows] != list(range(5)):
                    raise RuntimeError(f"invalid score-bin order: {category}")
                np.testing.assert_array_equal([rows[0]["score_edges"][0]] + [row["score_edges"][1] for row in rows], raw_edges[category])
                yearly.append([item["channels"][row["channel"]] for row in rows])
            for process in PROCESS_ORDER:
                prefix = mapping[process]
                for field, target in (("sumw", process_values), ("sumw2", process_variances)):
                    values = np.zeros(5)
                    for directories in yearly:
                        for index, directory in enumerate(directories):
                            values[index] += sum(float(leaf[field][0]) for name, leaf in directory.items()
                                                 if name == prefix or name.startswith(prefix + "_"))
                    target[process].append(values)
            for key, _, _ in BENCHMARKS:
                name = key.replace("signal_", "sig_", 1)
                signal_values[key].append(np.sum([[float(directory[name]["sumw"][0])
                                                   for directory in directories] for directories in yearly], axis=0))
    with ExitStack() as stack:
        root_files = [stack.enter_context(uproot.open(path)) for path in args.templates or []]
        for category in CATEGORIES if root_files else []:
            for process in PROCESS_ORDER:
                value_sum = None
                variance_sum = None
                for root_file in root_files:
                    histogram = root_file[f"{category}/{process}"]
                    values = np.asarray(histogram.values(), dtype=float)
                    variance = histogram.variances()
                    if variance is None:
                        raise RuntimeError(
                            f"missing sumw2 for {category}/{process}"
                        )
                    current_variance = np.asarray(variance, dtype=float)
                    value_sum = values.copy() if value_sum is None else value_sum + values
                    variance_sum = (
                        current_variance.copy()
                        if variance_sum is None
                        else variance_sum + current_variance
                    )
                process_values[process].append(value_sum)
                process_variances[process].append(variance_sum)
            for key, _, _ in BENCHMARKS:
                value_sum = None
                for root_file in root_files:
                    path = f"{category}/{key}"
                    if path not in root_file:
                        path = f"{category}/{key.replace('signal_', 'signal_T2bW_')}"
                    if path not in root_file:
                        raise RuntimeError(f"missing T2bW benchmark histogram: {path}")
                    values = np.asarray(root_file[path].values(), dtype=float)
                    value_sum = values.copy() if value_sum is None else value_sum + values
                signal_values[key].append(value_sum)

    processes = {
        key: np.concatenate(value) for key, value in process_values.items()
    }
    variances = {
        key: np.concatenate(value) for key, value in process_variances.items()
    }
    signals = {
        key: np.concatenate(value) for key, value in signal_values.items()
    }
    background = np.sum([processes[key] for key in PROCESS_ORDER], axis=0)
    background_variance = np.sum(
        [variances[key] for key in PROCESS_ORDER], axis=0
    )
    uncertainty = np.sqrt(np.clip(background_variance, 0.0, None))
    if len(background) != 30:
        raise RuntimeError(f"expected 30 SR bins, found {len(background)}")

    import mplhep as hep

    hep.style.use("CMS")
    flat_edges = np.arange(0.5, 31.5)
    centers = np.arange(1.0, 31.0)
    figure, (axis, significance_axis) = plt.subplots(
        2,
        1,
        figsize=(20.5, 8.4),
        gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.04},
        sharex=True,
    )
    signed_stack(
        axis,
        [processes[key] for key in PROCESS_ORDER],
        flat_edges,
        [PROCESS_LABELS[key] for key in PROCESS_ORDER],
        [PROCESS_COLORS[key] for key in PROCESS_ORDER],
    )
    lower = np.maximum(background - uncertainty, 1.0e-12)
    upper = np.maximum(background + uncertainty, 1.0e-12)
    axis.fill_between(
        flat_edges,
        np.r_[lower, lower[-1]],
        np.r_[upper, upper[-1]],
        step="post",
        facecolor="0.82",
        edgecolor="0.15",
        hatch="////",
        linewidth=0.0,
        alpha=0.65,
        label="MC stat. unc.",
        zorder=6,
    )
    axis.stairs(background, flat_edges, color="black", linewidth=1.4, zorder=7)

    denominator = np.sqrt(np.maximum(background, 0.0))
    for key, label, color in BENCHMARKS:
        values = signals[key]
        axis.stairs(
            values,
            flat_edges,
            color=color,
            linewidth=2.8,
            linestyle="--",
            label=label,
            zorder=9,
        )
        significance = np.divide(
            np.clip(values, 0.0, None),
            denominator,
            out=np.zeros_like(values),
            where=denominator > 0.0,
        )
        significance_axis.stairs(
            significance,
            flat_edges,
            color=color,
            linewidth=2.6,
            linestyle="--",
        )

    xlabels = []
    for category_index, category in enumerate(CATEGORIES):
        start = category_index * 5
        stop = start + 5
        if category_index:
            boundary = start + 0.5
            axis.axvline(boundary, color="black", linewidth=1.2)
            significance_axis.axvline(boundary, color="black", linewidth=1.2)
        axis.text(
            (start + stop + 1.0) / 2.0,
            0.73,
            CATEGORY_LABELS[category],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=19.0,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.96,
            },
            zorder=20,
        )
        edges = raw_edges[category]
        xlabels.extend(
            score_interval(low, high, index == 4)
            for index, (low, high) in enumerate(
                zip(edges[:-1], edges[1:])
            )
        )

    axis.set_yscale("log")
    axis.set_ylim(*HIGHDM_MAIN_YLIM)
    axis.set_ylabel("Events", fontsize=30)
    significance_axis.axhline(0.0, color="black", linewidth=1.0)
    significance_axis.set_ylim(0.0, 5.0)
    significance_axis.set_yticks([0.0, 2.5, 5.0])
    significance_axis.set_yticklabels(["0.0", "2.5", "5.0"])
    significance_axis.set_ylabel(r"$S/\sqrt{B}$", fontsize=26)
    significance_axis.set_xlabel("GNN output", fontsize=30, loc="right")
    significance_axis.set_xticks(centers)
    significance_axis.set_xticklabels(xlabels, rotation=90)
    for current in (axis, significance_axis):
        current.set_xlim(0.5, 30.5)
        current.tick_params(
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=20,
            length=9,
        )
        current.tick_params(
            which="minor", direction="in", top=True, right=True, length=5
        )
        current.minorticks_on()
    significance_axis.tick_params(axis="x", labelsize=13)

    hep.cms.label(
        llabel="Work in progress",
        rlabel=rf"{args.luminosity_fb:.2f} fb$^{{-1}}$ (13.6 TeV)",
        ax=axis,
    )
    # Keep the blinded SR legend layout identical to the High-dM plot, which
    # retains a DATA legend proxy even though no SR data markers are drawn.
    axis.errorbar(
        [],
        [],
        xerr=[],
        yerr=[],
        fmt="o",
        color="black",
        markersize=5.5,
        capsize=4.5,
        capthick=1.4,
        elinewidth=1.4,
        label="DATA",
    )
    handles, labels = axis.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    desired = [
        *[PROCESS_LABELS[key] for key in reversed(PROCESS_ORDER)],
        "MC stat. unc.",
        *[label for _, label, _ in BENCHMARKS],
        "DATA",
    ]
    axis.legend(
        [by_label[label] for label in desired],
        desired,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        fontsize=12,
        frameon=False,
        columnspacing=1.05,
        handlelength=2.0,
    )
    figure.savefig(args.output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    summary = {
        "schema_version": "lowdm30_sr_category_plot_v1",
        "status": "complete",
        "luminosity_fb": args.luminosity_fb,
        "templates": [str(path.resolve()) for path in args.templates or []],
        "payloads": [str(path.resolve()) for path in args.payloads or []],
        "binning": str(args.binning.resolve()),
        "categories": list(CATEGORIES),
        "score_bins_per_category": 5,
        "total_bins": 30,
        "signal_scale": 1.0,
        "prediction_stage": "prefit" if exports else "template_input",
        "uncertainty": "mc_statistical_only",
        "signals": [
            {
                "topology": "T2bW",
                "mStop": 800,
                "mLSP": 650,
                "deltaM": 150,
                "chargino_minus_lsp": 75.0,
            },
            {
                "topology": "T2bW",
                "mStop": 1000,
                "mLSP": 850,
                "deltaM": 150,
                "chargino_minus_lsp": 75.0,
            },
        ],
        "significance_definition": "S/sqrt(B)",
        "significance_ylim": [0.0, 5.0],
        "main_panel_ylim": list(HIGHDM_MAIN_YLIM),
        "typography_reference": "High-dM 73-bin plot",
        "png": str(args.output.with_suffix(".png").resolve()),
        "pdf": str(args.output.with_suffix(".pdf").resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
