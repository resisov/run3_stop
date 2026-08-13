#!/usr/bin/env python3
"""Compare two signal mass points in the adopted High-/Low-dM SR binning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LUMINOSITY_FB = 109.82
MASS_POINTS = (
    ("mStop1250_mLSP400", r"$m_{\widetilde{t}}=1250$ GeV, $m_{\widetilde{\chi}^{0}_{1}}=400$ GeV", "#E31A1C"),
    ("mStop1300_mLSP400", r"$m_{\widetilde{t}}=1300$ GeV, $m_{\widetilde{\chi}^{0}_{1}}=400$ GeV", "#1F78B4"),
)

# These are the sparse High-dM bins selected for the adjacent-bin merge
# diagnostic.  The boxes are diagnostic annotations only; no histogram
# contents, selections, uncertainties, or statistical-model inputs change.
SUSPECTED_BINS = {
    ("T2bW", "High-dM"): (24, 36, 47, 48, 54),
    ("T2tb", "High-dM"): (24, 42, 53, 59),
}

HIGHDM_CATEGORIES = (
    ("Nb1plus_T0_W0", 6, r"$N_{b}\geq1$, $N_{t}=0$" "\n" r"$N_{W}=0$"),
    ("Nb1plus_T0_W1plus", 6, r"$N_{b}\geq1$, $N_{t}=0$" "\n" r"$N_{W}\geq1$"),
    ("Nb1_T1plus_W0", 6, r"$N_{b}=1$, $N_{t}\geq1$" "\n" r"$N_{W}=0$"),
    ("Nb1_T1plus_W1plus", 6, r"$N_{b}=1$, $N_{t}\geq1$" "\n" r"$N_{W}\geq1$"),
    ("Nb2_T1_W0", 6, r"$N_{b}=2$, $N_{t}=1$" "\n" r"$N_{W}=0$"),
    ("Nb2_T1_W1", 6, r"$N_{b}=2$, $N_{t}=1$" "\n" r"$N_{W}=1$"),
    ("Nb2_Nt2plus_W0", 6, r"$N_{b}=2$, $N_{t}\geq2$" "\n" r"$N_{W}=0$"),
    ("Nb3plus_T1_W0", 6, r"$N_{b}\geq3$, $N_{t}=1$" "\n" r"$N_{W}=0$"),
    ("Nb3plus_T1_W1", 6, r"$N_{b}\geq3$, $N_{t}=1$" "\n" r"$N_{W}=1$"),
    ("Nb3plus_T2_W0", 6, r"$N_{b}\geq3$, $N_{t}=2$" "\n" r"$N_{W}=0$"),
)

HIGHDM_TAIL_MERGED_CATEGORIES = tuple(
    (name, 5, label)
    for name, _size, label in HIGHDM_CATEGORIES
)

LOWDM_CATEGORY_LABELS = {
    "Nb1_PISR300to500_PTb20to40": (
        r"$N_{b}=1$" "\n" r"$300\leq p_{T}^{\mathrm{ISR}}<500$" "\n" r"$20<p_{T}^{b}<40$"
    ),
    "Nb1_PISR300to500_PTb40to70": (
        r"$N_{b}=1$" "\n" r"$300\leq p_{T}^{\mathrm{ISR}}<500$" "\n" r"$40<p_{T}^{b}<70$"
    ),
    "Nb1_PISR500plus_PTb20to40": (
        r"$N_{b}=1$" "\n" r"$p_{T}^{\mathrm{ISR}}\geq500$" "\n" r"$20<p_{T}^{b}<40$"
    ),
    "Nb1_PISR500plus_PTb40to70": (
        r"$N_{b}=1$" "\n" r"$p_{T}^{\mathrm{ISR}}\geq500$" "\n" r"$40<p_{T}^{b}<70$"
    ),
    "Nb2plus_PISR300to500_PTb40to80_Nj2plus": (
        r"$N_{b}\geq2$" "\n" r"$300\leq p_{T}^{\mathrm{ISR}}<500$" "\n" r"$40<p_{T}^{b}<80$"
    ),
    "Nb2plus_PISR300to500_PTb80to140_Nj2plus": (
        r"$N_{b}\geq2$" "\n" r"$300\leq p_{T}^{\mathrm{ISR}}<500$" "\n" r"$80<p_{T}^{b}<140$"
    ),
    "Nb2plus_PISR300to500_PTb140plus_Nj7plus": (
        r"$N_{b}\geq2$, $N_{j}\geq7$" "\n" r"$300\leq p_{T}^{\mathrm{ISR}}<500$" "\n" r"$p_{T}^{b}>140$"
    ),
    "Nb2plus_PISR500plus_PTb40to80_Nj2plus": (
        r"$N_{b}\geq2$" "\n" r"$p_{T}^{\mathrm{ISR}}\geq500$" "\n" r"$40<p_{T}^{b}<80$"
    ),
    "Nb2plus_PISR500plus_PTb80to140_Nj2plus": (
        r"$N_{b}\geq2$" "\n" r"$p_{T}^{\mathrm{ISR}}\geq500$" "\n" r"$80<p_{T}^{b}<140$"
    ),
    "Nb2plus_PISR500plus_PTb140plus_Nj7plus": (
        r"$N_{b}\geq2$, $N_{j}\geq7$" "\n" r"$p_{T}^{\mathrm{ISR}}\geq500$" "\n" r"$p_{T}^{b}>140$"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2tt-root", type=Path, required=True)
    parser.add_argument("--t2bw-root", type=Path, required=True)
    parser.add_argument("--t2tb-root", type=Path, required=True)
    parser.add_argument("--t2bw-summary", type=Path, required=True)
    parser.add_argument("--t2tt-entries", type=Path, required=True)
    parser.add_argument("--t2bw-entries", type=Path, required=True)
    parser.add_argument("--t2tb-entries", type=Path, required=True)
    parser.add_argument("--t2tt-normalization", type=Path, required=True)
    parser.add_argument("--new-signal-normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--highdm-tail-merged",
        action="store_true",
        help=(
            "Read the 50-bin High-dM model obtained by merging the final "
            "two MET bins in each of the ten categories"
        ),
    )
    parser.add_argument(
        "--highdm-only",
        action="store_true",
        help="Render only the High-dM signal-category plots",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lowdm_categories(summary_path: Path) -> list[tuple[str, int, str]]:
    payload = json.loads(summary_path.read_text())
    result: list[tuple[str, int, str]] = []
    current = ""
    count = 0
    for index in range(34):
        label = payload["channels"][f"lSR_b{index:02d}"]["bin_label"]
        category = label.rsplit("_recoil_", 1)[0]
        if current and category != current:
            result.append((current, count, LOWDM_CATEGORY_LABELS[current]))
            count = 0
        current = category
        count += 1
    if current:
        result.append((current, count, LOWDM_CATEGORY_LABELS[current]))
    if sum(size for _, size, _ in result) != 34:
        raise RuntimeError(f"Low-dM category coverage is not 34 bins: {result}")
    return result


def read_signal_bins(
    root_path: Path,
    prefix: str,
    nbin: int,
    source_indices: list[int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    import uproot

    if source_indices is None:
        source_indices = list(range(nbin))
    if len(source_indices) != nbin:
        raise ValueError(
            f"source index count {len(source_indices)} does not match {nbin}"
        )
    output: dict[str, dict[str, np.ndarray]] = {}
    with uproot.open(root_path) as root_file:
        for key, _, _ in MASS_POINTS:
            values = np.zeros(nbin, dtype=float)
            variances = np.zeros(nbin, dtype=float)
            for output_index, source_index in enumerate(source_indices):
                hist = root_file[
                    f"{prefix}_b{source_index:02d}/sig_{key}"
                ]
                values[output_index] = float(hist.values(flow=False)[0])
                variance = hist.variances(flow=False)
                variances[output_index] = (
                    0.0 if variance is None else float(variance[0])
                )
            # Combine templates use 1e-9 as a technical floor. It is not a
            # physical signal yield and should not set the display range.
            values[(values <= 1.0e-8) & (variances <= 1.0e-16)] = 0.0
            output[key] = {
                "yield": values,
                "stat_unc": np.sqrt(np.maximum(variances, 0.0)),
            }
    return output


def load_event_counts(
    topology: str,
    entries_path: Path,
    normalization_path: Path,
) -> dict[str, dict[str, int | float]]:
    entries_payload = json.loads(entries_path.read_text())
    normalization_payload = json.loads(normalization_path.read_text())
    output: dict[str, dict[str, int | float]] = {}
    for mass_key, _, _ in MASS_POINTS:
        sample = f"{topology}_{mass_key}"
        normalization_key = mass_key if topology == "T2tt" else sample
        normalization = normalization_payload["signal_mass_points"][normalization_key]
        total = int(round(float(normalization["sumw_mass_point"])))
        point_counts = entries_payload["samples"][sample]
        output[mass_key] = {
            "total_generated": total,
            "highdm_selected": int(point_counts["highdm"]["selected_entries"]),
            "lowdm_selected": int(point_counts["lowdm"]["selected_entries"]),
        }
    return output


def draw_plot(
    topology: str,
    regime: str,
    categories: list[tuple[str, int, str]],
    signals: dict[str, dict[str, np.ndarray]],
    event_counts: dict[str, dict[str, int | float]],
    output_base: Path,
    show_suspected_bins: bool = True,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.patches import Patch, Rectangle

    hep.style.use("CMS")
    nbin = sum(size for _, size, _ in categories)
    centers = np.arange(1, nbin + 1, dtype=float)
    edges = np.arange(0.5, nbin + 1.5, dtype=float)
    boundaries = np.cumsum([0] + [size for _, size, _ in categories])

    width = 18.0 if regime == "High-dM" else 16.4
    fig, (ax, lax) = plt.subplots(
        2,
        1,
        figsize=(width, 8.6),
        gridspec_kw={"height_ratios": [4.2, 1.25], "hspace": 0.035},
        sharex=True,
    )

    positive = []
    upper_envelopes = []
    regime_count_key = "highdm_selected" if regime == "High-dM" else "lowdm_selected"
    for key, label, color in MASS_POINTS:
        values = signals[key]["yield"]
        errors = signals[key]["stat_unc"]
        selected = int(event_counts[key][regime_count_key])
        total = int(event_counts[key]["total_generated"])
        display_label = (
            label
            + "\n"
            + f"selected / generated: {selected:,} / {total:,}"
        )
        positive.extend(values[values > 0].tolist())
        upper_envelopes.extend((values + errors)[values + errors > 0].tolist())

        draw_floor = 1.0e-12
        lower = np.maximum(values - errors, draw_floor)
        upper = np.maximum(values + errors, draw_floor)
        visible = values > 0
        lower = np.where(visible, lower, np.nan)
        upper = np.where(visible, upper, np.nan)
        ax.fill_between(
            edges,
            np.r_[lower, lower[-1]],
            np.r_[upper, upper[-1]],
            step="post",
            color=color,
            alpha=0.22,
            linewidth=0.0,
            zorder=2,
        )
        ax.stairs(
            values,
            edges,
            color=color,
            linewidth=2.8,
            label=display_label,
            zorder=4,
        )

    suspected_bins = (
        SUSPECTED_BINS.get((topology, regime), ())
        if show_suspected_bins
        else ()
    )
    for bin_number in suspected_bins:
        ax.add_patch(
            Rectangle(
                (bin_number - 0.5, 0.0),
                1.0,
                1.0,
                transform=ax.get_xaxis_transform(),
                facecolor="none",
                edgecolor="#D40000",
                linewidth=2.6,
                zorder=8,
                clip_on=True,
            )
        )

    for axis in (ax, lax):
        for boundary in boundaries[1:-1]:
            axis.axvline(boundary + 0.5, color="black", linewidth=1.15)
        axis.set_xlim(0.5, nbin + 0.5)
        axis.tick_params(which="major", direction="in", top=True, right=True)
        axis.tick_params(which="minor", direction="in", top=True, right=True)
        axis.minorticks_on()

    if positive and upper_envelopes:
        ymin = max(1.0e-4, min(positive) / 3.5)
        ymax = max(1.0, max(upper_envelopes) * 7.0)
    else:
        ymin, ymax = 1.0e-3, 1.0
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Expected signal events / bin", fontsize=25)
    ax.tick_params(axis="y", labelsize=17, length=8)
    ax.grid(axis="y", which="major", color="0.88", linewidth=0.7, zorder=0)

    lax.set_ylim(0.0, 1.0)
    lax.set_yticks([])
    lax.tick_params(axis="x", labelsize=11 if nbin > 40 else 13, length=7)
    lax.set_xticks(centers)
    lax.set_xticklabels([str(index) for index in range(1, nbin + 1)])
    lax.set_xlabel("Search bin number", fontsize=25, loc="right")
    for start, end, (_, _, label) in zip(boundaries[:-1], boundaries[1:], categories):
        center = 0.5 * (start + end) + 0.5
        lax.text(
            center,
            0.5,
            label,
            transform=lax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=9.8 if regime == "High-dM" else 9.2,
            bbox={
                "boxstyle": "round,pad=0.26",
                "facecolor": "white",
                "edgecolor": "0.70",
                "alpha": 0.96,
            },
            zorder=10,
        )

    hep.cms.label(
        llabel="Work in progress",
        rlabel=rf"{LUMINOSITY_FB:.2f} fb$^{{-1}}$ (13.6 TeV)",
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.55", alpha=0.22, edgecolor="none"))
    labels.append("Signal MC stat. unc.")
    if suspected_bins:
        handles.append(Patch(facecolor="none", edgecolor="#D40000", linewidth=2.6))
        labels.append("Sparse bin under investigation")
    ax.legend(
        handles,
        labels,
        title=f"{topology} signal — {regime} SR",
        fontsize=13.0,
        title_fontsize=13.5,
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.90,
        loc="upper right",
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    points = {}
    for key, _, _ in MASS_POINTS:
        points[key] = {
            "yield_sum": float(np.sum(signals[key]["yield"])),
            "stat_unc_sum": float(np.sqrt(np.sum(signals[key]["stat_unc"] ** 2))),
            "nonzero_bins": int(np.count_nonzero(signals[key]["yield"] > 0)),
            "selected_entries": int(event_counts[key][regime_count_key]),
            "total_generated": int(event_counts[key]["total_generated"]),
            "selection_efficiency": float(
                event_counts[key][regime_count_key] / event_counts[key]["total_generated"]
            ),
        }
    return {
        "topology": topology,
        "regime": regime,
        "bins": nbin,
        "suspected_bins": list(suspected_bins),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "points": points,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        "T2tt": args.t2tt_root,
        "T2bW": args.t2bw_root,
        "T2tb": args.t2tb_root,
    }
    event_counts = {
        "T2tt": load_event_counts("T2tt", args.t2tt_entries, args.t2tt_normalization),
        "T2bW": load_event_counts("T2bW", args.t2bw_entries, args.new_signal_normalization),
        "T2tb": load_event_counts("T2tb", args.t2tb_entries, args.new_signal_normalization),
    }
    lowdm = (
        []
        if args.highdm_only
        else lowdm_categories(args.t2bw_summary)
    )
    highdm_source_indices = (
        [index for index in range(60) if index % 6 != 5]
        if args.highdm_tail_merged
        else list(range(60))
    )
    highdm_categories = (
        list(HIGHDM_TAIL_MERGED_CATEGORIES)
        if args.highdm_tail_merged
        else list(HIGHDM_CATEGORIES)
    )
    highdm_bin_count = len(highdm_source_indices)
    plots = []
    for topology, root_path in roots.items():
        high_signals = read_signal_bins(
            root_path,
            "hSR",
            highdm_bin_count,
            source_indices=highdm_source_indices,
        )
        stem = topology.lower()
        plots.append(
            draw_plot(
                topology,
                "High-dM",
                highdm_categories,
                high_signals,
                event_counts[topology],
                args.output_dir
                / (
                    f"{stem}_highdm_sr_mstop1250_1300_mlsp400_"
                    + (
                        "tailmerged_signal_statunc"
                        if args.highdm_tail_merged
                        else "signal_statunc"
                    )
                ),
                show_suspected_bins=not args.highdm_tail_merged,
            )
        )
        if args.highdm_only:
            continue
        low_signals = read_signal_bins(root_path, "lSR", 34)
        plots.append(
            draw_plot(
                topology,
                "Low-dM",
                lowdm,
                low_signals,
                event_counts[topology],
                args.output_dir / f"{stem}_lowdm_sr_mstop1250_1300_mlsp400_signal_statunc",
            )
        )

    manifest = {
        "status": "complete",
        "luminosity_fb": LUMINOSITY_FB,
        "mass_points": [key for key, _, _ in MASS_POINTS],
        "highdm_tail_merged": args.highdm_tail_merged,
        "highdm_bins": highdm_bin_count,
        "highdm_merge_pairs_1based": (
            [[index, index + 1] for index in range(5, 60, 6)]
            if args.highdm_tail_merged
            else []
        ),
        "uncertainty": "signal MC statistical uncertainty from TH1 sumw2",
        "suspected_bin_annotation": (
            {
                "enabled": False,
                "reason": (
                    "the displayed High-dM bins already contain the "
                    "requested tail-bin merges"
                ),
            }
            if args.highdm_tail_merged
            else {
                "enabled": True,
                "style": "unfilled red box",
                "meaning": (
                    "sparse High-dM signal bin selected for the "
                    "adjacent-bin merge diagnostic"
                ),
                "physics_inputs_modified": False,
            }
        ),
        "binning_modification": (
            "the final two MET bins of every High-dM category are summed"
            if args.highdm_tail_merged
            else "none"
        ),
        "event_count_definition": {
            "selected": "sum of nominal unweighted search-bin entries in the specified SR",
            "generated": "mass-point Runs.genEventSumw; unit generator weights make this the generated event count",
        },
        "inputs": {
            topology: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for topology, path in roots.items()
        },
        "plots": plots,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
