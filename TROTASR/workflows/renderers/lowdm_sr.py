"""Original low-dM 30-bin drawing body, with an internal array boundary.
Source fragments, constants, signal choices and styles remain unchanged.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
from TROTASR.utils.plot_adapter import renderer as _renderer
validated_plot_uncertainty = _renderer.validated_plot_uncertainty
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


def draw(args, processes, variances, signals, raw_edges, band):
    background = np.sum([processes[key] for key in PROCESS_ORDER], axis=0)
    background_variance = np.sum(
        [variances[key] for key in PROCESS_ORDER], axis=0
    )
    uncertainty = validated_plot_uncertainty(background, background_variance, band)
    uncertainty_label = "MC stat+syst unc." if band is not None else "MC stat. unc."
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
        label=uncertainty_label,
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
        uncertainty_label,
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

