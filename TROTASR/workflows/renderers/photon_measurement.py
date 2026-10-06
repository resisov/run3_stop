"""Verbatim legacy rendering fragments; no external analysis I/O."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
hep.style.use("CMS")


CMS_LABEL = {
    "llabel": "Work in progress",
    "rlabel": "2024 (13.6 TeV)",
}


LOW_GROUPS = ("Nb1", "Nb2plus")


def low_ut_geometry(
    isr_group: str,
    nbin: int,
    overflow_cap: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isr_group == "PISR300to500":
        lower = 300.0
    elif isr_group == "PISR500plus":
        lower = 450.0
    else:
        raise ValueError(f"unknown Low-dM ISR group: {isr_group}")
    edges = lower + 100.0 * np.arange(nbin + 1, dtype=float)
    if overflow_cap is not None:
        if overflow_cap <= edges[-2]:
            raise ValueError(
                f"overflow cap {overflow_cap} does not exceed the last "
                f"Low-dM bin edge {edges[-2]}"
            )
        edges[-1] = overflow_cap
    return 0.5 * (edges[:-1] + edges[1:]), 0.5 * np.diff(edges)


def save_square_figure(fig: plt.Figure, base: Path) -> list[str]:
    """Save a true square canvas without tight-bbox label clipping."""
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.90)
    paths = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_q(factors: dict[str, Any], output_dir: Path) -> list[str]:
    categories = [
        ("highdm", "Nb1", r"$N_b=1$"),
        ("highdm", "Nb2plus", r"$N_b\geq2$"),
        ("lowdm_Q_groups", "Nb1", r"$N_b=1$"),
        ("lowdm_Q_groups", "Nb2plus", r"$N_b\geq2$"),
    ]
    values, errors, labels = [], [], []
    for section, group, label in categories:
        record = (
            factors[section][group]["Q"]
            if section == "highdm"
            else factors[section][group]
        )
        labels.append(label)
        if record.get("status") == "complete":
            values.append(float(record["value"]))
            errors.append(float(record["stat"]))
        else:
            values.append(float("nan"))
            errors.append(float("nan"))
    x = np.arange(len(labels), dtype=float)
    colors = ["#FF0000"] * 2 + ["#0000FF"] * 2
    markers = ["o"] * 2 + ["s"] * 2
    fig, ax = plt.subplots(figsize=(10.2, 10.2))
    for index in range(len(labels)):
        ax.errorbar(
            x[index],
            values[index],
            xerr=0.5,
            yerr=errors[index],
            fmt=markers[index],
            color=colors[index],
            lw=2.6,
            ms=11,
            mew=1.6,
            capsize=4,
            label=(
                r"High-$\Delta m$"
                if index == 0
                else r"Low-$\Delta m$"
                if index == 2
                else None
            ),
        )
    ax.axhline(1.0, color="0.45", lw=1.5, ls=":")
    ax.axvline(1.5, color="0.75", lw=1.2)
    ax.set_xticks(x, labels, fontsize=24)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xmargin(0)
    ax.set_ylabel(r"$Q=(N_{\mathrm{data}}-N_{\mathrm{other}})"
                  r"/N_{\gamma,\mathrm{MC}}$", fontsize=28)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(
        frameon=False,
        fontsize=27,
        markerscale=1.4,
        handlelength=2.0,
        labelspacing=0.7,
    )
    ax.grid(axis="y", alpha=0.16)
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_square_figure(fig, output_dir / "photon_q_normalization")


def plot_sgamma(
    factors: dict[str, Any],
    regime: str,
    output_dir: Path,
    recoil_edges: list[float] | np.ndarray | None = None,
) -> list[str]:
    paths: list[str] = []
    nb_recoil_mode = regime == "highdm" or set(factors) == set(LOW_GROUPS)
    if nb_recoil_mode:
        default_edges = (
            [250, 300, 350, 400, 500, 800, 1500]
            if regime == "highdm"
            else [250, 300, 350, 400, 500, 650, 800, 1000, 1500]
        )
        edges = np.asarray(
            default_edges if recoil_edges is None else recoil_edges,
            dtype=float,
        )
        expected_bins = len(next(iter(factors.values()))["bins"])
        if len(edges) != expected_bins + 1:
            raise ValueError(
                f"{regime} Sgamma has {expected_bins} bins but "
                f"{len(edges) - 1} plotting bins"
            )
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = 0.5 * (edges[1:] - edges[:-1])
        fig, ax = plt.subplots(figsize=(10.2, 10.2))
        styles = {
            "Nb1": ("o", r"$N_b=1$"),
            "Nb2plus": ("s", r"$N_b\geq2$"),
        }
        for group in styles:
            records = factors[group]["bins"]
            values = np.asarray(
                [
                    item["Sgamma"]["value"]
                    if item["Sgamma"]["status"] == "complete"
                    else np.nan
                    for item in records
                ]
            )
            errors = np.asarray(
                [
                    item["Sgamma"]["stat"]
                    if item["Sgamma"]["status"] == "complete"
                    else np.nan
                    for item in records
                ]
            )
            marker, label = styles[group]
            ax.errorbar(
                centers,
                values,
                xerr=widths,
                yerr=errors,
                fmt=marker,
                ls="none",
                lw=2.6,
                ms=9,
                mew=1.5,
                capsize=3,
                label=label,
            )
        ax.axhline(1.0, color="0.45", lw=1.5, ls=":")
        ax.set_xlabel(r"$U_T$ (GeV)", fontsize=30)
        ax.set_xlim(float(edges[0]), float(edges[-1]))
        ax.set_xmargin(0)
        ax.set_ylabel(r"$S_{\gamma,i}$", fontsize=30)
        ax.tick_params(labelsize=24)
        ax.text(
            0.04,
            0.08,
            (r"High-$\Delta m$" if regime == "highdm" else r"Low-$\Delta m$"),
            transform=ax.transAxes,
            fontsize=22,
        )
        ax.legend(
            frameon=False,
            fontsize=22,
            markerscale=1.25,
            handlelength=1.8,
            labelspacing=0.7,
        )
        ax.grid(alpha=0.16)
        hep.cms.label(**CMS_LABEL, ax=ax)
        paths.extend(save_square_figure(fig, output_dir / f"sgamma_{regime}"))
    else:
        # Keep every adopted Low-dM search-bin family separate.  In
        # particular, do not sum the pTb/Nj categories that happen to share
        # the same U_T boundaries: that would turn the 34-bin measurement
        # into a 14-point diagnostic rather than plotting the measurement.
        panel_specs = (
            (
                r"$N_b=1$" "\n" r"$300\leq p_T^{ISR}<500$",
                "PISR300to500",
                (
                    ("Nb1_PISR300to500_PTb20to40", r"$20<p_T^b<40$", "o", "#D62728"),
                    ("Nb1_PISR300to500_PTb40to70", r"$40<p_T^b<70$", "s", "#FF8C00"),
                ),
            ),
            (
                r"$N_b=1$" "\n" r"$p_T^{ISR}\geq500$",
                "PISR500plus",
                (
                    ("Nb1_PISR500plus_PTb20to40", r"$20<p_T^b<40$", "o", "#D62728"),
                    ("Nb1_PISR500plus_PTb40to70", r"$40<p_T^b<70$", "s", "#FF8C00"),
                ),
            ),
            (
                r"$N_b\geq2$" "\n" r"$300\leq p_T^{ISR}<500$",
                "PISR300to500",
                (
                    ("Nb2plus_PISR300to500_PTb40to80_Nj2plus", r"$40<p_T^b<80,\ N_j\geq2$", "o", "#1F77B4"),
                    ("Nb2plus_PISR300to500_PTb80to140_Nj2plus", r"$80<p_T^b<140,\ N_j\geq2$", "s", "#9467BD"),
                ),
            ),
            (
                r"$N_b\geq2$" "\n" r"$p_T^{ISR}\geq500$",
                "PISR500plus",
                (
                    ("Nb2plus_PISR500plus_PTb40to80_Nj2plus", r"$40<p_T^b<80,\ N_j\geq2$", "o", "#1F77B4"),
                    ("Nb2plus_PISR500plus_PTb80to140_Nj2plus", r"$80<p_T^b<140,\ N_j\geq2$", "s", "#9467BD"),
                ),
            ),
            (
                r"$N_b\geq2$" "\n" r"$300\leq p_T^{ISR}<500$",
                "PISR300to500",
                (
                    ("Nb2plus_PISR300to500_PTb140plus_Nj7plus", r"$p_T^b>140,\ N_j\geq7$", "^", "#2CA02C"),
                ),
            ),
            (
                r"$N_b\geq2$" "\n" r"$p_T^{ISR}\geq500$",
                "PISR500plus",
                (
                    ("Nb2plus_PISR500plus_PTb140plus_Nj7plus", r"$p_T^b>140,\ N_j\geq7$", "^", "#2CA02C"),
                ),
            ),
        )
        fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.0))
        for ax, (annotation, isr_group, series) in zip(axes.flat, panel_specs):
            lower_extent: list[float] = []
            upper_extent: list[float] = []
            for family, label, marker, color in series:
                records = factors[family]["bins"]
                centers, widths = low_ut_geometry(
                    isr_group, len(records), overflow_cap=1500.0
                )
                values = np.asarray(
                    [
                        item["Sgamma"]["value"]
                        if item["Sgamma"]["status"] == "complete"
                        else np.nan
                        for item in records
                    ],
                    dtype=float,
                )
                errors = np.asarray(
                    [
                        item["Sgamma"]["stat"]
                        if item["Sgamma"]["status"] == "complete"
                        else np.nan
                        for item in records
                    ],
                    dtype=float,
                )
                valid = np.isfinite(values) & np.isfinite(errors)
                lower_extent.extend((values[valid] - errors[valid]).tolist())
                upper_extent.extend((values[valid] + errors[valid]).tolist())
                ax.errorbar(
                    centers[valid],
                    values[valid],
                    xerr=widths[valid],
                    yerr=errors[valid],
                    fmt=marker,
                    ls="none",
                    lw=2.0,
                    ms=7.5,
                    mew=1.3,
                    capsize=2.5,
                    color=color,
                    label=label,
                )
            ymin = max(0.0, min(0.70, min(lower_extent, default=0.70) - 0.05))
            ymax_data = max(upper_extent, default=1.25)
            ymax = max(1.25, ymax_data + 0.06 * max(ymax_data - ymin, 0.2))
            ax.set_ylim(ymin, ymax)
            ax.axhline(1.0, color="0.45", lw=1.3, ls=":")
            ax.set_xlim(300.0 if isr_group == "PISR300to500" else 450.0, 1500.0)
            ax.set_xmargin(0)
            ax.tick_params(labelsize=14)
            ax.text(
                0.04,
                0.06,
                annotation,
                transform=ax.transAxes,
                fontsize=14,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
            )
            ax.grid(alpha=0.16)
            ax.legend(frameon=False, fontsize=11.5, loc="upper right")
        fig.supxlabel(r"$U_T$ (GeV)", fontsize=25, x=0.57, y=0.045)
        fig.supylabel(r"$S_{\gamma,i}$", fontsize=25, x=0.045, y=0.52)
        label_ax = fig.add_subplot(111, frameon=False)
        label_ax.set_xticks([])
        label_ax.set_yticks([])
        label_ax.patch.set_visible(False)
        label_ax.set_zorder(-1)
        hep.cms.label(**CMS_LABEL, ax=label_ax)
        paths.extend(
            save_square_figure(fig, output_dir / "sgamma_lowdm_nb_isr_shared")
        )
    return paths
