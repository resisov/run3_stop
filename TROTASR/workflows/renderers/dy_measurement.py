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


CHANNELS = ("DY2E", "DY2M")


GROUPS = ("Nb1", "Nb2plus")


def save_figure(fig: plt.Figure, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def plot_rt(
    rz: dict[str, Any], selection: str, output_dir: Path
) -> list[str]:
    fig, ax = plt.subplots(figsize=(10.2, 10.2))
    x = np.arange(len(GROUPS), dtype=float)
    colors = {"DY2E": "#D62728", "DY2M": "#1F77B4"}
    markers = {"DY2E": "o", "DY2M": "^"}
    labels = {"DY2E": r"$ee$", "DY2M": r"$\mu\mu$"}
    for channel in CHANNELS:
        values: list[float] = []
        errors: list[float] = []
        positions: list[float] = []
        for index, group in enumerate(GROUPS):
            record = ((rz.get("channels") or {}).get(channel) or {}).get(
                group, {}
            )
            if record.get("status") != "complete":
                continue
            positions.append(float(x[index]))
            values.append(float(record["RT"]))
            errors.append(float(record["RT_stat"]))
        ax.errorbar(
            positions,
            values,
            xerr=np.full(len(positions), 0.5, dtype=float),
            yerr=errors,
            fmt=markers[channel],
            color=colors[channel],
            lw=2.0,
            capsize=4,
            label=labels[channel],
        )
    ax.axhline(1.0, color="0.55", lw=1.5, ls=":")
    ax.set_xticks(x, [r"$N_b=1$", r"$N_b\geq2$"], fontsize=26)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_xmargin(0)
    ax.set_ylabel(r"$R_T$", fontsize=30)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(frameon=False, fontsize=24)
    ax.grid(axis="y", alpha=0.18)
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"rt_{selection}")


def plot_rz_nb(
    rz: dict[str, Any], selection: str, output_dir: Path
) -> list[str]:
    fig, ax = plt.subplots(figsize=(10.2, 10.2))
    x = np.arange(len(GROUPS), dtype=float)
    colors = {"DY2E": "#D62728", "DY2M": "#1F77B4"}
    markers = {"DY2E": "o", "DY2M": "^"}
    labels = {"DY2E": r"$ee$", "DY2M": r"$\mu\mu$"}
    for channel in CHANNELS:
        records = [rz["channels"][channel][group] for group in GROUPS]
        ax.errorbar(
            x,
            [record["RZ"] for record in records],
            xerr=np.full(len(records), 0.5, dtype=float),
            yerr=[record["RZ_stat"] for record in records],
            fmt=markers[channel],
            color=colors[channel],
            lw=2.0,
            capsize=4,
            label=labels[channel],
        )
    combined = [rz["combined"][group] for group in GROUPS]
    ax.errorbar(
        x,
        [record["RZ"] for record in combined],
        xerr=np.full(len(combined), 0.5, dtype=float),
        yerr=[record["RZ_stat"] for record in combined],
        fmt="D",
        color="#111111",
        markerfacecolor="white",
        markeredgewidth=2.0,
        markersize=10,
        lw=2.4,
        capsize=5,
        zorder=5,
        label="Combined",
    )
    ax.axhline(1.0, color="0.55", lw=1.5, ls=":")
    ax.set_xticks(x, [r"$N_b=1$", r"$N_b\geq2$"], fontsize=26)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_xmargin(0)
    ax.set_ylabel(r"$R_Z$", fontsize=30)
    ax.tick_params(axis="y", labelsize=24)
    ax.legend(frameon=False, fontsize=24)
    ax.grid(axis="y", alpha=0.18)
    hep.cms.label(**CMS_LABEL, ax=ax)
    return save_figure(fig, output_dir / f"rz_{selection}")


def plot_mll(
    source: dict[str, Any],
    rz: dict[str, Any],
    selection: str,
    output_dir: Path,
    *,
    corrected: bool,
) -> list[str]:
    """Plot the matrix input, optionally after the fitted RZ/RT scaling."""

    paths: list[str] = []
    for channel in CHANNELS:
        for group in GROUPS:
            node = ((source.get(channel) or {}).get(group) or {})
            if not node:
                continue
            first = next(iter(node.values()))
            edges = np.asarray(first["edges"], dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            data = np.asarray(
                (node.get("data") or {}).get("sumw", []), dtype=float
            )
            data2 = np.asarray(
                (node.get("data") or {}).get("sumw2", []), dtype=float
            )
            zll = np.asarray(
                (node.get("zll") or {}).get("sumw", []), dtype=float
            )
            zll2 = np.asarray(
                (node.get("zll") or {}).get("sumw2", []), dtype=float
            )
            other = np.asarray(
                (node.get("other") or {}).get("sumw", []), dtype=float
            )
            other2 = np.asarray(
                (node.get("other") or {}).get("sumw2", []), dtype=float
            )
            if not len(data):
                continue

            if corrected:
                factors = ((rz.get("channels") or {}).get(channel) or {}).get(
                    group, {}
                )
                if factors.get("status") != "complete":
                    continue
                z_scale = float(factors["RZ"])
                other_scale = float(factors["RT"])
                zll = zll * z_scale
                zll2 = zll2 * z_scale * z_scale
                other = other * other_scale
                other2 = other2 * other_scale * other_scale

            total = zll + other
            total_error = np.sqrt(np.maximum(zll2 + other2, 0.0))
            data_error = np.sqrt(np.maximum(data2, 0.0))
            valid_ratio = total > 0.0
            ratio = np.full_like(data, np.nan)
            ratio_error = np.full_like(data, np.nan)
            ratio[valid_ratio] = data[valid_ratio] / total[valid_ratio]
            ratio_error[valid_ratio] = (
                data_error[valid_ratio] / total[valid_ratio]
            )
            relative_mc_error = np.zeros_like(total)
            relative_mc_error[valid_ratio] = (
                total_error[valid_ratio] / total[valid_ratio]
            )

            fig, (ax, rax) = plt.subplots(
                2,
                1,
                figsize=(10.2, 10.2),
                sharex=True,
                gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.04},
            )
            ax.stairs(
                zll,
                edges,
                fill=True,
                baseline=0.0,
                color="#35B6B4",
                edgecolor="black",
                linewidth=0.7,
                label="DY",
                zorder=1,
            )
            ax.stairs(
                total,
                edges,
                fill=True,
                baseline=zll,
                color="#6A625F",
                edgecolor="black",
                linewidth=0.7,
                label="Others",
                zorder=2,
            )
            ax.stairs(
                total + total_error,
                edges,
                baseline=np.maximum(total - total_error, 0.0),
                fill=True,
                facecolor="none",
                edgecolor="0.35",
                hatch="////",
                linewidth=0.0,
                label="Stat. unc.",
                zorder=3,
            )
            ax.errorbar(
                centers,
                data,
                yerr=data_error,
                fmt="o",
                color="black",
                ms=6,
                lw=2.0,
                capsize=2,
                label="Data",
                zorder=4,
            )
            ax.axvspan(81.0, 101.0, color="#FFD166", alpha=0.18)
            rax.axvspan(81.0, 101.0, color="#FFD166", alpha=0.18)
            rax.stairs(
                1.0 + relative_mc_error,
                edges,
                baseline=1.0 - relative_mc_error,
                fill=True,
                facecolor="0.75",
                edgecolor="0.55",
                alpha=0.55,
                linewidth=0.0,
            )
            rax.errorbar(
                centers[valid_ratio],
                ratio[valid_ratio],
                yerr=ratio_error[valid_ratio],
                fmt="o",
                color="black",
                ms=6,
                lw=2.0,
                capsize=2,
            )
            rax.axhline(1.0, color="black", lw=1.5)
            ax.set_ylabel("Events / bin", fontsize=30)
            rax.set_ylabel("Data/MC", fontsize=28)
            rax.set_xlabel(r"$m_{\ell\ell}$ (GeV)", fontsize=30, loc="right")
            ax.set_yscale("log")
            peak = max(float(np.max(data + data_error)), float(np.max(total + total_error)))
            ax.set_ylim(1.0e-1, max(1.0e3, 3.0 * peak))
            rax.set_ylim(0.0, 2.0)
            for axis in (ax, rax):
                axis.set_xlim(float(edges[0]), float(edges[-1]))
                axis.set_xmargin(0)
                axis.tick_params(
                    which="major",
                    direction="in",
                    top=True,
                    right=True,
                    labelsize=24,
                    length=9,
                )
                axis.tick_params(
                    which="minor",
                    direction="in",
                    top=True,
                    right=True,
                    length=5,
                )
                axis.minorticks_on()
            handles, labels = ax.get_legend_handles_labels()
            order = ["Stat. unc.", "Others", "DY", "Data"]
            ordered = [
                (handles[labels.index(label)], label)
                for label in order
                if label in labels
            ]
            ax.legend(
                [item[0] for item in ordered],
                [item[1] for item in ordered],
                frameon=False,
                fontsize=22,
                ncol=2,
                loc="upper right",
            )
            hep.cms.label(**CMS_LABEL, ax=ax)
            suffix = "_post" if corrected else ""
            paths.extend(
                save_figure(
                    fig,
                    output_dir
                    / f"mll_{selection}_{channel.lower()}_{group.lower()}{suffix}",
                )
            )
    return paths
