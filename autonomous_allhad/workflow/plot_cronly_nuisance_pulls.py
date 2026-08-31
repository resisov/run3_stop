#!/usr/bin/env python3
"""Plot standard-normal CR-only nuisance pulls in the analysis CMS style."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


BLUE = "#0000FF"
TEAL = "#1B9E77"
PURPLE = "#7C3AED"
REFERENCE = "#6B7280"


def split_scope(name: str) -> tuple[str, str]:
    for year in ("2024", "2025"):
        suffix = f"_{year}"
        if name.endswith(suffix):
            return name[: -len(suffix)], year
    if name.endswith("_correlated"):
        return name, "correlated"
    return name, "unscoped"


def configure_style() -> None:
    plt.style.use(hep.style.CMS)
    mpl.rcParams.update(
        {
            "axes.labelsize": 28,
            "axes.titlesize": 24,
            "font.size": 20,
            "legend.fontsize": 19,
            "savefig.facecolor": "white",
            "xtick.labelsize": 18,
            "ytick.labelsize": 19,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    rows = [
        (name, record)
        for name, record in payload["parameters"].items()
        if record.get("constraint_model") == "standard_normal"
    ]
    if not rows:
        raise SystemExit("no standard-normal nuisance parameters found")

    names = [name for name, _ in rows]
    values = np.asarray([record["pull"] for _, record in rows], dtype=float)
    errors = np.asarray([record["error"] for _, record in rows], dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(errors)):
        raise SystemExit("nonfinite nuisance pull or uncertainty")

    configure_style()
    x = np.arange(len(rows), dtype=float)
    fig, axis = plt.subplots(figsize=(25.0, 7.0))

    axis.axhspan(-1.0, 1.0, color="#D1D5DB", alpha=0.72, zorder=0)
    for boundary in np.arange(len(rows) - 1, dtype=float) + 0.5:
        axis.axvline(
            boundary,
            color="#6B7280",
            linestyle=(0, (2.0, 4.0)),
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )
    axis.axhline(0.0, color="#111827", linewidth=1.25, zorder=1)
    for reference in (-1.0, 1.0):
        axis.axhline(
            reference,
            color=REFERENCE,
            linestyle=(0, (5, 4)),
            linewidth=1.15,
            zorder=1,
        )
    axis.axhline(-2.0, color="#D1D5DB", linewidth=0.8, zorder=0)
    axis.axhline(2.0, color="#D1D5DB", linewidth=0.8, zorder=0)

    for position, name, value, error in zip(x, names, values, errors):
        axis.errorbar(
            position,
            value,
            yerr=error,
            fmt="+",
            markersize=13.0,
            color=BLUE,
            markeredgewidth=2.6,
            elinewidth=2.35,
            capsize=0.0,
            zorder=4,
        )

    axis.set_xlim(-0.7, len(rows) - 0.3)
    axis.set_ylim(-3.0, 3.0)
    axis.set_ylabel(r"B-only fit nuisance pull  $\hat{\theta}$")
    axis.set_xticks(x)
    axis.set_xticklabels(
        names,
        rotation=90,
        ha="center",
        va="top",
        fontsize=18.0,
        fontweight="medium",
    )
    axis.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    axis.tick_params(
        axis="x",
        which="major",
        direction="in",
        top=False,
        bottom=True,
        length=8,
        width=1.25,
        pad=9,
    )
    axis.tick_params(
        axis="y",
        which="both",
        direction="in",
        right=True,
        length=7,
        width=1.15,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(1.35)
        spine.set_color("#111827")
    axis.minorticks_on()
    axis.tick_params(axis="x", which="minor", bottom=False, top=False)
    axis.grid(axis="y", which="minor", color="#E5E7EB", linewidth=0.55)
    axis.grid(axis="x", visible=False)

    uncertainty = mpl.lines.Line2D(
        [], [], color=BLUE, marker="+", markersize=18,
        markeredgewidth=2.6, linestyle="none", label="B-only fit"
    )
    prefit = mpl.patches.Patch(
        facecolor="#D1D5DB", edgecolor="none", alpha=0.72,
        label="Prefit"
    )
    axis.legend(
        handles=[prefit, uncertainty],
        loc="upper right",
        ncol=2,
        fontsize=28,
        frameon=False,
        handlelength=1.5,
        handleheight=1.15,
        handletextpad=0.55,
        columnspacing=1.35,
    )
    hep.cms.label(
        llabel="Work in progress",
        rlabel="2024 + 2025 (13.6 TeV)",
        fontsize=34,
        ax=axis,
    )

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(
        json.dumps(
            {
                "status": "complete",
                "nuisances": len(rows),
                "png": str(args.output_stem.with_suffix(".png")),
                "pdf": str(args.output_stem.with_suffix(".pdf")),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
