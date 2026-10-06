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


CMS_LABEL = {"llabel": "Work in progress", "rlabel": "2024 (13.6 TeV)"}


def plot(regime: str, rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    edges = np.asarray([rows[0]["low"]] + [row["high"] for row in rows])
    x = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * np.diff(edges)
    z = np.asarray([row["z_data_over_mc"] for row in rows])
    zerr = np.asarray([row["z_stat"] for row in rows])
    gamma = np.asarray([row["photon_data_over_mc"] for row in rows])
    gamma_err = np.asarray([row["photon_stat"] for row in rows])
    double = np.asarray([row["double_ratio"] for row in rows])
    double_err = np.asarray([row["double_ratio_stat"] for row in rows])
    systematic = np.asarray([row["systematic"] for row in rows])

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.25], "hspace": 0.05},
    )
    ax.errorbar(x, z, xerr=xerr, yerr=zerr, fmt="o", ms=9, color="#ff0000",
                capsize=3, label=r"$Z(\ell\ell)$ CR")
    ax.errorbar(x, gamma, xerr=xerr, yerr=gamma_err, fmt="s", ms=8,
                color="#0000ff", capsize=3, label=r"$\gamma$ CR")
    ax.axhline(1.0, color="0.35", lw=1.5)
    ax.set_ylabel("Normalized Data/MC")
    ax.legend(fontsize=18, loc="best")
    ax.text(0.04, 0.08, r"High-$\Delta m$" if regime == "highdm" else r"Low-$\Delta m$",
            transform=ax.transAxes, fontsize=20)
    hep.cms.label(ax=ax, loc=0, **CMS_LABEL)

    band_low = 1.0 - systematic
    band_high = 1.0 + systematic
    rax.stairs(band_low, edges, baseline=band_high, fill=True, color="#66bb66",
               alpha=0.4, label="Assigned systematic")
    rax.errorbar(x, double, xerr=xerr, yerr=double_err, fmt="o", ms=9,
                 color="black", capsize=3, label=r"$Z/\gamma$ double ratio")
    rax.axhline(1.0, color="0.25", lw=1.5)
    rax.set_ylabel(r"$Z/\gamma$")
    rax.set_xlabel(r"$U_T$ (GeV)")
    rax.legend(fontsize=15, loc="best")
    for axis in (ax, rax):
        axis.set_xlim(edges[0], edges[-1])
        axis.margins(x=0)
        axis.grid(axis="y", alpha=0.22, linestyle=":")
    finite = np.concatenate((z[np.isfinite(z)], gamma[np.isfinite(gamma)]))
    if len(finite):
        ax.set_ylim(0.0, max(1.6, float(np.max(finite)) * 1.35))
    finite_double = double[np.isfinite(double)]
    finite_span = (double_err + systematic)[np.isfinite(double)]
    if len(finite_double):
        rax.set_ylim(
            max(0.0, float(np.min(finite_double - finite_span)) - 0.1),
            max(1.5, float(np.max(finite_double + finite_span)) + 0.1),
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"zgamma_double_ratio_{regime}.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths
