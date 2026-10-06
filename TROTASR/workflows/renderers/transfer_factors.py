"""Verbatim legacy rendering fragments; no data or analysis path dependencies."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from .measurement_style import CMS_LABEL_FONT_SIZE, FIGURE_SIZE, _apply_style as apply_photon_hlt_style
hep.style.use("CMS")


CMS_LABEL = {
    "llabel": "Work in progress",
    "rlabel": "2024 (13.6 TeV)",
}


PATHS = (
    {
        "key": "top_llcr",
        "display": "Top",
        "numerator_process": "Top",
        "denominator_region": "LLCR",
        "denominator_process": "Top",
        "ratio_label": r"Top: SR / LLCR",
    },
    {
        "key": "w_llcr",
        "display": r"$W\!\to\!\ell\nu$",
        "numerator_process": "WtoLNu",
        "denominator_region": "LLCR",
        "denominator_process": "WtoLNu",
        "ratio_label": r"$W\!\to\!\ell\nu$: SR / LLCR",
    },
    {
        "key": "qcd_qcdcr",
        "display": "QCD",
        "numerator_process": "QCD",
        "denominator_region": "QCDCR",
        "denominator_process": "QCD",
        "ratio_label": "QCD: SR / QCDCR",
    },
)


HIGH_STYLES = {
    "inclusive": ("o", "#e31a1c", "Inclusive"),
    "Nb1": ("o", "#e31a1c", r"$N_b=1$"),
    "Nb2plus": ("s", "#1f78b4", r"$N_b\geq2$"),
    "Nb2": ("s", "#1f78b4", r"$N_b=2$"),
    "Nb3plus": ("^", "#168B38", r"$N_b\geq3$"),
}


GNN_STYLES = {
    f"{nb}_{isr}": (marker, color, nb_label + ", " + isr_label)
    for nb, color, nb_label in (("Nb1", "#e31a1c", r"$N_b=1$"),
                                ("Nb2plus", "#1f78b4", r"$N_b\geq2$"))
    for isr, marker, isr_label in (("NISR0", "o", r"$N_{\mathrm{ISR}}=0$"),
                                   ("NISR1", "s", r"$N_{\mathrm{ISR}}=1$"),
                                   ("NISR2plus", "^", r"$N_{\mathrm{ISR}}\geq2$"))
}


GNN_PATHS = PATHS + ({
    "key": "zinv_gcr",
    "ratio_label": r"Raw $Z\!\to\!\nu\nu$: SR / $\gamma$ CR",
},)


def finite_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(
        [np.nan if value is None else value for value in record["transfer_factor"]],
        dtype=float,
    )
    errors = np.asarray(
        [np.nan if value is None else value for value in record["mcstat"]],
        dtype=float,
    )
    return values, errors, np.isfinite(values) & np.isfinite(errors)


def set_tf_ylim(axis: plt.Axes, records: list[dict[str, Any]]) -> None:
    upper: list[float] = []
    for record in records:
        values, errors, valid = finite_arrays(record)
        upper.extend((values[valid] + errors[valid]).tolist())
    ymax = max(upper, default=1.0)
    axis.set_ylim(0.0, max(1.2 * ymax, 0.05))


def save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    paths = []
    for suffix, kwargs in (
        (".png", {"dpi": 180}),
        (".pdf", {}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_highdm(
    output_dir: Path,
    path: dict[str, Any],
    edges: np.ndarray | dict[str, np.ndarray],
    records: dict[str, Any],
    regime: str = "highdm",
    *,
    xlabel: str = r"$U_T$ (GeV)",
    annotation: str | None = None,
    output_suffix: str | None = None,
    ylabel: str = r"Transfer factor $N_{\mathrm{SR}}/N_{\mathrm{CR}}$",
    styles: dict[str, tuple[str, str, str]] | None = None,
) -> list[str]:
    apply_photon_hlt_style()
    fig, axis = plt.subplots(figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.88)
    used: list[dict[str, Any]] = []
    styles = HIGH_STYLES if styles is None else styles
    group_order = [group for group in styles if group in records]
    if set(group_order) != set(records):
        raise ValueError("missing style for a plotted TF category")
    bounds = []
    for group in group_order:
        record = records[group]
        current_edges = np.asarray(edges[group] if isinstance(edges, dict) else edges)
        centers = 0.5 * (current_edges[:-1] + current_edges[1:])
        widths = 0.5 * np.diff(current_edges)
        bounds.extend((float(current_edges[0]), float(current_edges[-1])))
        values, errors, valid = finite_arrays(record)
        marker, color, label = styles[group]
        axis.errorbar(
            centers[valid],
            values[valid],
            xerr=widths[valid],
            yerr=errors[valid],
            fmt=marker,
            ls="none",
            color=color,
            lw=1.1,
            ms=5.5,
            capsize=2,
            label=label,
            zorder=3,
        )
        used.append(record)
    set_tf_ylim(axis, used)
    if len(group_order) > 3:
        axis.set_ylim(0.0, axis.get_ylim()[1] * 1.25)
    axis.set_xlim(min(bounds), max(bounds))
    axis.set_xmargin(0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle=":", color="0.78", linewidth=0.9)
    axis.text(
        0.04,
        0.70 if len(group_order) > 3 else (0.73 if path["key"] == "qcd_qcdcr" and regime == "highdm" else 0.07),
        annotation if annotation is not None else ("High-" if regime == "highdm" else "Low-")
        + r"$\Delta m$"
        + "\n"
        + path["ratio_label"],
        transform=axis.transAxes,
        fontsize=14,
    )
    axis.legend(
        frameon=False,
        ncol=2 if len(group_order) > 3 else 1,
        loc="upper right" if len(group_order) > 3 else "best",
    )
    with plt.rc_context({"font.size": CMS_LABEL_FONT_SIZE}):
        hep.cms.label(**CMS_LABEL, loc=0, ax=axis)
    return save_figure(
        fig,
        output_dir / f"transfer_factor_{path['key']}_{output_suffix or regime}",
    )


def render_factors(output_dir: Path, factors: dict[str, Any], regime: str) -> tuple[list[str], list[str]]:
    """Use the existing renderer once per background, overlaying categories."""
    high_plots, gnn_plots = [], []
    if regime in {"all", "highdm"}:
        for path in PATHS:
            high_plots.extend(plot_highdm(
                output_dir, path, np.asarray(factors["highdm"]["edges"]),
                factors["highdm"]["records"][path["key"]],
            ))
    if regime in {"all", "lowdm"}:
        (output_dir / "gnn").mkdir(parents=True, exist_ok=True)
        for path in GNN_PATHS:
            records = factors["lowdm"]["records"][path["key"]]
            gnn_plots.extend(plot_highdm(
                output_dir / "gnn", path,
                {category: np.asarray(record["score_edges"]) for category, record in records.items()},
                records, regime="lowdm", xlabel="GNN output",
                output_suffix="lowdm_gnn", styles=GNN_STYLES,
                ylabel=r"Transfer factor $N_{\mathrm{SR,bin}}/N_{\mathrm{CR,parent}}$",
            ))
    return high_plots, gnn_plots
