"""Draw before/after CR distributions with the canonical renderer, without a fit."""

from pathlib import Path

import numpy as np

from TROTASR.utils.paths import luminosity_fb
from . import plot_control_search_bins_style as canonical
from .background_process_groups import BACKGROUND_DISPLAY_LABELS, background_process_for_sample


REGIONS = ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M")
STAGES = ("Before cut", r"After $m_{T,b}\geq175$ GeV")
CATEGORIES = (("Nb1", r"$N_b=1$"), ("Nb2plus", r"$N_b\geq2$"))


def _cut_block(region, stage, label, edges, groups, observed, uncertainty, stat):
    xlabels = [f"{a:g}–{b:g}" for a, b in zip(edges[:-2], edges[1:-1])]
    xlabels.append(r"$\geq$" + f"{edges[-2]:g}")
    return dict(nbin=len(xlabels), xlabels=xlabels, label=region + "\n" + label,
        label_box=True, label_fontsize=17, label_box_pad=.30, label_va="bottom",
        category_labels_on_main=True, category_label_y=.04,
        main_annotation=STAGES[stage], main_annotation_x=.40, main_annotation_y=1.055,
        main_annotation_fontsize=24, main_annotation_fontweight="bold",
        main_panel_ymax_factor=4., groups=groups, background=sum(groups.values()),
        background_unc=uncertainty, background_stat_unc=stat, data=observed,
        data_unc=canonical.poisson_unc(observed), signals={}, signal_specs=[],
        physics_scope=region, unit_area=False, blind_data=False)


def _blocks(row, edges, region, stage):
    n = len(edges) - 1
    samples = {key: np.asarray(value, dtype=float) for key, value in row["samples"].items()}
    if any(a.shape != (2, 2, n, 3) or not np.isfinite(a).all() or np.any(a[..., 1:] < 0)
           for a in samples.values()):
        raise ValueError(f"Invalid saved CR histogram: {region}")
    data = samples.pop("data")[..., 2]
    if np.any(data != np.floor(data)) or np.any(data[:, 1] > data[:, 0]) or not samples:
        raise ValueError(f"Invalid saved CR data or MC: {region}")
    blocks = []
    for nb, (category, label) in enumerate(CATEGORIES):
        groups = {key: np.zeros(n) for key in canonical.GROUP_ORDER}
        variance = np.zeros(n)
        for sample, values in samples.items():
            process = sample if sample in BACKGROUND_DISPLAY_LABELS else background_process_for_sample(sample)
            groups[BACKGROUND_DISPLAY_LABELS[process]] += values[nb, stage, :, 0]
            variance += values[nb, stage, :, 1]
        background = sum(groups.values())
        observed = data[nb, stage]
        key = ("before", "after")[stage]
        if (observed.sum() != row["totals"]["data"][category][key]
                or not np.isclose(background.sum(), row["totals"]["mc"][category][key])):
            raise ValueError(f"Rendered yields differ from saved totals: {region}/{category}/{key}")
        unc = np.sqrt(variance)
        blocks.append(_cut_block(region, stage, label, edges, groups, observed, unc, unc))
    return blocks


def render(result, output):
    """Render both saved cut stages and both Nb categories without reweighting."""
    edges = np.asarray(result["recoil_display_edges"], dtype=float)
    if (result.get("year") != "2024" or result.get("regime") != "highdm"
            or result.get("cut") != "finite mTb >= 175 GeV"
            or result.get("sr_observations_read") is not False
            or result.get("last_recoil_bin_open") is not True
            or result.get("completed") != result.get("total")
            or tuple(result["regions"]) != REGIONS or set(result["results"]) != set(REGIONS)
            or edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("Expected the complete stored 2024 high-dM CR cut diagnostic")

    outdir = Path(output)
    lumi = luminosity_fb(result["year"])
    for region in REGIONS:
        outbase = outdir / f"recoil_{region}"
        panels = [_blocks(result["results"][region], edges, region, stage) for stage in range(2)]
        _draw_cut_comparison(panels, region, lumi, "MC stat. unc.", outbase, pdf=False)


def _draw_cut_comparison(panels, region, lumi, uncertainty_label, outbase, *, pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("CMS")
    fig = plt.figure(figsize=(24., 11.))
    axes = []
    peaks = []
    try:
        for stage, blocks in enumerate(panels):
            offset = 5.5 if stage == 0 else 0.
            grid = fig.add_gridspec(2, 1, height_ratios=[3., 1.1], hspace=.04,
                left=matplotlib.rcParams["figure.subplot.left"],
                right=matplotlib.rcParams["figure.subplot.right"],
                bottom=(offset+1.)/11., top=(offset+5.)/11.)
            ax = fig.add_subplot(grid[0])
            rax = fig.add_subplot(grid[1], sharex=ax)
            xlabel = (r"$p_{\mathrm{T}}^{\mathrm{miss}}$ (GeV)" if region in ("LLCR", "QCDCR")
                      else r"Recoil $p_{\mathrm{T}}$ (GeV)")
            canonical.draw_flat_blocks(blocks, outbase, xlabel=xlabel, reference_style=True,
                show_yields=True, luminosity_fb=lumi, uncertainty_label_override=uncertainty_label,
                figure_axes=(fig, ax, rax), legend_ncol=9)
            ax.tick_params(axis="x", labelbottom=False)
            axes.append(ax)
            peaks.extend(float(np.max(np.maximum(b["data"], b["background"] + b["background_unc"])))
                         for b in blocks)
        for ax in axes:
            ax.set_ylim(top=11.5 * max(peaks))
        fig.savefig(outbase.with_suffix(".png"), dpi=180)
        if pdf:
            fig.savefig(outbase.with_suffix(".pdf"))
    finally:
        plt.close(fig)


def render_prefit_comparison(payload, outbase):
    """Draw saved six-era LLCR data and supplied prefit MC bands; do not recalculate them."""
    from TROTASR.utils.cr_binning import DIAGNOSTIC_GROUPS
    from TROTASR.utils.sr_binning import NATIVE_EDGES

    edges = [NATIVE_EDGES[group[0]] for group in DIAGNOSTIC_GROUPS] + [NATIVE_EDGES[-1]]
    panels = []
    for stage, key in enumerate(("before", "after")):
        blocks = []
        for category, label in CATEGORIES:
            row = payload["stages"][key][category]
            groups = {g: np.asarray(v, dtype=float) for g, v in row["groups"].items()}
            observed, unc, stat = (np.asarray(row[f], dtype=float) for f in
                                   ("data", "background_unc", "background_stat_unc"))
            if (not groups or set(groups) - set(canonical.GROUP_ORDER)
                    or any(v.shape != (len(edges)-1,) or not np.isfinite(v).all()
                           for v in (*groups.values(), observed, unc, stat))
                    or np.any(unc < 0) or np.any(stat < 0) or np.any(observed < 0)
                    or np.any(observed != np.floor(observed))):
                raise ValueError(f"Invalid prefit LLCR arrays: {key}/{category}")
            blocks.append(_cut_block("LLCR", stage, label, edges, groups, observed, unc, stat))
        panels.append(blocks)
    outbase = Path(outbase)
    _draw_cut_comparison(panels, "LLCR", payload["luminosity_fb"],
                         payload["uncertainty_label"], outbase, pdf=True)
    return {fmt: str(outbase.with_suffix("." + fmt)) for fmt in ("png", "pdf")}
