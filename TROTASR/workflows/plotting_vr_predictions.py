"""Render VR predictions and observed JetMET with the canonical stack style."""

from pathlib import Path
import re

import numpy as np

from TROTASR.utils.paths import EARLY_ERAS, ERAS, internal_path, luminosity_fb
from TROTASR.utils.plot_adapter import renderer
from TROTASR.utils.sr_binning import STORAGE_END, label
from TROTASR.utils.renderers.background_process_groups import BACKGROUND_DISPLAY_LABELS


SCHEMA = "trotasr_vr_prefit_postfit_v1"
VIEWS = {"highdm": ERAS, "lowdm_early": EARLY_ERAS, "lowdm_late": ERAS[-2:]}
STAGES = ("prefit", "postfit")
LOW_KEYS = tuple(f"Nb{nb}_NISR{i}" for nb in ("1", "2plus") for i in ("0", "1", "2plus"))
BAND_LABELS = {"prefit": "VR MC stat. + CR transfer unc.",
               "postfit": "VR MC stat. + CR postfit + transfer unc."}
DISPLAY = {**BACKGROUND_DISPLAY_LABELS, "VV": "VV+VVV", "DYJet": "DY",
           "WJet": "W -> lv", "ZJet": "Z -> vv", "GJ": "Photon+jet"}
SIGNAL = re.compile(r"(?:T2(?:bW|tb)_)?mStop\d+_mLSP\d+")
NOMINAL_BAND = "VR MC stat. unc."


def _values(values, n, what, *, nonnegative=False, integral=False):
    a = np.asarray(values, dtype=float)
    if (a.shape != (n,) or not np.isfinite(a).all() or (nonnegative and np.any(a < 0))
            or (integral and np.any(a != np.floor(a)))):
        raise ValueError(f"Invalid {what} for a {n}-bin VR view")
    return a


def _covered(coverage):
    return (isinstance(coverage, (list, tuple)) and len(coverage) == 2
            and all(type(n) is int and n > 0 for n in coverage) and coverage[0] == coverage[1])


def _observations(mc, data):
    if (mc.get("schema") != "trotasr_vr_mc_nominal_v1" or mc.get("status") != "complete"
            or mc.get("mc_only") is not True or mc.get("sr_observations_read") is not False
            or mc.get("vr_observations_read") is not False
            or set(mc.get("coverage", {})) != set(ERAS)
            or any(not all(_covered(mc["coverage"][era].get(kind)) for kind in ("background", "signal"))
                   for era in ERAS)):
        raise ValueError("A complete blinded six-era VR MC summary is required")
    if (data.get("schema") != "trotasr_vr_jetmet_v1" or data.get("status") != "complete"
            or data.get("sr_observations_used") is not False or data.get("sr_histograms_filled") is not False
            or data.get("low_bins") != 30 or data.get("high_bins") != 118
            or set(data.get("coverage", {})) != set(ERAS)
            or set(data.get("observed_by_era", {})) != set(ERAS)
            or not all(_covered(data["coverage"][era]) for era in ERAS)):
        raise ValueError("Complete six-era JetMET observations from the VR-only data stage are required")
    counts = {era: _values(data["observed_by_era"][era], 148, f"{era} JetMET observations",
                           nonnegative=True, integral=True) for era in ERAS}
    combined = _values(data["observed_total"], 148, "combined JetMET observations",
                       nonnegative=True, integral=True)
    if not np.array_equal(sum(counts.values()), combined):
        raise ValueError("Combined VR observations disagree with the era counts")
    return counts


def _layout(mc, name, eras):
    if name == "highdm":
        blocks = []
        for i, category in enumerate(mc["high_categories"], 1):
            edges = np.asarray(category["edges"], dtype=float)
            if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                    or not np.all(np.diff(edges) > 0) or edges[-1] != STORAGE_END):
                raise ValueError("Invalid high-dM VR MET edges")
            blocks.append(dict(key=f"class{i}", label=label(category), size=len(edges) - 1,
                               edges=[*edges[:-1].tolist(), None],
                               predicates=category["predicates"]))
        if sum(b["size"] for b in blocks) != 118:
            raise ValueError("High-dM VR must retain 118 bins")
        return blocks

    definitions = mc.get("low_edges_by_era", {})
    if set(definitions) != set(ERAS) or any(set(definitions[era]) != set(LOW_KEYS) for era in eras):
        raise ValueError("Missing approved low-dM VR GNN categories")
    blocks = []
    for key in LOW_KEYS:
        a = _values(definitions[eras[0]][key], 6, "GNN boundaries")
        if (a[0] != 0 or a[-1] != 1 or not np.all(np.diff(a) > 0)
                or any(not np.array_equal(a, definitions[era][key]) for era in eras[1:])):
            raise ValueError("A low-dM VR display cannot combine different GNN boundaries")
        nb, isr = key.split("_")
        nb_label = r"$N_b=1$" if nb == "Nb1" else r"$N_b\geq2$"
        isr_label = r"$N_{\mathrm{ISR}}\geq2$" if isr == "NISR2plus" else rf"$N_{{\mathrm{{ISR}}}}={isr[-1]}$"
        blocks.append(dict(key=key, label=nb_label + "\n" + isr_label, size=5, edges=a.tolist()))
    return blocks


def _vectors(stage, n):
    bkg = _values(stage["background"], n, "background")
    unc = _values(stage["background_unc"], n, "background uncertainty", nonnegative=True)
    stat = _values(stage["background_stat_unc"], n, "VR MC statistical uncertainty", nonnegative=True)
    groups = {}
    for group, vals in stage["groups"].items():
        display = DISPLAY.get(group, group)
        if display not in renderer.GROUP_ORDER or display in groups:
            raise ValueError(f"Unknown or duplicate VR stack group {group!r}")
        groups[display] = _values(vals, n, f"{group} stack")
    if not groups or not np.allclose(sum(groups.values()), bkg, rtol=1e-8, atol=1e-8):
        raise ValueError("VR background must equal the sum of the plotted stack groups")
    if np.any(unc + 1e-8 < stat):
        raise ValueError("VR total background band is narrower than its MC statistical component")
    return dict(background=bkg, background_unc=unc, background_stat_unc=stat, groups=groups)


def _fit_category_boxes(fig, panels):
    """Keep the canonical category labels within their own bin blocks."""
    fig.canvas.draw()
    canvas_renderer = fig.canvas.get_renderer()
    for ax, layout in panels:
        labels = [artist for artist in ax.texts if artist.get_bbox_patch() is not None]
        if len(labels) != len(layout):
            raise RuntimeError("Canonical renderer did not supply the expected VR category boxes")
        bin_width = ax.bbox.width / sum(b["size"] for b in layout)
        for artist, category in zip(labels, layout):
            width = artist.get_bbox_patch().get_window_extent(canvas_renderer).width
            allowed = .96 * bin_width * category["size"]
            if width > allowed:
                artist.set_fontsize(artist.get_fontsize() * allowed / width)


def _draw(name, eras, layout, prediction, observed, outbase, *, luminosities=None,
          stages=STAGES, band_labels=BAND_LABELS, titles=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    import mplhep as hep

    hep.style.use("CMS")
    era_lumi = sum(luminosity_fb(era) if luminosities is None else luminosities[era] for era in eras)
    if not np.isfinite(era_lumi) or era_lumi <= 0:
        raise ValueError("Invalid luminosity for the VR display")
    canvas_height = 5.5 * len(stages)
    fig = plt.figure(figsize=(24.0, canvas_height))
    panels, category_panels = {}, []
    peak = max(float(np.max(observed)), *(float(np.max(prediction[s]["background"]
                           + prediction[s]["background_unc"])) for s in stages))
    try:
        for index, stage in enumerate(stages):
            offset = 5.5 * (len(stages) - index - 1)
            grid = fig.add_gridspec(2, 1, height_ratios=[3., 1.1], hspace=.09,
                                    bottom=(offset + 1.) / canvas_height,
                                    top=(offset + 5.) / canvas_height)
            ax = fig.add_subplot(grid[0])
            rax = fig.add_subplot(grid[1], sharex=ax)
            values = prediction[stage]
            scope = r"high $\Delta m$" if name == "highdm" else r"low $\Delta m$"
            title = stage.capitalize() if titles is None else titles[stage]
            blocks, offset = [], 0
            for category in layout:
                nbin = category["size"]
                slc = slice(offset, offset + nbin)
                hi = name == "highdm"
                blocks.append(dict(nbin=nbin, label=category["label"], label_box=True,
                    label_fontsize=9.5 if hi else 17, label_box_pad=.12 if hi else .30,
                    label_va="bottom", category_labels_on_main=True, category_label_y=.04,
                    main_panel_ymax_factor=4., main_annotation=f"{title} - {scope} VR",
                    main_annotation_y=.78, main_annotation_fontsize=24, main_annotation_fontweight="bold",
                    groups={k: v[slc] for k, v in values["groups"].items()},
                    background=values["background"][slc], background_unc=values["background_unc"][slc],
                    background_stat_unc=values["background_stat_unc"][slc], data=observed[slc],
                    data_unc=renderer.poisson_unc(observed[slc]), signals={},
                    xlabels=[str(i) for i in range(offset + 1, offset + nbin + 1)],
                    physics_scope="VR", unit_area=False, blind_data=False))
                offset += nbin
            info = renderer.draw_flat_blocks(blocks, outbase, xlabel="VR bin", reference_style=True,
                show_yields=False, uncertainty_label_override=band_labels[stage],
                luminosity_fb=era_lumi,
                x_tick_rotation=90. if name == "highdm" else 0.,
                figure_axes=(fig, ax, rax), legend_ncol=10)
            floor = ax.get_ylim()[0]
            if peak > floor:
                # The same y scale in both stages leaves a clear row for the
                # stage title below the legend and above the largest bin.
                ymax = max(11.5 * peak, floor * (1.15 * peak / floor) ** (1. / .74))
                ax.set_ylim(top=ymax)
            info["main_panel_ylim"] = [float(v) for v in ax.get_ylim()]
            info["nonpositive_background_bins_1based"] = (np.flatnonzero(values["background"] <= 0) + 1).tolist()
            info["uncertainty_label"] = band_labels[stage]
            ax.tick_params(axis="x", labelbottom=False)
            rax.yaxis.label.set_fontsize(24)
            panels[stage] = info
            category_panels.append((ax, layout))
        _fit_category_boxes(fig, category_panels)
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        crop = Bbox.from_extents(tight.x0 - .10, 0., tight.x1 + .10, canvas_height)
        fig.savefig(outbase.with_suffix(".png"), dpi=180, bbox_inches=crop, pad_inches=0)
        fig.savefig(outbase.with_suffix(".pdf"), bbox_inches=crop, pad_inches=0)
        return dict(years=list(eras), luminosity_fb=float(era_lumi), bins=len(observed), panels=panels,
                    categories=layout, figure_size_inches=[float(crop.width), canvas_height],
                    png=str(outbase.with_suffix(".png")), pdf=str(outbase.with_suffix(".pdf")))
    finally:
        plt.close(fig)


def render(prediction, mc, data, output):
    """Render three blinded-SR VR comparisons; make no fit or event reads."""
    if (prediction.get("schema") != SCHEMA or tuple(prediction.get("years", ())) != ERAS
            or set(prediction.get("views", {})) != set(VIEWS)
            or prediction.get("sr_observations_included", False) is not False
            or prediction.get("sr_observations_used", False) is not False):
        raise ValueError("Expected the approved six-era VR-only prediction payload")
    counts = _observations(mc, data)
    prepared = {}
    for name, eras in VIEWS.items():
        view = prediction["views"][name]
        if tuple(view.get("years", ())) != eras:
            raise ValueError(f"Incorrect era coverage for {name}")
        layout = _layout(mc, name, eras)
        n = sum(x["size"] for x in layout)
        obs = sum(counts[era] for era in eras)[30:] if name == "highdm" else sum(counts[era][:30] for era in eras)
        prepared[name] = eras, layout, {stage: _vectors(view[stage], n) for stage in STAGES}, obs
    output = internal_path(Path(output))
    return {name: _draw(name, *args, output / f"vr_{name}_prefit_postfit")
            for name, args in prepared.items()}


def _nominal_years(mc, eras, slc):
    groups, variance = {}, np.zeros(148)
    for era in eras:
        for sample, values in mc["era_samples"][era].items():
            if SIGNAL.fullmatch(sample):
                continue
            display = DISPLAY.get(sample, sample)
            if display not in renderer.GROUP_ORDER:
                raise ValueError(f"Unclassified VR background {sample!r} in {era}")
            values = np.asarray(values, dtype=float)
            if values.shape != (148, 3) or not np.isfinite(values).all() or np.any(values[:, 1:] < 0):
                raise ValueError(f"Invalid nominal VR histogram for {era}/{sample}")
            groups.setdefault(display, np.zeros(148))[:] += values[:, 0]
            variance += values[:, 1]
    stat_unc = np.sqrt(variance[slc])
    return _vectors(dict(groups={k: v[slc] for k, v in groups.items()},
                         background=sum(groups.values(), np.zeros(148))[slc],
                         background_unc=stat_unc, background_stat_unc=stat_unc), len(stat_unc))


def render_nominal(mc, data, output, luminosities=None):
    """Plot observed VR against nominal MC with statistical uncertainty only."""
    counts = _observations(mc, data)
    if (mc.get("fit_or_transfer_factors_applied") is not False
            or mc.get("fields") != ["sumw", "sumw2", "entries"]
            or set(mc.get("era_samples", {})) != set(ERAS)):
        raise ValueError("Require the unmodified era-by-era nominal VR histograms")
    if luminosities is not None and (set(luminosities) != set(ERAS)
            or any(not np.isfinite(float(v)) or float(v) <= 0 for v in luminosities.values())):
        raise ValueError("Require positive per-era luminosities for the six approved eras")
    luminosities = None if luminosities is None else {e: float(luminosities[e]) for e in ERAS}
    output = internal_path(Path(output))
    prepared = {}
    for name, eras in VIEWS.items():
        slc = slice(30, 148) if name == "highdm" else slice(0, 30)
        prepared[name] = (eras, _layout(mc, name, eras),
                          {"prefit": _nominal_years(mc, eras, slc)},
                          sum(counts[era][slc] for era in eras))
    return {name: _draw(name, *args, output / f"vr_{name}_nominal_prefit", luminosities=luminosities,
                        stages=("prefit",), band_labels={"prefit": NOMINAL_BAND},
                        titles={"prefit": "Nominal prefit (MC stat. only)"})
            for name, args in prepared.items()}


def main():
    import argparse
    import fcntl
    from TROTASR.utils.io import read_json, sha256, write_json
    from TROTASR.utils.paths import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-prefit", action="store_true",
                        help="Compare nominal background MC with JetMET using MC-statistical bands")
    parser.add_argument("--nominal-input", type=Path,
                        help="Render an exported, complete VR-only aggregate without the hep2 runtime")
    parser.add_argument("--output", type=Path, help="Nominal prefit figure directory")
    args = parser.parse_args()
    if (args.nominal_input or args.output) and not args.nominal_prefit:
        parser.error("--nominal-input and --output require --nominal-prefit")
    run_root = ROOT / "stats" / "run3"
    vr = run_root / "vr"
    output = internal_path(run_root / "plots" / "vr")
    if args.nominal_prefit:
        output = internal_path(args.output or output / "nominal_prefit")
        if args.nominal_input:
            source = internal_path(args.nominal_input)
            bundle = read_json(source)
            source_identity = dict(input=source.name, sha256=sha256(source))
        else:
            with (vr / "controller.lock").open("rb") as production:
                fcntl.flock(production, fcntl.LOCK_SH | fcntl.LOCK_NB)
                sources = {k: vr / (k + "_summary.json.gz") for k in ("mc", "data")}
                bundle = {k: read_json(p) for k, p in sources.items()}
                source_identity = {k: dict(name=p.name, size=p.stat().st_size,
                                           mtime_ns=p.stat().st_mtime_ns) for k, p in sources.items()}
                bundle["state"] = {k: {field: state[field] for field in ("status", "updated", "completed", "total")}
                                   for k in sources for state in [read_json(vr / (k + "_state.json"))]}
        output.mkdir(parents=True, exist_ok=True)
        with (output / "plot.lock").open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plots = render_nominal(bundle["mc"], bundle["data"], output, bundle.get("luminosities"))
            write_json(output / "plot_manifest.json", dict(status="generated_pending_visual_QA",
                scope="nominal background MC vs observed JetMET in VR; no transfer or fit",
                input_identity=source_identity, source_state=bundle.get("state"),
                plots=plots, files={Path(row[k]).name: sha256(row[k]) for row in plots.values()
                                    for k in ("png", "pdf")},
                vr_data_fitted=False, sr_observations_used=False, fit_recalculated=False))
        print(dict(status="generated_pending_visual_QA", plots=list(plots)), flush=True)
        return

    compare_saved_fit()


def compare_saved_fit():
    """Render the saved CR fit after the VR responses are complete."""
    import fcntl
    from TROTASR.utils.io import read_json, write_json
    from TROTASR.utils.paths import ROOT
    from TROTASR.workflows.vr_fit_projection import project

    run_root = ROOT / "stats" / "run3"
    vr = run_root / "vr"
    output = internal_path(run_root / "plots" / "vr")
    with (vr / "controller.lock").open("rb") as production:
        fcntl.flock(production, fcntl.LOCK_SH | fcntl.LOCK_NB)
        response = read_json(vr / "response_summary.json.gz")
        mc = read_json(vr / "mc_summary.json.gz")
        if response.get("all_applicable_responses_available") is not True:
            raise ValueError("VR background responses are incomplete; do not publish a full postfit plot")
        if (response.get("sr_observations_read") is not False or response.get("mc_only") is not True
                or response.get("no_fit_or_transfer_factors_applied") is not True
                or response.get("source_nominal_reused") is not True):
            raise ValueError("Require the accepted MC-only VR response inventory")
        if (mc.get("status") != "complete"
                or any(response["coverage"][e] != mc["coverage"][e]["background"] for e in ERAS)):
            raise ValueError("VR response coverage does not match the completed nominal MC")
        output.mkdir(parents=True, exist_ok=True)
        with (output / "plot.lock").open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            payload = project(run_root, response)
            # The prediction is fully computed before opening the VR data summary.
            data = read_json(vr / "data_summary.json.gz")
            write_json(output / "prediction_payload.json.gz", payload)
            plots = render(payload, mc, data, output)
            write_json(output / "plot_manifest.json", dict(status="generated_pending_visual_QA",
                plots=plots, fit=payload["fit"], same_jetmet_after_projection=True,
                sr_observations_used=False, vr_data_fitted=False, fit_recalculated=False))
            print(dict(status="generated_pending_visual_QA", plots=list(plots),
                       covariance_quality=payload["fit"]["covariance_quality"]), flush=True)


if __name__ == "__main__":
    main()
