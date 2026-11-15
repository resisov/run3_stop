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
BAND_LABELS = {"prefit": "VR MC stat. + background syst.",
               "postfit": "VR MC stat. + CR fit + background syst."}
DISPLAY = {**BACKGROUND_DISPLAY_LABELS, "VV": "VV+VVV", "DYJet": "DY",
           "WJet": "W -> lv", "ZJet": "Z -> vv", "GJ": "Photon+jet"}
SIGNAL = re.compile(r"(?:T2(?:bW|tb)_)?mStop\d+_mLSP\d+")
NOMINAL_BAND = "VR MC stat. unc."
NORMALIZATION_BANDS = {"prefit": NOMINAL_BAND,
                       "postfit": "VR MC stat. + correlated CR normalization unc."}
NORMALIZATION_TITLES = {"prefit": "Nominal prefit", "postfit": "Postfit (CR normalization only)"}


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
    background_only = mc.get("background_only") is True
    if (mc.get("schema") != "trotasr_vr_mc_nominal_v1" or mc.get("status") not in ("complete", "incomplete")
            or mc.get("pilot") is True
            or mc.get("mc_only") is not True or mc.get("sr_observations_read") is not False
            or mc.get("vr_observations_read") is not False
            or set(mc.get("coverage", {})) != set(ERAS)
            or any(not _covered(mc["coverage"][era].get("background")) for era in ERAS)
            or (background_only and (mc.get("status") != "complete" or mc.get("full_era_signal_points") != 0
                    or mc.get("missing_era_signal_points", {}) is not None
                    or any(mc["coverage"][era].get("signal") != [0, 0] for era in ERAS)))
            or (not background_only and not isinstance(mc.get("missing_era_signal_points"), dict))):
        raise ValueError("A complete blinded six-era VR background summary is required")
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


def _mc_scope(mc):
    return dict(mc_scope="background_only" if mc.get("background_only") is True else "background_and_signal",
                signal_drawn=False)


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


def _mark_unmapped(rax, bins):
    """Annotate unmapped bins without replacing or covering Data/MC points."""
    for index in bins:
        rax.axvspan(index - .5, index + .5, facecolor="#f7c469", edgecolor="#9e5500",
                    alpha=.35, linewidth=.7, zorder=1.2, gid="vr_unmapped_bin")
    if bins:
        rax.text(.004, -.65, "Shaded: contains MC kept nominal (no approved CR mapping)",
                 transform=rax.transAxes, ha="left", va="top", fontsize=11, color="#744600")


def _draw(name, eras, layout, prediction, observed, outbase, *, luminosities=None,
          stages=STAGES, band_labels=BAND_LABELS, titles=None, unmapped_bins=None):
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
            if stage == "postfit" and unmapped_bins is not None:
                _mark_unmapped(rax, unmapped_bins)
                info["unmapped_bins_1based"] = list(unmapped_bins)
                info["unmapped_bins_preserved"] = True
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


def render(prediction, mc, data, output, *, normalization_only=None, luminosities=None):
    """Render three blinded-SR VR comparisons; make no fit or event reads."""
    if (prediction.get("schema") != SCHEMA or tuple(prediction.get("years", ())) != ERAS
            or set(prediction.get("views", {})) != set(VIEWS)
            or prediction.get("sr_observations_included", False) is not False
            or prediction.get("sr_observations_used", False) is not False):
        raise ValueError("Expected the approved six-era VR-only prediction payload")
    counts = _observations(mc, data)
    if normalization_only is None:
        normalization_only = (prediction.get("normalization_only") is True
                              or any("unmapped_bins_1based" in view for view in prediction["views"].values()))
    prepared = {}
    for name, eras in VIEWS.items():
        view = prediction["views"][name]
        if tuple(view.get("years", ())) != eras:
            raise ValueError(f"Incorrect era coverage for {name}")
        layout = _layout(mc, name, eras)
        n = sum(x["size"] for x in layout)
        obs = sum(counts[era] for era in eras)[30:] if name == "highdm" else sum(counts[era][:30] for era in eras)
        unmapped = None
        if normalization_only:
            bins = view.get("unmapped_bins_1based", ())
            if (not isinstance(bins, (list, tuple)) or any(type(b) is not int or not 1 <= b <= n for b in bins)
                    or len(set(bins)) != len(bins)):
                raise ValueError(f"Invalid unmapped bin indices in {name}")
            unmapped = sorted(bins)
        prepared[name] = (eras, layout, {stage: _vectors(view[stage], n) for stage in STAGES}, obs), unmapped
    output = internal_path(Path(output))
    stem = "cr_normalization_only" if normalization_only else "prefit_postfit"
    return {name: _draw(name, *args, output / f"vr_{name}_{stem}", luminosities=luminosities,
                        band_labels=NORMALIZATION_BANDS if normalization_only else BAND_LABELS,
                        titles=NORMALIZATION_TITLES if normalization_only else None, unmapped_bins=unmapped)
            for name, (args, unmapped) in prepared.items()}


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


def _display_inputs(mc, data, luminosities, *, nominal=False):
    from TROTASR.utils.statistical_templates import json_ready

    counts = _observations(mc, data)
    out = {}
    for name, eras in VIEWS.items():
        slc = slice(30, 148) if name == "highdm" else slice(0, 30)
        out[name] = dict(geometry=_layout(mc, name, eras),
                         observed=sum(counts[era][slc] for era in eras),
                         luminosity=sum(luminosity_fb(e) if luminosities is None else float(luminosities[e])
                                        for e in eras))
        if nominal:
            out[name]["prediction"] = _nominal_years(mc, eras, slc)
    return json_ready(out)


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


def _completed_figures(output, stem):
    from TROTASR.utils.io import read_json

    receipt = output / "plot_manifest.json"
    if not receipt.is_file():
        return None
    prior = read_json(receipt)
    expected = {f"vr_{view}_{stem}.{extension}" for view in VIEWS for extension in ("png", "pdf")}
    if (set(prior.get("plots", {})) != set(VIEWS)
            or set(prior.get("files", ())) != expected
            or prior.get("sr_observations_used") is not False
            or prior.get("vr_data_fitted") is not False or prior.get("fit_recalculated") is not False
            or any(not (output / name).is_file() or not (output / name).stat().st_size for name in expected)):
        raise ValueError("Incompatible or incomplete VR figures; preserve existing artifacts")
    return prior


def main():
    import argparse
    import fcntl
    from TROTASR.utils.io import read_json, write_json
    from TROTASR.utils.paths import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--nominal-prefit", action="store_true",
                      help="Compare nominal background MC with JetMET using MC-statistical bands")
    mode.add_argument("--cr-normalization-only", action="store_true",
                      help="Apply exact saved CR ratios to existing nominal VR, without event reads")
    parser.add_argument("--nominal-input", type=Path,
                        help="Render an exported, complete VR-only aggregate without the hep2 runtime")
    parser.add_argument("--normalization-input", type=Path,
                        help="Optional compact export of saved CR normalization draws")
    parser.add_argument("--output", type=Path, help="Figure directory for either reduced mode")
    args = parser.parse_args()
    if (args.nominal_input or args.output) and not (args.nominal_prefit or args.cr_normalization_only):
        parser.error("--nominal-input and --output require a reduced mode")
    if args.normalization_input and not args.cr_normalization_only:
        parser.error("--normalization-input requires --cr-normalization-only")
    run_root = ROOT / "stats" / "run3"
    vr = run_root / "vr"
    output = internal_path(run_root / "plots" / "vr")
    if args.cr_normalization_only:
        from TROTASR.workflows.vr_normalization import project, saved_rates

        output = internal_path(args.output or output / "cr_normalization_only")
        source = internal_path(args.nominal_input) if args.nominal_input else None
        bundle = read_json(source) if source else {x: read_json(vr / (x + "_summary.json.gz"))
                                                   for x in ("mc", "data")}
        rates_source = internal_path(args.normalization_input) if args.normalization_input else None
        output.mkdir(parents=True, exist_ok=True)
        with (output / "plot.lock").open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            display = _display_inputs(bundle["mc"], bundle["data"], bundle.get("luminosities"),
                                      nominal=True)
            prior = _completed_figures(output, "cr_normalization_only")
            if prior is not None:
                if prior.get("display_inputs", display) != display:
                    raise ValueError("Current VR yields or bins differ from the existing figures")
                if rates_source:
                    saved = read_json(output / "prediction_payload.json.gz")
                    current = project(bundle["mc"], read_json(rates_source))
                    fields = ("views", "covariance", "years", "normalization_only")
                    from TROTASR.utils.statistical_templates import json_ready
                    if any(saved.get(k) != json_ready(current.get(k)) for k in fields):
                        raise ValueError("Supplied CR normalization gives a different VR prediction")
                print(prior["status"], "(verified existing outputs)")
                return
            rates = read_json(rates_source) if rates_source else saved_rates()
            prediction = project(bundle["mc"], rates)
            plots = render(prediction, bundle["mc"], bundle["data"], output,
                           normalization_only=True, luminosities=bundle.get("luminosities"))
            write_json(output / "prediction_payload.json.gz", prediction)
            write_json(output / "plot_manifest.json", dict(status="generated_pending_visual_QA",
                scope=prediction["scope"], fit=prediction["fit"], plots=plots,
                display_inputs=display,
                nominal_input=str(source) if source else None,
                rates_input=str(rates_source) if rates_source else None,
                source_state=bundle.get("state"),
                **_mc_scope(bundle["mc"]),
                files=sorted(Path(row[k]).name for row in plots.values() for k in ("png", "pdf")),
                same_jetmet=True, vr_data_fitted=False, sr_observations_used=False, fit_recalculated=False))
        print(dict(status="generated_pending_visual_QA", plots=list(plots)), flush=True)
        return
    if args.nominal_prefit:
        output = internal_path(args.output or output / "nominal_prefit")
        if args.nominal_input:
            source = internal_path(args.nominal_input)
            bundle = read_json(source)
            source_identity = dict(input=source.name, size=source.stat().st_size)
        else:
            with (vr / "controller.lock").open("rb") as production:
                fcntl.flock(production, fcntl.LOCK_SH | fcntl.LOCK_NB)
                sources = {k: vr / (k + "_summary.json.gz") for k in ("mc", "data")}
                bundle = {k: read_json(p) for k, p in sources.items()}
                source_identity = {k: dict(name=p.name, size=p.stat().st_size,
                                           mtime_ns=p.stat().st_mtime_ns) for k, p in sources.items()}
                states = {k: vr / (k + "_state.json") for k in sources}
                bundle["state"] = {k: {field: state.get(field) for field in ("status", "updated", "completed", "total")}
                                   for k, path in states.items() if path.is_file() for state in [read_json(path)]}
        output.mkdir(parents=True, exist_ok=True)
        with (output / "plot.lock").open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            display = _display_inputs(bundle["mc"], bundle["data"], bundle.get("luminosities"), nominal=True)
            prior = _completed_figures(output, "nominal_prefit")
            if prior is not None:
                if prior.get("display_inputs", display) != display:
                    raise ValueError("Current VR yields or bins differ from the existing figures")
                print(prior["status"], "(verified existing outputs)")
                return
            plots = render_nominal(bundle["mc"], bundle["data"], output, bundle.get("luminosities"))
            write_json(output / "plot_manifest.json", dict(status="generated_pending_visual_QA",
                scope="nominal background MC vs observed JetMET in VR; no transfer or fit",
                input_identity=source_identity, source_state=bundle.get("state"), display_inputs=display,
                **_mc_scope(bundle["mc"]),
                plots=plots, files=sorted(Path(row[k]).name for row in plots.values()
                                         for k in ("png", "pdf")),
                vr_data_fitted=False, sr_observations_used=False, fit_recalculated=False))
        print(dict(status="generated_pending_visual_QA", plots=list(plots)), flush=True)
        return

    compare_saved_fit()


def compare_saved_fit():
    """Publish only after the response controller has released its lock."""
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
        if (mc.get("status") not in ("complete", "incomplete")
                or any(response["coverage"][e] != mc["coverage"][e]["background"] for e in ERAS)):
            raise ValueError("VR response coverage does not match the completed nominal MC")
        output.mkdir(parents=True, exist_ok=True)
        with (output / "plot.lock").open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prior = _completed_figures(output, "prefit_postfit")
            if prior is not None:
                print(prior["status"], "(verified existing outputs)")
                return
            payload = project(run_root, response)
            # The prediction is fully computed before opening the VR data summary.
            data = read_json(vr / "data_summary.json.gz")
            write_json(output / "prediction_payload.json.gz", payload)
            plots = render(payload, mc, data, output)
            write_json(output / "plot_manifest.json", dict(status="generated_pending_visual_QA",
                plots=plots, fit=payload["fit"], same_jetmet_after_projection=True,
                **_mc_scope(mc),
                files=sorted(Path(row[k]).name for row in plots.values() for k in ("png", "pdf")),
                sr_observations_used=False, vr_data_fitted=False, fit_recalculated=False))
            print(dict(status="generated_pending_visual_QA", plots=list(plots),
                       covariance_quality=payload["fit"]["covariance_quality"]), flush=True)


if __name__ == "__main__":
    main()
