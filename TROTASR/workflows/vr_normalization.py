"""VR comparison with saved CR normalization only; no event inputs."""

import math
from collections import defaultdict

import numpy as np

from TROTASR.utils.paths import EARLY_ERAS, ERAS, ROOT
from TROTASR.utils.vr_binning import HIGH_VR_RATE_EDGES, N_HIGH_BINS, N_LOW_BINS, N_VR_BINS

VIEWS = {"highdm": ERAS, "lowdm_early": EARLY_ERAS, "lowdm_late": ERAS[-2:]}
CONTROLLED = ("Top", "WtoLNu", "QCD", "Zto2Nu")
OTHER = ("GJ", "DY", "VV")
CR12 = (250., 300., 350., 400., 500., 800., math.inf)
GAMMA = (250., 300., 350., 400., 500., math.inf)
HIGH_RECOIL = tuple(zip(HIGH_VR_RATE_EDGES, (*HIGH_VR_RATE_EDGES[1:], math.inf)))


def _cells(mc):
    bins = [("lowdm", "Nb1" if i < N_LOW_BINS // 2 else "Nb2plus", None, None)
            for i in range(N_LOW_BINS)]
    for category in mc["high_categories"]:
        predicate = category["predicates"]["Nb"]
        nb = {"Nb1": {"eq": 1}, "Nb2plus": {"ge": 2}}
        groups = [key for key in nb if nb[key] == predicate]
        if len(groups) != 1:
            raise ValueError("Unexpected Nb predicate")
        edges = (*category["edges"][:-1], math.inf)
        bins.extend(("highdm", groups[0], lo, hi) for lo, hi in zip(edges, edges[1:]))
    if len(bins) != N_VR_BINS:
        raise ValueError("Invalid VR normalization bin count")
    return bins


def _rate(geom, process):
    mode, nb, low, high = geom
    kind = {"Top": "ll_norm", "WtoLNu": "ll_norm", "QCD": "qcd_norm",
            "Zto2Nu": "sgamma_shape"}[process]
    if mode == "lowdm":
        if process == "Zto2Nu":
            return None, "no_Z_recoil_breakdown"
        unit = "inclusive"
    else:
        edges = GAMMA if process == "Zto2Nu" else CR12
        slots = [i for i, (a, b) in enumerate(zip(edges, edges[1:])) if a <= low < high <= b]
        if len(slots) != 1:
            return None, "crosses_CR_recoil_boundary"
        unit = f"bin{slots[0]}"
    return f"CMS_NPS26012_{kind}_{mode}_{nb}_{unit}", None


def _aggregates(mc, geometry, signal):
    """Match the displayed high-dM bins to their finer, exact CR recoil cells."""
    if (tuple(mc.get("high_recoil_edges", ())) != HIGH_VR_RATE_EDGES
            or set(mc.get("era_high_recoil_samples", {})) != set(ERAS)):
        raise ValueError("Require the exact high-dM VR recoil axes for CR normalization")
    aggregates = {}
    for era in ERAS:
        coarse, fine = mc["era_samples"][era], mc["era_high_recoil_samples"][era]
        if ({name for name in coarse if not signal.fullmatch(name)}
                != {name for name in fine if not signal.fullmatch(name)}):
            raise ValueError(f"Exact high-dM recoil sample coverage disagrees for {era}")
        aggregates[era] = {}
        for process, rows in coarse.items():
            if signal.fullmatch(process):
                continue
            if process not in (*CONTROLLED, *OTHER):
                raise ValueError(f"Unrecognized background: {process}")
            a, cells = np.asarray(rows, float), np.asarray(fine[process], float)
            if (a.shape != (N_VR_BINS, 3) or cells.shape != (N_HIGH_BINS, len(HIGH_RECOIL), 3)
                    or not np.isfinite(a).all() or not np.isfinite(cells).all()
                    or np.any(a[:, 1:] < 0) or np.any(cells[:, :, 1:] < 0)
                    or np.any(a[:, 2] != np.floor(a[:, 2]))
                    or np.any(cells[:, :, 2] != np.floor(cells[:, :, 2]))):
                raise ValueError(f"Invalid VR normalization aggregates for {era}/{process}")
            high = a[N_LOW_BINS:]
            for i, (_, _, dlo, dhi) in enumerate(geometry[N_LOW_BINS:]):
                allowed = [dlo <= lo < hi <= dhi for lo, hi in HIGH_RECOIL]
                if np.any(cells[i, np.logical_not(allowed)] != 0):
                    raise ValueError(f"High-dM recoil cells outside their display bin for {era}/{process}")
            subtotals = cells.sum(axis=1)
            # Separate accumulation can change floating-point rounding; counts must agree exactly.
            tolerances = 1e-10 + 1e-10 * np.sum(np.abs(cells[:, :, :2]), axis=1)
            if (np.any(np.abs(subtotals[:, :2] - high[:, :2]) > tolerances)
                    or not np.array_equal(subtotals[:, 2], high[:, 2])):
                raise ValueError(f"High-dM recoil cells do not reaggregate for {era}/{process}")
            aggregates[era][process] = (a, cells)
    return aggregates


def saved_rates(fit_root=None):
    """Read saved CR normalization coordinates and their correlated draws."""
    from TROTASR.utils.io import read_json
    from TROTASR.utils.era_merge import HIGHDM_Z_SHAPE
    from TROTASR.utils.paths import internal_path
    from TROTASR.workflows.cr_linked_workspace import RECOIL
    from TROTASR.workflows.vr_fit_projection import _card, _load_fit, fit_product_root

    baseline = ROOT / "stats" / "run3"
    run_root = fit_product_root(fit_root) if fit_root is not None else baseline
    grid = run_root / "grid"
    manifest = read_json(grid / "manifest.json")
    if (manifest.get("sr_data_blinded") is not True or manifest.get("era_merged") is not True
            or tuple(manifest["years"]) != ERAS):
        raise ValueError("Expected the accepted blinded six-era grid")
    spec = manifest["cr_only_card"]
    path = internal_path(ROOT / spec["card"])
    if path.parent != grid / "datacards":
        raise ValueError("Use the registered CR-only card directory")

    def rate_bindings(card):
        _, rates, _, _ = _card(card)
        bindings = {}
        for (channel, proc), (name, era_scale, _) in rates.items():
            if not channel.startswith(("LLCR_", "QCDCR_", "GCR_")) or name in RECOIL:
                continue
            era = proc.rsplit("_", 1)[-1]
            family = proc.split("_", 1)[0]
            if era not in ERAS or family not in ("Top", "WJet", "QCD", "PhotonJet"):
                raise ValueError("Unrecognized CR normalization binding")
            # The same eraScale multiplies alpha in prefit and postfit,
            # so it cancels only in the ratio applied to the nominal VR.
            key = (channel, proc, name)
            if key in bindings or not math.isfinite(era_scale) or era_scale <= 0:
                raise ValueError(f"Ambiguous CR binding: {key}")
            bindings[key] = era_scale
        return bindings

    by_column = rate_bindings(path)
    if fit_root is not None:
        frozen = read_json(baseline / "grid" / "manifest.json")
        frozen_card = internal_path(ROOT / frozen["cr_only_card"]["card"])
        if (frozen_card.parent != baseline / "grid" / "datacards"
                or by_column.keys() != rate_bindings(frozen_card).keys()):
            raise ValueError("The selected CR model changes the adopted VR rate-to-recoil mapping")
    bindings = {}
    for (_, proc, name), era_scale in by_column.items():
        key = f"{proc.rsplit('_', 1)[-1]}/{proc.split('_', 1)[0]}/{name}"
        if key in bindings:
            raise ValueError(f"Ambiguous CR binding: {key}")
        bindings[key] = era_scale
    parameters = {k.rsplit("/", 1)[-1] for k in bindings}
    pre, post, _, draws, _, fit = _load_fit(run_root, parameters, parameters, set(),
                                           explicit_card=path if fit_root is not None else None)
    if set(pre) != parameters or set(post) != parameters or len(draws) < 2:
        raise ValueError("Missing saved normalization coordinates")
    return dict(prefit=pre, postfit=post, draws=draws, fit=fit, bindings=bindings,
                **{HIGHDM_Z_SHAPE: manifest.get(HIGHDM_Z_SHAPE)},
                card=spec["card"], **({"manifest": fit["cr_manifest"]} if fit_root is not None else {}))


def project(mc, rates, *, interim=False):
    """Apply identifiable CR ratios to complete VR MC or an explicit interim view."""
    from TROTASR.utils.era_merge import HIGHDM_Z_SHAPE
    from TROTASR.workflows.plotting_vr_predictions import SIGNAL, _covered, _require_high_definition

    def partial_coverage(counts):
        return (isinstance(counts, (list, tuple)) and len(counts) == 2
                and all(type(n) is int for n in counts)
                and 0 <= counts[0] <= counts[1] and counts[1] > 0)

    coverage = mc.get("coverage", {})
    covered = partial_coverage if interim else _covered
    _require_high_definition(mc)
    if (mc.get("fields") != ["sumw", "sumw2", "entries"]
            or mc.get("status") not in ("complete", "incomplete") or mc.get("mc_only") is not True
            or mc.get("pilot") is True
            or mc.get("fit_or_transfer_factors_applied") is not False
            or mc.get("sr_observations_read") is not False
            or mc.get("vr_observations_read") is not False
            or set(coverage) != set(ERAS)
            or any(not covered(coverage[era].get("background")) for era in ERAS)
            or (interim and not any(coverage[era]["background"][0] for era in ERAS))
            or set(mc["era_samples"]) != set(ERAS)):
        if interim:
            raise ValueError("Require current VR MC receipts and honest partial six-era coverage")
        raise ValueError("Require the completed, unchanged VR MC aggregate")
    if interim and any(not coverage[era]["background"][0]
                       and (mc["era_samples"][era] or mc.get("era_high_recoil_samples", {}).get(era))
                       for era in ERAS):
        raise ValueError("An era with no completed MC receipts cannot contribute to the interim view")
    geometry = _cells(mc)
    aggregates = _aggregates(mc, geometry, SIGNAL)
    names = tuple(sorted(rates["prefit"]))
    pre = np.array([rates["prefit"][n] for n in names])
    post = np.array([rates["postfit"][n] for n in names])
    draws = np.array([[p[n] for n in names] for p in rates["draws"]])
    if (draws.ndim != 2 or len(draws) < 2 or not np.isfinite(pre).all()
            or not np.isfinite(post).all() or not np.isfinite(draws).all()
            or min(pre) <= 0 or min(post) < 0 or np.any(draws < 0)):
        raise ValueError("Invalid saved CR normalization or covariance draws")
    ratios = post / pre
    draw_ratios = draws / pre
    index = {n: i for i, n in enumerate(names)}
    z_shape = rates.get(HIGHDM_Z_SHAPE)
    z_normalizers = {}
    for era, groups in (z_shape or {}).items():
        for nb, coefficients in groups.items():
            indices = [index[name] for name in coefficients]
            weights = np.array(list(coefficients.values()))
            reference = float(pre[indices] @ weights)
            fitted = float(post[indices] @ weights)
            varied = draws[:, indices] @ weights
            if (not math.isfinite(reference) or reference <= 0. or not math.isfinite(fitted)
                    or fitted <= 0. or not np.isfinite(varied).all() or np.any(varied <= 0.)):
                raise ValueError(f"Invalid high-dM Z shape normalization for {era}/{nb}")
            z_normalizers[era, nb] = fitted / reference, varied / reference
    result, shifts, mcvars = {}, [], []
    views = ("highdm",) if interim else tuple(VIEWS)
    for view in views:
        years = VIEWS[view]
        highdm = view == "highdm"
        bins = range(N_LOW_BINS, N_VR_BINS) if highdm else range(N_LOW_BINS)
        pre_y, post_y = defaultdict(lambda: np.zeros(len(bins))), defaultdict(lambda: np.zeros(len(bins)))
        pre_v, post_v = np.zeros(len(bins)), np.zeros(len(bins))
        terms = np.zeros((len(bins), len(names)))
        z_terms = defaultdict(lambda: np.zeros_like(terms))
        unmapped, mapped = defaultdict(list), defaultdict(list)
        for era in years:
            for process, (a, fine) in aggregates[era].items():
                for i, b in enumerate(bins):
                    pre_y[process][i] += a[b, 0]
                    pre_v[i] += a[b, 1]
                    mode, nb, _, _ = geometry[b]
                    slots = ((mode, nb, lo, hi) for lo, hi in HIGH_RECOIL) if highdm else (geometry[b],)
                    cells = fine[i] if highdm else (a[b],)
                    for geom, (y, var, entries) in zip(slots, cells):
                        factor = 1.
                        if process in CONTROLLED and (y != 0 or var != 0 or entries != 0):
                            name, reason = _rate(geom, process)
                            family = {"WtoLNu": "WJet", "Zto2Nu": "PhotonJet"}.get(process, process)
                            key = f"{era}/{family}/{name}"
                            recoil = {"recoil": [geom[2], geom[3] if math.isfinite(geom[3]) else None]} \
                                if highdm else {}
                            if name not in index or key not in rates["bindings"]:
                                unmapped[i + 1].append(dict(era=era, process=process,
                                    reason=reason or "no_saved_CR_binding", nominal_yield=float(y),
                                    ratio=None, **recoil))
                            else:
                                j = index[name]
                                factor = ratios[j]
                                if process == "Zto2Nu" and highdm and z_shape:
                                    factor /= z_normalizers[era, nb][0]
                                    z_terms[era, nb][i, j] += y
                                else:
                                    terms[i, j] += y
                                mapped[i + 1].append(dict(era=era, process=process,
                                    parameter=name, factor=float(factor), **recoil))
                        post_y[process][i] += y * factor
                        post_v[i] += var * factor**2
        shifted = draw_ratios @ terms.T
        for key, projection in z_terms.items():
            shifted += (draw_ratios @ projection.T) / z_normalizers[key][1][:, None]
        delta = shifted - shifted.mean(axis=0)
        cov = delta.T @ delta / len(draws)
        shifts.append(shifted)
        mcvars.append(post_v)

        def stage(groups, stat, total):
            return dict(groups={k: v.tolist() for k, v in groups.items()},
                        background=sum(groups.values()).tolist(),
                        background_stat_unc=np.sqrt(stat).tolist(),
                        background_unc=np.sqrt(total).tolist())
        result[view] = dict(years=list(years),
            prefit=stage(pre_y, pre_v, pre_v),
            postfit=stage(post_y, post_v, post_v + np.diag(cov)),
            unmapped_bins_1based=sorted(unmapped),
            no_mapped_component_bins_1based=sorted(set(unmapped) - set(mapped)),
            unmapped_components=dict(unmapped), mapped_components=dict(mapped))
    joint = np.hstack(shifts)
    delta = joint - joint.mean(axis=0)
    scope = ("exact saved CR normalization ratios only; unbound components unchanged"
             if not interim else "interim completed VR MC shards; exact saved CR normalization ratios only; "
                                 "unbound components unchanged; no exposure matching")
    return dict(schema="trotasr_vr_prefit_postfit_v1",
                status="interim_normalization_only" if interim else "complete_normalization_only",
                normalization_only=True, years=list(ERAS), views=result,
                scope=scope,
                prefit="existing nominal VR MC; no new transfer factors",
                excluded_from_bands="VR shape-response systematics and unmapped CR effects",
                source=dict(cr_card=rates.get("card"),
                            **({"cr_manifest": rates["manifest"]} if "manifest" in rates else {})),
                fit=rates["fit"],
                covariance=dict(view_order=list(views),
                    cr_normalization=(delta.T @ delta / len(draws)).tolist(),
                    independent_mc_variance=np.concatenate(mcvars).tolist()),
                fit_recalculated=False, vr_data_fitted=False, sr_observations_used=False,
                **({"interim": True, "mc_coverage": {e: list(coverage[e]["background"]) for e in ERAS}}
                   if interim else {}))
