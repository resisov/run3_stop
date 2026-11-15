"""VR comparison with saved CR normalization only; no event inputs."""

import math
from collections import defaultdict

import numpy as np

from TROTASR.utils.paths import EARLY_ERAS, ERAS, ROOT
from TROTASR.utils.sr_binning import NATIVE_EDGES

VIEWS = {"highdm": ERAS, "lowdm_early": EARLY_ERAS, "lowdm_late": ERAS[-2:]}
CONTROLLED = ("Top", "WtoLNu", "QCD", "Zto2Nu")
OTHER = ("GJ", "DY", "VV")
CR12 = (250., 300., 350., 400., 500., 800., math.inf)
GAMMA = (250., 300., 350., 400., 500., math.inf)


def _cells(mc):
    bins = [("lowdm", "Nb1" if i < 15 else "Nb2plus", None, None) for i in range(30)]
    for category in mc["high_categories"]:
        predicate = category["predicates"]["Nb"]
        if set(predicate) not in ({"eq"}, {"ge"}):
            raise ValueError("Unexpected Nb predicate")
        nb = ("Nb1" if predicate == {"eq": 1} else None if predicate == {"ge": 1}
              else "Nb2plus" if next(iter(predicate.values())) >= 2 else None)
        edges = category["edges"]
        bins.extend(("highdm", nb, lo, math.inf if hi == NATIVE_EDGES[-1] else hi)
                    for lo, hi in zip(edges, edges[1:]))
    if len(bins) != 148:
        raise ValueError("Require the same 148 VR bins")
    return bins


def _rate(geom, process):
    mode, nb, low, high = geom
    if nb is None:
        return None, "mixed_Nb"
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


def saved_rates():
    """Export only saved normalization coordinates and their correlated draws."""
    from TROTASR.utils.io import read_json
    from TROTASR.workflows.cr_linked_workspace import RECOIL
    from TROTASR.workflows.vr_fit_projection import _card, _load_fit

    run_root = ROOT / "stats" / "run3"
    grid = run_root / "grid"
    manifest = read_json(grid / "manifest.json")
    if (manifest.get("sr_data_blinded") is not True or manifest.get("era_merged") is not True
            or tuple(manifest["years"]) != ERAS):
        raise ValueError("Expected the accepted blinded six-era grid")
    spec = manifest["cr_only_card"]
    path = ROOT / spec["card"]
    if path.parent != grid / "datacards":
        raise ValueError("Use the registered CR-only card directory")
    _, rates, _, _ = _card(path)
    bindings = {}
    for (channel, proc), (name, era_scale, _) in rates.items():
        if not channel.startswith(("LLCR_", "QCDCR_", "GCR_")) or name in RECOIL:
            continue
        era = proc.rsplit("_", 1)[-1]
        family = proc.split("_", 1)[0]
        if era not in ERAS or family not in ("Top", "WJet", "QCD", "PhotonJet"):
            raise ValueError("Unrecognized CR normalization binding")
        # Shared alpha times the same eraScale in prefit and postfit. The
        # multiplier is recorded for evidence and cancels in their ratio.
        key = f"{era}/{family}/{name}"
        if key in bindings or not math.isfinite(era_scale) or era_scale <= 0:
            raise ValueError(f"Ambiguous CR binding: {key}")
        bindings[key] = era_scale
    parameters = {k.rsplit("/", 1)[-1] for k in bindings}
    pre, post, _, draws, _, fit = _load_fit(run_root, parameters, parameters, set())
    if set(pre) != parameters or set(post) != parameters or len(draws) < 2:
        raise ValueError("Missing saved normalization coordinates")
    return dict(prefit=pre, postfit=post, draws=draws, fit=fit, bindings=bindings,
                card=spec["card"])


def project(mc, rates):
    """Apply only identifiable ratios to the existing prefit bin contents."""
    from TROTASR.workflows.plotting_vr_predictions import SIGNAL, _covered

    if (mc.get("fields") != ["sumw", "sumw2", "entries"]
            or mc.get("status") not in ("complete", "incomplete") or mc.get("mc_only") is not True
            or mc.get("pilot") is True
            or mc.get("fit_or_transfer_factors_applied") is not False
            or mc.get("sr_observations_read") is not False
            or mc.get("vr_observations_read") is not False
            or set(mc.get("coverage", {})) != set(ERAS)
            or any(not _covered(mc["coverage"][era].get("background")) for era in ERAS)
            or set(mc["era_samples"]) != set(ERAS)):
        raise ValueError("Require the completed, unchanged VR MC aggregate")
    geometry = _cells(mc)
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
    result, matrices, mcvars = {}, [], []
    for view, years in VIEWS.items():
        bins = list(range(30, 148) if view == "highdm" else range(30))
        pre_y, post_y = defaultdict(lambda: np.zeros(len(bins))), defaultdict(lambda: np.zeros(len(bins)))
        pre_v, post_v = np.zeros(len(bins)), np.zeros(len(bins))
        terms = np.zeros((len(bins), len(names)))
        unmapped, mapped = defaultdict(list), defaultdict(list)
        for era in years:
            for process, rows in mc["era_samples"][era].items():
                if SIGNAL.fullmatch(process):
                    continue
                if process not in (*CONTROLLED, *OTHER):
                    raise ValueError(f"Unrecognized background: {process}")
                a = np.asarray(rows, float)
                if a.shape != (148, 3) or not np.isfinite(a).all() or np.any(a[:, 1:] < 0):
                    raise ValueError(f"Invalid aggregate for {era}/{process}")
                for i, b in enumerate(bins):
                    y, var, entries = a[b]
                    factor = 1.
                    if process in CONTROLLED and (y != 0 or var != 0 or entries != 0):
                        name, reason = _rate(geometry[b], process)
                        family = {"WtoLNu": "WJet", "Zto2Nu": "PhotonJet"}.get(process, process)
                        key = f"{era}/{family}/{name}"
                        if name not in index or key not in rates["bindings"]:
                            unmapped[i + 1].append(dict(era=era, process=process,
                                reason=reason or "no_saved_CR_binding", nominal_yield=float(y),
                                ratio=None))
                        else:
                            j = index[name]
                            factor = ratios[j]
                            terms[i, j] += y
                            mapped[i + 1].append(dict(era=era, process=process,
                                parameter=name, factor=float(factor)))
                    pre_y[process][i] += y
                    post_y[process][i] += y * factor
                    pre_v[i] += var
                    post_v[i] += var * factor**2
        shifted = draw_ratios @ terms.T
        delta = shifted - shifted.mean(axis=0)
        cov = delta.T @ delta / len(draws)
        matrices.append(terms)
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
    shifts = draw_ratios @ np.vstack(matrices).T
    delta = shifts - shifts.mean(axis=0)
    return dict(schema="trotasr_vr_prefit_postfit_v1", status="complete_normalization_only",
                normalization_only=True, years=list(ERAS), views=result,
                scope="exact saved CR normalization ratios only; unbound components unchanged",
                prefit="existing nominal VR MC; no new transfer factors",
                excluded_from_bands="VR shape-response systematics and unmapped CR effects",
                source=dict(cr_card=rates.get("card")), fit=rates["fit"],
                covariance=dict(view_order=list(VIEWS),
                    cr_normalization=(delta.T @ delta / len(draws)).tolist(),
                    independent_mc_variance=np.concatenate(mcvars).tolist()),
                fit_recalculated=False, vr_data_fitted=False, sr_observations_used=False)
