"""Project the saved CR-only model into VR; never load observations or fit."""

from collections import defaultdict
from contextlib import ExitStack
import fcntl
import math
import re

import numpy as np

from TROTASR.utils.io import read_json
from TROTASR.utils.era_merge import HIGHDM_Z_SHAPE, highdm_z_shape_formula, highdm_z_shape_group
from TROTASR.utils.paths import EARLY_ERAS, ERAS, ROOT, internal_path
from TROTASR.utils.sr_binning import NATIVE_EDGES
from TROTASR.utils.statistics_model import build_rz_covariance, rate_parameter, rz_value
from TROTASR.utils.statistics_inputs import UT_BOUNDS, _extras
from TROTASR.utils.vr_binning import HIGH_VR_DEFINITION, N_LOW_BINS, N_VR_BINS
from TROTASR.utils.vr_prediction import N_CELL, N_NB, N_RECOIL
from TROTASR.workflows.build_nominal_grid import CR12_POLICY, cr12_parameter
from TROTASR.workflows.cr_linked_workspace import (
    GNN_BY_GROUP, PHOTON_GROUPS, RECOIL_BY_GROUP, TOP_PT_CRONLY_REVISION, _weights,
)
from TROTASR.workflows.diagnostic_common import grid_card, provisional_cronly_products


NB = ("Nb1", "Nb2plus")
FAMILIES = {"Top": "Top", "WtoLNu": "WJet", "QCD": "QCD", "Zto2Nu": "ZJet",
            "GJ": "PhotonJet", "PhotonJet": "PhotonJet", "DY": "DYJet",
            "VV": "VV", "VV_VVV": "VV"}
BENCHMARK = ("T2tt", "mStop1200_mLSP500")


def _card(path, highdm_z_shape=None):
    rows = [row.split() for row in path.read_text().splitlines()
            if row.strip() and not row.lstrip().startswith("#")]
    bin_rows = [r[1:] for r in rows if r[0] == "bin"]
    processes = [r[1:] for r in rows if r[0] == "process"]
    cols = list(zip(bin_rows[-1], processes[0]))
    if len(cols) != len(set(cols)) or len(cols) != len(processes[0]):
        raise ValueError("Invalid accepted card columns")
    rates, lnns, shapes = {}, defaultdict(dict), defaultdict(set)
    for row in rows:
        if len(row) < 2:
            continue
        name, kind, *args = row
        if kind == "rateParam":
            ch, proc, value, tail = args
            key = ch, proc
            if key not in cols or key in rates:
                raise ValueError(f"Unexpected CR rate binding {key!r}")
            if tail.startswith("["):
                rates[key] = (name, 1., float(value))
            elif "__zShape_" in name:
                parameter = tail.split(",")[0]
                era = proc.rsplit("_", 1)[-1]
                nb = highdm_z_shape_group(parameter)
                match = re.match(r"\(@0\*([0-9.eE+-]+)/\(", value)
                if (ch != "SR_highdm_c1" or nb is None or not proc.startswith(f"ZJet_{nb}_u")
                        or not highdm_z_shape or not match or name != parameter + "__zShape_" + proc):
                    raise ValueError(f"Missing or invalid high-dM Z shape binding {name}")
                scale = float(match[1])
                expression, arguments = highdm_z_shape_formula(parameter, scale, highdm_z_shape[era][nb])
                if scale <= 0. or value != expression or tail != ",".join(arguments):
                    raise ValueError(f"High-dM Z shape expression disagrees with the grid: {name}")
                rates[key] = (parameter, scale, None)
            else:
                match = re.fullmatch(r"\(@0\*([0-9.eE+-]+)\)", value)
                if not match or "__eraScale_" not in name:
                    raise ValueError(f"Unsupported rate expression {name}")
                rates[key] = (tail, float(match[1]), None)
            if (ch == "SR_highdm_c1" and highdm_z_shape and highdm_z_shape_group(rates[key][0])
                    and "__zShape_" not in name):
                raise ValueError(f"Missing shape normalization on SR transfer {name}")
        if kind not in ("shape", "lnN"):
            continue
        if len(args) != len(cols):
            raise ValueError(f"Malformed nuisance row: {name}")
        for key, value in zip(cols, args):
            if value == "-" or key[1].startswith("signal_"):
                continue
            if kind == "shape":
                if float(value) != 1.:
                    raise ValueError(f"Non-unit template coefficient: {name}")
                shapes[key].add(name)
            elif "/" in value:
                down, up = map(float, value.split("/"))
                lnns[key][name] = (down, up)
            else:
                up = float(value)
                lnns[key][name] = (1. / up, up)
    return set(cols), rates, lnns, shapes


def _get_measurements(grid):
    records, identities = {}, {}
    for era in ERAS:
        records[era], identities[era] = {}, {}
        for name in ("rz_high", "rz_low", "double_ratio"):
            # Use the retained measurement itself, not its retired execution path.
            p = internal_path(ROOT / "results" / "measurements" / era / (name + ".json.gz"))
            records[era][name] = read_json(p)
            identities[era][name] = str(p.relative_to(ROOT))
        records[era]["rz_covariance"] = build_rz_covariance(
            records[era]["rz_high"], records[era]["rz_low"], era)
    return records, identities


def _cells(record):
    cells = {int(k) for k in record["nominal"]}
    for pairs in record["variations"].values():
        for values in pairs.values():
            cells.update(int(k) for k in values)
    return cells


def _binding(era, family, mode, nb, recoil, rates, control_rates, transfer_columns=None):
    channel = f"SR_{mode}_c1"
    native = recoil if mode == "highdm" else min(recoil, 4)
    suffix = f"_u{native}" if (mode == "highdm" or family == "ZJet") else ""
    if family in ("Top", "WJet", "QCD", "ZJet"):
        proc = f"{family}_{nb}{suffix}_{era}"
    else:
        proc = f"{family}_{era}"
    key = channel, proc
    if family == "ZJet" and mode == "highdm" and recoil >= 4:
        mapped = (transfer_columns or {}).get((era, nb, recoil))
        if mapped is not None:
            key = mapped
        else:
            # Do not mistake the unsuffixed u4 label for the 500–650 bin.
            # A new VR-only process, if bound, is independent for morphing.
            key = channel, f"{family}_{nb}_native{recoil}_{era}"
    if key in rates or family not in ("Top", "WJet", "QCD", "ZJet"):
        return key, rates.get(key)
    # An absent or excluded SR cell is not an exclusion in the VR. Resolve
    # the identical shared normalization from the native CR process instead.
    if family == "ZJet":
        control = f"GCR_{mode}_c1", f"PhotonJet_{nb}_u{min(recoil, 4)}_{era}"
        expected = rate_parameter("sgamma_shape", mode, nb, min(recoil, 4), era)
    else:
        group = next(i for i, members in enumerate(CR12_POLICY["native_recoil_groups"])
                     if recoil in members) if mode == "highdm" else None
        cr, kind = ("QCDCR", "qcd_norm") if family == "QCD" else ("LLCR", "ll_norm")
        control = (f"{cr}_{mode}_c1",
                   f"{family}_{nb}{f'_u{group}' if group is not None else ''}_{era}")
        expected = rate_parameter(kind, mode, nb, recoil if group is not None else "inclusive", era)
        expected = cr12_parameter(expected)
    value = control_rates.get(control)
    if value is not None and value[0] == expected.rsplit("_", 1)[0]:
        return key, value
    raise ValueError(f"No approved VR normalization binding for {key!r}")


def _high_z_transfer_columns(shapes):
    """Resolve packed card process names using the actual closure interval."""
    result = {}
    for (channel, process), terms in shapes.items():
        match = re.fullmatch(r"ZJet_(Nb1|Nb2plus)_u4(?:_part\d+)?_(.+)", process)
        if channel != "SR_highdm_c1" or not match or match[2] not in ERAS:
            continue
        nb, era = match.groups()
        hits = [i for i in range(4, 8) if
                f"CMS_NPS26012_zgammaNonclosure_highdm_u{int(NATIVE_EDGES[i])}to"
                f"{int(NATIVE_EDGES[i+1])}_{era}" in terms]
        if len(hits) > 1 or any((era, nb, i) in result for i in hits):
            raise ValueError(f"Ambiguous native recoil binding for {process}")
        for i in hits:
            result[era, nb, i] = channel, process
    return result


def _projection_inputs(payload, cols, rates, lnns, shapes, control_rates, measurements):
    blocks, extra_shapes, factors = {}, {}, {}
    transfer_columns = _high_z_transfer_columns(shapes)
    for era, samples in payload["era_samples"].items():
        for sample, record in samples.items():
            family = FAMILIES[sample]
            key = era, sample
            blocks[key], extra_shapes[key], factors[key] = {}, defaultdict(
                lambda: {"down": {}, "up": {}}), defaultdict(list)
            for cell in sorted(_cells(record)):
                vr_bin, rem = divmod(cell, N_NB * N_RECOIL)
                nb, recoil = divmod(rem, N_RECOIL)
                mode = "lowdm" if vr_bin < N_LOW_BINS else "highdm"
                column, rate = _binding(era, family, mode, NB[nb], recoil,
                                        rates, control_rates, transfer_columns)
                blocks[key][cell] = "/".join(column)
                # No SR yields or endpoints enter the extrapolation. An
                # absent SR process may inherit only demonstrably uniform
                # multiplicative terms from other columns of its family.
                uniform = lnns.get(column, {})
                if column not in cols:
                    analogues = [lnns.get(c, {}) for c in cols if c[0] == column[0]
                                 and c[1].split("_", 1)[0] == family and c[1].endswith("_" + era)]
                    if not analogues or any(a != analogues[0] for a in analogues[1:]):
                        raise ValueError(f"Missing uniform nuisance binding for {column!r}")
                    uniform = analogues[0]
                norm = 1.
                projection = None
                if family == "ZJet":
                    records = measurements[era]
                    rz = records["rz_covariance"]
                    group = mode + "_" + NB[nb]
                    norm = rz_value(rz, group)
                    if not math.isfinite(norm) or norm <= 0:
                        raise ValueError("Invalid measured Z/gamma central correction")
                    gamma_bin = min(recoil, 4)
                    interval = (NATIVE_EDGES[recoil:recoil + 2] if mode == "highdm"
                                else UT_BOUNDS[gamma_bin:gamma_bin + 2])
                    transfer = _extras(rz, records["double_ratio"], mode,
                                       NB[nb], gamma_bin, era, interval)
                    for term in transfer:
                        name = term["name"]
                        # A variation with no accepted SR template entry is
                        # allowed here only when the same adopted measurement
                        # exists and the VR bin has independent MC support.
                        for side in ("up", "down"):
                            value = record["nominal"].get(str(cell), (0., 0., 0.))[0]
                            extra_shapes[key][name][side][str(cell)] = float(value) * term[side]
                    if mode == "lowdm":
                        projection = (PHOTON_GROUPS[nb], gamma_bin)
                lnn = tuple(sorted((uniform or {}).items()))
                factors[key][(rate, norm, projection, lnn)].append(cell)
            # Extra shapes are absolute. Outside the affected recoil cells
            # they must equal nominal, not zero.
            for sides in extra_shapes[key].values():
                for side in ("up", "down"):
                    for c, value in record["nominal"].items():
                        sides[side].setdefault(c, float(value[0]))
    return blocks, extra_shapes, factors


def _factor_evaluator(specs, photon, highdm_z_shape=None):
    from TROTASR.utils.vr_prediction import asym_pow, photon_recoil_factors

    grouped = {key: [(entry, np.asarray(cells, dtype=np.intp))
                     for entry, cells in v.items()] for key, v in specs.items()}
    last = None
    projections = {}
    normalizers = {}

    def evaluate(era, sample, parameters):
        nonlocal last
        out = np.ones(N_CELL)
        if parameters is not last:
            last = parameters
            for nb, weights in photon.items():
                theta = [parameters[name] for name in GNN_BY_GROUP[nb]]
                for e, w in weights.items():
                    projections[e, nb] = photon_recoil_factors(w, theta)
            for e, groups in (highdm_z_shape or {}).items():
                for nb, coefficients in groups.items():
                    total = math.fsum(parameters[name] * c for name, c in coefficients.items())
                    if not math.isfinite(total) or total <= 0.:
                        raise ValueError(f"Invalid high-dM Z shape normalization for {e}/{nb}")
                    normalizers[e, nb] = total
        for (rate, scale, projection, lnns), cells in grouped[era, sample]:
            result = scale
            if rate is not None:
                name, relative, _ = rate
                result *= parameters[name] * relative
                nb = highdm_z_shape_group(name)
                if highdm_z_shape and nb is not None:
                    result /= normalizers[era, nb]
            if projection is not None:
                nb, recoil = projection
                if (era, nb) in projections:
                    result *= projections[era, nb][recoil]
            for name, (down, up) in lnns:
                result *= float(asym_pow(parameters[name], down, up))
            out[cells] = result
        return out

    return evaluate


def fit_product_root(path):
    """Resolve an explicitly selected, separate Run 3 statistical product."""
    path = internal_path(path)
    baseline = ROOT / "stats" / "run3"
    if (baseline not in path.parents or path.relative_to(baseline).parts[0] in
            {"grid", "cronly", "limits", "impacts", "plots", "vr"}):
        raise ValueError("Select a separate Run 3 model product with its own grid and CR fit")
    return path


def completed_cr_fit_inputs(run_root, expected_card=None):
    """Bind the registered blinded CR card to a completed native fit receipt.

    Call while holding a shared CR controller lock. The command outcome and
    fit quality are the scientific checks; no content hashes are required.
    """
    run_root = fit_product_root(run_root)
    grid, result = run_root / "grid", run_root / "cronly"
    card, origins = grid_card(grid / "manifest.json", "cronly")
    if card.parent != grid / "datacards" or (expected_card is not None and card != expected_card):
        raise ValueError("Fit requires the selected product's own CR-only card")
    state = read_json(result / "state.json")
    contract = read_json(result / "contract.json")
    if (state.get("status") not in ("fits_complete_pending_plots", "generated_pending_visual_QA")
            or state.get("stage") not in ("fit_complete", "profile_pulls_complete")
            or state.get("pid") is not None or state.get("sr_observed_data_used") is not False
            or state.get("origins") != origins or contract.get("origins") != origins
            or contract.get("options", {}).get("signal_strength") != 0
            or state.get("fit_status") != 0 or state.get("covariance_quality") not in (2, 3)
            or not math.isfinite(float(state.get("edm", math.nan)))):
        raise ValueError("The selected model has no matching completed observed CR-only B-only r0 fit")
    products = {internal_path(ROOT / name) for name in state.get("products", ())}
    workspace, pulls = result / "workspace.root", result / "pulls.json"
    if not {workspace, pulls} <= products:
        raise ValueError("The completed CR receipt must register its workspace and pulls")
    fit_result = result / "fitDiagnostics_cronly.root"
    receipt_path = result / "state.json"
    if state.get("recovery_state"):
        receipt_path = internal_path(ROOT / state["recovery_state"])
        if result not in receipt_path.parents or receipt_path.name != "state.json":
            raise ValueError("CR recovery receipt belongs to a different fit")
        recovery = read_json(receipt_path)
        accepted = [row for row in recovery.get("attempts", ())
                    if row.get("passed") is True and row.get("exit_code") == 0]
        fields = ("fit_status", "covariance_quality", "edm")
        if (recovery.get("status") != "fits_complete_pending_plots" or recovery.get("pid") is not None
                or len(accepted) != 1
                or any(accepted[0].get(k) != state[k] for k in fields)):
            raise ValueError("No matching successful CR recovery attempt")
        fit_result = internal_path(ROOT / accepted[0]["fit"])
        if receipt_path.parent not in fit_result.parents:
            raise ValueError("CR recovery result belongs to a different attempt")
    if (fit_result not in products
            or any(not f.is_file() or not f.stat().st_size for f in (workspace, pulls, fit_result))):
        raise ValueError("The accepted CR fit products are missing")
    payload = read_json(pulls)
    if (payload.get("sr_observed_data_used") is not False
            or internal_path(payload["source_card"]) != card
            or set(payload.get("channels", ())) != set(state.get("channels", ()))):
        raise ValueError("CR fit payload is not bound to the selected blinded card")
    publication = dict(status="completed_fit_receipt", canonical_model="era_merged", kind="cronly",
                       dataset="observed CR-only B-only r0", sr_observed_data_used=False,
                       complete=True, process_exit_code=0, termination="normal",
                       fit_status=state["fit_status"], covariance_quality=state["covariance_quality"],
                       edm=state["edm"], manifest=origins["manifest"], card=origins["card"],
                       model_revision=state.get("model_revision"),
                       receipt=str(receipt_path.relative_to(ROOT)))
    return card, dict(workspace=workspace, fit_result=fit_result), publication


def _load_fit(run_root, needed, rate_names, gaussian_names, *, explicit_card=None):
    import ROOT as root

    root.gROOT.SetBatch(True)
    root.gSystem.Load("libHiggsAnalysisCombinedLimit")
    root.RooMsgService.instance().setGlobalKillBelow(root.RooFit.ERROR)
    r = run_root / "cronly"
    def ident(p):
        stat = p.stat()
        return dict(file=p.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    with ExitStack() as stack:
        lock = stack.enter_context((r / "controller.lock").open("rb"))
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if explicit_card is not None:
            _, products, pub = completed_cr_fit_inputs(run_root, explicit_card)
        else:
            pub = read_json(r / "publication.json")
            products = provisional_cronly_products(r, pub, required=("workspace", "fit_result"))
            if products is None:
                if (pub.get("kind") != "cronly" or pub.get("complete") is not True
                        or pub.get("dataset") != "observed CR-only B-only r0"
                        or pub.get("sr_observed_data_used") is not False):
                    raise ValueError("Expected the accepted observed CR-only fit")
                products = {key: internal_path(r / pub.get(key, default)) for key, default in (
                    ("workspace", "workspace.root"), ("fit_result", "fitDiagnostics_cronly.root"))}
                if any(path.parent != r for path in products.values()):
                    raise ValueError("CR-only fit inputs must belong to the accepted result directory")
        files = []
        for key in ("workspace", "fit_result"):
            f = root.TFile.Open(str(products[key]), "READ")
            if not f or f.IsZombie() or f.TestBit(root.TFile.kRecovered):
                raise ValueError("Cannot read the published CR-only fit inputs")
            stack.callback(f.Close)
            files.append(f)
        w, result = files[0].Get("w"), files[1].Get("fit_b")
        if (not w or not result or int(result.status()) != 0
                or int(result.covQual()) < 2 or not math.isfinite(float(result.edm()))
                or w.var("r") is None or w.cat("CMS_channel") is None):
            raise ValueError("Invalid saved CR-only fit_b")
        if explicit_card is not None and (int(result.status()), int(result.covQual()),
                float(result.edm())) != (pub["fit_status"], pub["covariance_quality"], pub["edm"]):
            raise ValueError("Saved CR fit_b does not match its fit-quality receipt")
        channels = {str(p.first) for p in w.cat("CMS_channel")}
        if channels != {f"{c}_{m}_c1" for c in ("LLCR", "QCDCR", "GCR")
                        for m in ("lowdm", "highdm")}:
            raise ValueError("Only the six CRs are allowed in the donor workspace")
        w.var("r").setVal(0.)
        groups = PHOTON_GROUPS if pub.get("model_revision") == TOP_PT_CRONLY_REVISION else ("Nb1",)
        gnn = {name for nb in groups for name in GNN_BY_GROUP[nb]}
        fixed_legacy = {name for nb in groups for name in RECOIL_BY_GROUP[nb]}
        needed = needed | gnn
        photon = {
            nb: {era: np.asarray(values, dtype=float) for era, values in _weights(w, nb).items()}
            for nb in groups
        }
        if any(a.shape != (5, 5) or np.any(a < 0) or np.any(a.sum(1) <= 0)
               or not np.isfinite(a).all() for weights in photon.values() for a in weights.values()):
            raise ValueError("Invalid prefit photon weights for GNN-to-recoil projection")
        floats = {str(x.GetName()): float(x.getVal()) for x in result.floatParsFinal()}
        fixed_r = result.constPars().find("r")
        if "r" in floats or (fixed_r and float(fixed_r.getVal()) != 0):
            raise ValueError("Expected the saved B-only fit with r fixed to zero")
        if (not all(math.isfinite(value) for value in floats.values())
                or not gnn.issubset(floats) or fixed_legacy & floats.keys()):
            raise ValueError("The saved fit does not use the accepted Photon CR GNN model")
        bounded, prefit, constrained = {}, {}, {}
        for name in sorted(needed):
            v = w.var(name)
            if v:
                prefit[name] = float(v.getVal())
                bounded[name] = float(v.getMin()), float(v.getMax())
                if name in rate_names or name in gnn:
                    if name in fixed_legacy and not v.isConstant():
                        raise ValueError(f"Pre-GNN recoil coefficient is not fixed: {name}")
                    if name not in fixed_legacy and name not in floats:
                        raise ValueError(f"Missing fitted CR normalization {name}")
                else:
                    error = float(v.getError())
                    if name not in gaussian_names or error <= 0:
                        raise ValueError(f"Unknown prior for {name}")
                    constrained[name] = error
            else:
                if name in rate_names or name in gnn or name in floats or name not in gaussian_names:
                    raise ValueError(f"Missing model parameter {name}")
                # The benchmark card defines an independent Gaussian prior;
                # it stays prefit if that background effect is absent in CR.
                prefit[name], constrained[name] = 0., 1.
        postfit = dict(prefit, **{k: v for k, v in floats.items() if k in needed})
        floating = set(floats) & needed
        rng = np.random.default_rng(12345)
        n_draws = 500
        root.RooRandom.randomGenerator().SetSeed(12345)
        randomized = list(result.randomizePars())
        draws, clipped = [], 0
        for _ in range(n_draws):
            current = dict(postfit)
            result.randomizePars()
            for p in randomized:
                name = str(p.GetName())
                if name not in floating:
                    continue
                low, high = bounded[name]
                x = float(p.getVal())
                if not math.isfinite(x):
                    raise ValueError(f"Nonfinite covariance draw for {name}")
                current[name] = min(high, max(low, x))
                clipped += current[name] != x
            for name in sorted(constrained.keys() - floating):
                x = prefit[name] + constrained[name] * rng.standard_normal()
                low, high = bounded.get(name, (-math.inf, math.inf))
                current[name] = min(high, max(low, x))
            draws.append(current)
        prior = {}
        for name, sigma in constrained.items():
            low, high = bounded.get(name, (-math.inf, math.inf))
            prior[name] = (dict(prefit, **{name: max(low, prefit[name] - sigma)}),
                           dict(prefit, **{name: min(high, prefit[name] + sigma)}))
        quality = dict(fit_status=int(result.status()), covariance_quality=int(result.covQual()),
                       edm=float(result.edm()), saved_process_exit=pub.get("process_exit_code"),
                       termination=pub.get("termination"), publication=pub.get("status"),
                       fit_recalculated=False, cr_only=True, draw_count=n_draws,
                       draw_seed=12345, bounded_values=clipped,
                       constrained_from_fit=sorted(floating),
                       independent_unconstrained_in_cr=sorted(constrained.keys() - floating),
                       identity={key: ident(products[key]) for key in ("workspace", "fit_result")})
        if explicit_card is not None:
            quality.update(cr_manifest=pub["manifest"], cr_card=pub["card"], fit_receipt=pub["receipt"])
    return prefit, postfit, prior, draws, photon, quality


def project(run_root, response):
    """Compute the two predictions before the renderer receives JetMET counts."""
    from TROTASR.utils.vr_prediction import VRResponse, propagate, validate_response_axes

    validate_response_axes(response)
    if (response.get("status") != "complete"
            or response.get("all_applicable_responses_available") is not True
            or any(response.get("unsupported", {}).values())
            or response.get("mc_only") is not True
            or response.get("sr_observations_read") is not False):
        raise ValueError("Cannot project an incomplete or unsupported VR background response")
    run_root = internal_path(run_root)
    grid = run_root / "grid"
    manifest = read_json(grid / "manifest.json")
    if (manifest.get("sr_data_blinded") is not True or manifest.get("era_merged") is not True
            or tuple(manifest.get("years", ())) != ERAS):
        raise ValueError("Expected the adopted blinded six-era grid")
    point = next(p for p in manifest["points"] if (p["model"], p["mass"]) == BENCHMARK)
    card = internal_path(ROOT / point["card"])
    if card.parent != grid / "datacards" or not card.is_file():
        raise ValueError("Background template card is missing from the grid")
    highdm_z_shape = manifest.get(HIGHDM_Z_SHAPE)
    cols, rates, lnns, shapes = _card(card, highdm_z_shape)
    cr_spec = manifest["cr_only_card"]
    cr_card = internal_path(ROOT / cr_spec["card"])
    if cr_card.parent != grid / "datacards" or not cr_card.is_file():
        raise ValueError("CR card is missing from the grid")
    _, control_rates, _, _ = _card(cr_card)
    measurements, hashes = _get_measurements(grid)
    blocks, extra, specs = _projection_inputs(response, cols, rates, lnns, shapes,
                                              control_rates, measurements)
    needed = set()
    needed.update(name for era in (highdm_z_shape or {}).values()
                  for coefficients in era.values() for name in coefficients)
    for key, groups in specs.items():
        for rate, _, _, norms in groups:
            if rate is not None:
                needed.add(rate[0])
            needed.update(name for name, _ in norms)
        era, sample = key
        needed.update(response["era_samples"][era][sample]["variations"])
        needed.update(extra[key])
    rate_names = {row[0] for row in (*rates.values(), *control_rates.values())}
    gaussian_names = (set().union(*shapes.values(), *lnns.values()) |
                      set().union(*(set(v) for v in extra.values())))
    prefit, postfit, prior, draws, photon, fit = _load_fit(run_root, needed, rate_names, gaussian_names)
    evaluate_factors = _factor_evaluator(specs, photon, highdm_z_shape)
    kernel = VRResponse.from_payload(response, blocks, extra_shapes=extra)
    views = {"highdm": ERAS, "lowdm_early": EARLY_ERAS, "lowdm_late": ERAS[-2:]}
    indexes = {name: np.arange(N_LOW_BINS, N_VR_BINS) if name == "highdm" else np.arange(N_LOW_BINS)
               for name in views}
    selections = {}
    for name, eras in views.items():
        mask = np.zeros(N_VR_BINS)
        mask[indexes[name]] = 1.
        selections[name] = {era: mask for era in eras}
    central_prefit = kernel.evaluate(prefit, factors=evaluate_factors)
    evaluated_postfit = propagate(kernel, postfit, draws, views=selections, factors=evaluate_factors)

    def join(evaluated):
        return np.concatenate([evaluated.view(selections[name])["total"][indexes[name]]
                               for name in views])

    joint_size = sum(len(index) for index in indexes.values())
    prior_cov = np.zeros((joint_size, joint_size))
    for down, up in prior.values():
        delta = (join(kernel.evaluate(up, factors=evaluate_factors))
                 - join(kernel.evaluate(down, factors=evaluate_factors))) / 2.
        prior_cov += np.outer(delta, delta)

    def stage(value, index, systematic):
        groups = defaultdict(lambda: np.zeros(len(index)))
        for (_, sample), y in value["by_era_sample"].items():
            family = FAMILIES[sample]
            groups[family] += y[index]
        variance = np.maximum(value["mc_variance"][index], 0.)
        return dict(groups={k: y.tolist() for k, y in groups.items()},
                    background=value["total"][index].tolist(),
                    background_stat_unc=np.sqrt(variance).tolist(),
                    background_unc=np.sqrt(np.maximum(np.diag(systematic), 0.) + variance).tolist())

    result = {}
    offset = 0
    joint_index = []
    for i, (name, eras) in enumerate(views.items()):
        index = indexes[name]
        where = slice(offset, offset + len(index))
        offset = where.stop
        joint_index.extend(i * N_VR_BINS + index)
        post = evaluated_postfit["views"][name]
        pre = central_prefit.view(selections[name])
        result[name] = dict(years=list(eras),
            raw_nominal_mc=pre["raw_nominal"][index].tolist(),
            prefit=stage(pre, index, prior_cov[where, where]),
            postfit=stage(post, index, post["systematic_covariance"][np.ix_(index, index)]))
    pre_stat = np.concatenate([v["prefit"]["background_stat_unc"] for v in result.values()])
    return dict(schema="trotasr_vr_prefit_postfit_v1", status="complete", years=list(ERAS),
                high_definition=HIGH_VR_DEFINITION,
                response_scope=response.get("endpoint_scope", response["object_method"]),
                fit=fit, views=result, vr_data_fitted=False, fit_recalculated=False,
                sr_observations_used=False, sr_observations_included=False,
                covariance=dict(view_order=list(views),
                    prefit=(prior_cov + np.diag(pre_stat**2)).tolist(),
                    postfit=evaluated_postfit["joint_covariance"][np.ix_(joint_index, joint_index)].tolist(),
                    prefit_shape_prior="symmetric independent prior shifts; rate bounds are not uncertainties",
                    postfit_method="shared saved correlated CR-only draws plus unconstrained background prior",
                    mc_stat="independent VR nominal sumw2, with no CR autoMCStats transfer"),
                native_clipping=dict(prefit=list(central_prefit.clipping),
                    postfit=list(evaluated_postfit["central"].clipping),
                    postfit_draws=evaluated_postfit["draws_with_native_clipping"]),
                source=dict(card=point["card"], measurements=hashes,
                            cr_card=cr_spec["card"],
                            photon_prefit_weights={k: a.tolist() for k, a in photon["Nb1"].items()},
                            **({"photon_prefit_weights_nb2plus": {
                                k: a.tolist() for k, a in photon["Nb2plus"].items()
                            }} if "Nb2plus" in photon else {})))
