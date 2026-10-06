"""Public correction API. No chdir, campaign paths, or data-directory copying."""
from __future__ import annotations
import contextlib
from functools import lru_cache
from pathlib import Path
import numpy as np
from .paths import ROOT
from .io import read_json, sha256
from .private_scales import (topw_file_input_policy, topw_file_sf_triplet,
    topw_pass_fail_triplet, topw_event_variations, TopWEvents)


@contextlib.contextmanager
def analysis_workdir(repo):
    # Compatibility with the extracted mathematical evaluator, deliberately no
    # cwd mutation or file staging. Every payload path is now absolute.
    if Path(repo).resolve() != ROOT.resolve():
        raise ValueError("Corrections must use this TROTASR payload root")
    yield


@lru_cache(maxsize=1)
def load_analysis_corrections(repo=ROOT):
    from coffea.util import load
    manifest = read_json(ROOT / "utils/compiled/manifest.json")
    for rel, digest in manifest["sources"].items():
        if sha256(ROOT / rel) != digest:
            raise ValueError("Stale compiled corrections/IDs; compile_definitions.py required: " + rel)
    p = ROOT / "utils/compiled/corrections.coffea"
    if sha256(p) != manifest["artifacts"]["corrections.coffea"]:
        raise ValueError("Compiled correction hash mismatch")
    return load(str(p))


def event_weights(arrays, dataset, process, year, region_masks, topw_policy, cleaned=None, apply_topw=True,
                  include_variations=True):
    """Strict nominal/weight-systematic triplets; no raw-genweight fallback."""
    from .weight_inputs import flat_arrays_for_weights, signal_btag_efficiency_dataset
    from .weight_components import compute_weight_bundle
    raw, values = flat_arrays_for_weights(dict(arrays))
    values.pop("electron_eta_source")
    values["gcr_mask"] = region_masks["GCR"]
    values["met_trigger_mask"] = region_masks["SR"] | region_masks["LLCR"] | region_masks["QCDCR"]
    correction_dataset = dataset
    if np.any(np.asarray(arrays["is_signal"], dtype=bool)):
        masses = np.unique(np.asarray(arrays["mStop"], dtype=int))
        if len(masses) != 1:
            raise ValueError("Evaluate signal weights per mass point")
        _, correction_dataset = signal_btag_efficiency_dataset(int(masses[0]), dataset)
    _, variations, audit = compute_weight_bundle(raw, ROOT, correction_dataset, process, str(year),
                                                **values, analysis_sf_components=("met_trigger", "photon_trigger"),
                                                include_variations=include_variations)
    if np.all(np.asarray(arrays["is_data"], dtype=bool)):
        return variations, audit
    required = {"pileup", "btagSF", "electron_id", "electron_reco", "electron_hlt", "muon_id",
                "muon_iso", "muon_hlt", "photon_id", "photon_csev", "met_trigger", "photon_trigger"}
    allowed = {"electron_hlt": "2025 electron HLT SF is not available"} if year == 2025 else {}
    missing = []
    for name in sorted(required):
        state = audit["components"].get(name, {})
        if not state.get("applied"):
            if name in allowed and allowed[name] in state.get("error", ""):
                audit.setdefault("known_unavailable", {})[name] = state["error"]
            else:
                missing.append(name + ": " + str(state))
    if missing:
        raise ValueError("Correction failure: " + "; ".join(missing))
    if not apply_topw:
        return variations, audit
    topw, topw_audit = topw_event_variations(ROOT, str(year), dataset, process, arrays, topw_policy,
                                          cleaned=cleaned, include_variations=include_variations)
    audit["topw"] = topw_audit
    base = variations["nominal"].copy()
    variations = {name: weight * topw["nominal"] for name, weight in variations.items()}
    variations.update({name: base * value for name, value in topw.items() if name != "nominal"})
    if any(not np.isfinite(value).all() for value in variations.values()):
        raise ValueError("Nonfinite corrected event weights")
    return variations, audit


# Existing 2024/2025 object-calibration mathematics, internalized from the
# frozen retained-event implementation. Payloads are supplied only by the
# hash-verified local resolver below; no external analysis import is used.
import hashlib
from typing import Any
import awkward as ak
from scipy.special import erf, erfinv

def _flatten(values: Any) -> tuple[np.ndarray, np.ndarray]:
    counts = ak.to_numpy(ak.num(values, axis=1))
    return ak.to_numpy(ak.flatten(values, axis=1)), counts


def _broadcast_flat(event_values: Any, like: Any) -> np.ndarray:
    return ak.to_numpy(ak.flatten(ak.broadcast_arrays(like, event_values)[1], axis=1))


def _evaluate_jagged(correction: Any, like: Any, *inputs: Any) -> Any:
    _, counts = _flatten(like)
    flat_inputs = []
    for value in inputs:
        if isinstance(value, (str, bytes)) or np.isscalar(value):
            flat_inputs.append(value)
            continue
        try:
            is_jagged = ak.ndim(value) > 1
        except Exception:
            is_jagged = False
        flat_inputs.append(_flatten(value)[0] if is_jagged else _broadcast_flat(value, like))
    if int(np.sum(counts)) == 0:
        return ak.unflatten(np.asarray([], dtype=float), counts)
    return ak.unflatten(np.asarray(correction.evaluate(*flat_inputs), dtype=float), counts)


def _deterministic_normal(run: int, lumi: int, event: int, prefix: str, index: int) -> float:
    token = f"{run}:{lumi}:{event}:{prefix}:{index}".encode()
    seed = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
    return float(np.random.default_rng(seed).standard_normal())


def _deterministic_normal_jagged(like: Any, run: Any, lumi: Any, event: Any, prefix: str) -> Any:
    counts = ak.to_list(ak.num(like, axis=1))
    out = []
    for iev, count in enumerate(counts):
        out.append([
            _deterministic_normal(int(run[iev]), int(lumi[iev]), int(event[iev]), prefix, index)
            for index in range(int(count))
        ])
    return ak.Array(out)


def _valid_scaled_pt(original: Any, candidate: Any, valid: Any) -> Any:
    ratio = candidate / ak.where(original > 0, original, 1.0)
    usable = valid & np.isfinite(candidate) & (candidate > 0) & (ratio >= 0.1) & (ratio <= 2.0)
    return ak.where(usable, candidate, original)


def _calibrate_egm_collection(
    arrays: Any,
    prefix: str,
    is_data: bool,
    shift: str,
    root: Path,
    *,
    correction_set: Any = None,
) -> tuple[Any, Any | None, dict[str, Any]]:
    payload_name = "electronSS_EtDependent.json.gz" if prefix == "Electron" else "photonSS_EtDependent.json.gz"
    if correction_set is None:
        raise ValueError("An internal, hash-verified correction set is required")
    cset = correction_set
    pt = arrays[f"{prefix}_pt"]
    eta_sc = arrays["Electron_eta"] + arrays["Electron_deltaEtaSC"] if prefix == "Electron" else arrays["Photon_eta"]
    r9 = arrays[f"{prefix}_r9"]
    seed_gain = ak.values_astype(arrays[f"{prefix}_seedGain"], np.float64)
    valid = (pt >= 20.0) & (abs(eta_sc) < 2.5) & np.isfinite(pt) & np.isfinite(eta_sc) & np.isfinite(r9)
    safe_pt = ak.where(valid, pt, 25.0)
    safe_eta_sc = ak.where(valid, eta_sc, 0.0)
    safe_r9 = ak.where(valid, r9, 0.95)
    safe_gain = ak.where(valid, seed_gain, 12.0)
    status = {
        "payload": payload_name,
        "prefix": prefix,
        "minimum_pt_gev": 20.0,
        "shift": shift,
    }
    if is_data:
        scale = _evaluate_jagged(
            cset.compound["Scale"], pt, "scale", arrays["run"], safe_eta_sc, safe_r9, safe_pt, safe_gain,
        )
        corrected_pt = _valid_scaled_pt(pt, pt * scale, valid)
        status.update({"mode": "data_scale", "correction": "Scale(scale,run,ScEta,r9,pt,seedGain)"})
    else:
        smear_syst = "smear"
        if shift == f"{prefix.lower()}SmearUp":
            smear_syst = "smear_up"
        elif shift == f"{prefix.lower()}SmearDown":
            smear_syst = "smear_down"
        width = _evaluate_jagged(cset["SmearAndSyst"], pt, smear_syst, safe_pt, safe_r9, safe_eta_sc)
        random_value = _deterministic_normal_jagged(
            pt, arrays["run"], arrays["luminosityBlock"], arrays["event"], prefix,
        )
        corrected_pt = pt * (1.0 + width * random_value)
        scale_syst = None
        if shift == f"{prefix.lower()}ScaleUp":
            scale_syst = "scale_up"
        elif shift == f"{prefix.lower()}ScaleDown":
            scale_syst = "scale_down"
        if scale_syst:
            scale = _evaluate_jagged(cset["SmearAndSyst"], pt, scale_syst, safe_pt, safe_r9, safe_eta_sc)
            corrected_pt = corrected_pt * scale
        corrected_pt = _valid_scaled_pt(pt, corrected_pt, valid)
        status.update({
            "mode": "mc_gaussian_smearing",
            "smearing_variation": smear_syst,
            "scale_variation": scale_syst or "nominal",
            "random_seed": "blake2b(run,lumi,event,object,index)",
        })
    mass = arrays["Electron_mass"] if prefix == "Electron" else None
    corrected_mass = mass * corrected_pt / ak.where(pt > 0, pt, 1.0) if mass is not None else None
    return corrected_pt, corrected_mass, status


NONJME_ENDPOINTS = tuple(name + direction for name in (
    'electronScale', 'electronSmear', 'photonScale', 'photonSmear',
    'muonScale', 'muonResolution', 'tauEnergyScale', 'metUnclustered')
    for direction in ('Up', 'Down'))


def nonjme_payload_contract(year):
    manifest = read_json(ROOT/'scales/object_shapes.json')
    records = manifest['years'][str(year)]
    for record in records.values():
        if sha256(ROOT/record['path']) != record['sha256']:
            raise ValueError('Changed object-shape payload: ' + record['path'])
    return dict(manifest_sha256=sha256(ROOT/'scales/object_shapes.json'),
                payloads={k: v['sha256'] for k, v in records.items()})


@lru_cache(maxsize=8)
def _nonjme_cset(year, flavor):
    import correctionlib
    nonjme_payload_contract(year)
    spec = read_json(ROOT/'scales/object_shapes.json')['years'][str(year)][flavor]
    return correctionlib.CorrectionSet.from_file(str(ROOT/spec['path']))


def nonjme_calibration_inputs(chunk, flavor):
    prefix = flavor.capitalize()
    arrays = {key: chunk[key] for key in ('run', 'luminosityBlock', 'event')}
    for field in ('pt', 'mass', 'eta', 'phi'):
        if flavor == 'photon' and field == 'mass':
            continue
        branch = flavor + ('_nanoaod_' if field in ('pt', 'mass') else '_') + field
        if field in ('eta', 'phi'):
            branch += '_all'
        arrays[prefix+'_'+field] = chunk[branch]
    fields = dict(electron=dict(r9='r9', seedGain='seed_gain'),
                  photon=dict(r9='r9', seedGain='seed_gain'),
                  muon=dict(charge='charge', nTrackerLayers='tracker_layers'),
                  tau=dict(decayMode='decay_mode', genPartFlav='genpart_flavour'))
    for target, source in fields[flavor].items():
        arrays[prefix+'_'+target] = chunk[flavor+'_'+source+'_all']
    if flavor == 'electron':
        arrays['Electron_deltaEtaSC'] = chunk['electron_eta_sc_all']-chunk['electron_eta_all']
    return arrays


def nonjme_calibrate(arrays, year, flavor, shift):
    cset = _nonjme_cset(year, flavor)
    if flavor in ('electron', 'photon'):
        return _calibrate_egm_collection(arrays, flavor.capitalize(), False, shift, ROOT, correction_set=cset)
    function = _calibrate_muons if flavor == 'muon' else _calibrate_taus
    return function(arrays, False, shift, ROOT, correction_set=cset)


def nonjme_event_views(chunks, year, resolved, audit, endpoints=NONJME_ENDPOINTS):
    """Frozen retained-event delta method; no TROTA inference or ROOT writing.

    Unaffected rows retain every stored value, including original precision.
    The nominal view is the original chunk, not a rebuilt central convention.
    """
    from .shape_kinematics import rebuild
    if not endpoints or len(set(endpoints)) != len(endpoints) or set(endpoints)-set(NONJME_ENDPOINTS):
        raise ValueError('Unknown or duplicate non-JME endpoint')
    for original in chunks:
        yield original, resolved, 'nominal'
        if np.any(np.asarray(original['is_data'], bool)):
            if not np.all(np.asarray(original['is_data'], bool)):
                raise ValueError('Mixed data and MC chunk')
            continue
        chunk = {k: original[k] for k in ak.fields(original)}
        cache = {}
        for endpoint in endpoints:
            out = dict(chunk)
            record = audit.setdefault(endpoint, dict(events=0, affected_events=0))
            record['events'] += len(original)
            if endpoint.startswith('metUnclustered'):
                direction = 'up' if endpoint.endswith('Up') else 'down'
                p, f = (np.asarray(chunk[k], float) for k in (
                    'puppi_met_unclustered_'+direction, 'puppi_met_phi_unclustered_'+direction))
                oldp, oldf = np.asarray(chunk['puppi_met_nanoaod']), np.asarray(chunk['puppi_met_nanoaod_phi'])
                x = np.asarray(chunk['met'])*np.cos(chunk['met_phi']) + p*np.cos(f)-oldp*np.cos(oldf)
                y = np.asarray(chunk['met'])*np.sin(chunk['met_phi']) + p*np.sin(f)-oldp*np.sin(oldf)
                affected = (p != oldp) | (f != oldf)
            else:
                flavor = next(n for n in ('electron','photon','muon','tau') if endpoint.startswith(n))
                if flavor not in cache:
                    raw = nonjme_calibration_inputs(chunk, flavor)
                    cache[flavor] = (raw, nonjme_calibrate(raw, year, flavor, 'nominal'))
                raw, nominal = cache[flavor]
                varied = nonjme_calibrate(raw, year, flavor, endpoint)
                for i, field in enumerate(('pt','mass')):
                    if nominal[i] is None:
                        continue
                    stored = chunk[flavor+'_corrected_'+field]
                    value = stored + (varied[i]-nominal[i])
                    if not np.isfinite(np.asarray(ak.flatten(value))).all():
                        raise ValueError('Nonfinite object endpoint: '+endpoint)
                    # Delta response is anchored to the adopted stored central.
                    # No central replacement or nominal pass/fail tolerance gate.
                    out[flavor+'_corrected_'+field] = value
                    out[flavor+'_'+field+'_all'] = value
                delta = out[flavor+'_corrected_pt']-chunk[flavor+'_corrected_pt']
                angle = chunk[flavor+'_phi_all']
                x = chunk['met']*np.cos(chunk['met_phi'])-ak.sum(delta*np.cos(angle), axis=1)
                y = chunk['met']*np.sin(chunk['met_phi'])-ak.sum(delta*np.sin(angle), axis=1)
                affected = np.asarray(ak.any(delta != 0, axis=1))
            record['affected_events'] += int(np.sum(affected))
            if not np.any(affected):
                yield original, resolved, endpoint
                continue
            out['met'], out['met_phi'] = np.hypot(x,y), np.arctan2(y,x)
            out['puppi_met_corrected'], out['puppi_met_corrected_phi'] = out['met'], out['met_phi']
            view = rebuild(out, objects_only=True)
            # Same changed-row selection and storage precision as frozen run.py,
            # performed only in memory (no shifted ROOT/event sidecar).
            rows = np.arange(len(original))+np.where(affected, 0, len(original))
            # Replacing each field on an Awkward record reconstructs its full
            # layout repeatedly. Assemble exactly the same columns once.
            columns = {}
            for name in ak.fields(view):
                if name not in chunk:
                    columns[name] = view[name]
                    continue
                value = ak.concatenate([view[name], original[name]], axis=0)[rows]
                dtype = np.asarray(ak.flatten(original[name], axis=None)).dtype
                columns[name] = ak.values_astype(value, dtype)
            yield ak.Array(columns), resolved, endpoint


def _crystal_ball_invcdf(uniform: Any, mean: Any, sigma: Any, alpha: Any, power: Any) -> Any:
    flat_u, counts = _flatten(uniform)
    if flat_u.size == 0:
        return ak.unflatten(np.asarray([], dtype=float), counts)
    m = _flatten(mean)[0]
    s = np.maximum(abs(_flatten(sigma)[0]), 1e-9)
    a = np.maximum(abs(_flatten(alpha)[0]), 1e-9)
    n = np.maximum(_flatten(power)[0], 1.000001)
    u = np.clip(flat_u, 1e-12, 1.0 - 1e-12)
    sqrt_pi_over_2 = np.sqrt(np.pi / 2.0)
    sqrt2 = np.sqrt(2.0)
    exponent = np.exp(-a * a / 2.0)
    c1 = n / a / (n - 1.0) * exponent
    d1 = 2.0 * sqrt_pi_over_2 * erf(a / sqrt2)
    c = (d1 + 2.0 * c1) / c1
    d = (d1 + 2.0 * c1) / 2.0
    norm = 1.0 / s / (d1 + 2.0 * c1)
    ns = norm * s
    nc = ns * c1
    f = 1.0 - a * a / n
    g = s * n / a

    def cdf(x: np.ndarray) -> np.ndarray:
        delta = (x - m) / s
        result = np.ones_like(delta)
        left_base = f - s * delta / g
        right_base = f + s * delta / g
        left = (delta < -a) & (left_base > 0)
        left_edge = (delta < -a) & ~left
        right = (delta > a) & (right_base > 0)
        right_edge = (delta > a) & ~right
        core = ~(left | left_edge | right | right_edge)
        result[left] = nc[left] / np.power(left_base[left], n[left] - 1.0)
        result[left_edge] = nc[left_edge]
        result[right] = nc[right] * (c[right] - np.power(right_base[right], 1.0 - n[right]))
        result[right_edge] = nc[right_edge] * c[right_edge]
        result[core] = ns[core] * (d[core] - sqrt_pi_over_2 * erf(-delta[core] / sqrt2))
        return result

    cdf_minus = cdf(m - a * s)
    cdf_plus = cdf(m + a * s)
    result = np.zeros_like(u)
    left = (u < cdf_minus) & (nc / u > 0)
    left_edge = (u < cdf_minus) & ~left
    right = (u > cdf_plus) & (c - u / nc > 0)
    right_edge = (u > cdf_plus) & ~right
    core = ~(left | left_edge | right | right_edge)
    result[left] = m[left] + g[left] * (f[left] - np.power(nc[left] / u[left], 1.0 / (n[left] - 1.0)))
    result[left_edge] = m[left_edge] + g[left_edge] * f[left_edge]
    result[right] = m[right] - g[right] * (
        f[right] - np.power(c[right] - u[right] / nc[right], -1.0 / (n[right] - 1.0))
    )
    result[right_edge] = m[right_edge] - g[right_edge] * f[right_edge]
    argument = (d[core] - u[core] / ns[core]) / sqrt_pi_over_2
    result[core] = m[core] - sqrt2 * s[core] * erfinv(np.clip(argument, -1.0, 1.0))
    return ak.unflatten(result, counts)


def _calibrate_muons(arrays: Any, is_data: bool, shift: str, root: Path,
                     *, correction_set: Any = None,
                     preserve_scaled_fallback: bool = False) -> tuple[Any, Any, dict[str, Any]]:
    if correction_set is None:
        raise ValueError("An internal, hash-verified correction set is required")
    cset = correction_set
    pt = arrays["Muon_pt"]
    eta = arrays["Muon_eta"]
    phi = arrays["Muon_phi"]
    charge = arrays["Muon_charge"]
    layers = ak.values_astype(arrays["Muon_nTrackerLayers"], np.float64)
    valid = (pt >= 26.0) & (pt <= 200.0) & (abs(eta) < 2.4) & np.isfinite(pt)
    safe_pt = ak.where(valid, pt, 50.0)
    safe_eta = ak.where(valid, eta, 0.0)
    safe_phi = ak.where(valid, phi, 0.0)
    safe_charge = ak.where(valid, charge, 1)
    safe_layers = ak.where(valid, layers, 10.0)
    sample = "data" if is_data else "mc"
    additive = _evaluate_jagged(cset[f"a_{sample}"], pt, safe_eta, safe_phi, "nom")
    multiplicative = _evaluate_jagged(cset[f"m_{sample}"], pt, safe_eta, safe_phi, "nom")
    scaled = 1.0 / (multiplicative / safe_pt + safe_charge * additive)
    scaled = _valid_scaled_pt(pt, scaled, valid)
    status = {
        "payload": "muon_scalesmearing.json.gz",
        "scale_formula": "1/(m/pt + charge*a)",
        "valid_pt_gev": [26.0, 200.0],
        "shift": shift,
    }
    if is_data:
        corrected_pt = scaled
        status["mode"] = "data_scale"
    else:
        p0 = _evaluate_jagged(cset["poly_params"], pt, abs(safe_eta), safe_layers, 0)
        p1 = _evaluate_jagged(cset["poly_params"], pt, abs(safe_eta), safe_layers, 1)
        p2 = _evaluate_jagged(cset["poly_params"], pt, abs(safe_eta), safe_layers, 2)
        resolution = np.maximum(0.0, p0 + p1 * scaled + p2 * scaled * scaled)
        mean = _evaluate_jagged(cset["cb_params"], pt, abs(safe_eta), safe_layers, 0)
        sigma = _evaluate_jagged(cset["cb_params"], pt, abs(safe_eta), safe_layers, 1)
        power = _evaluate_jagged(cset["cb_params"], pt, abs(safe_eta), safe_layers, 2)
        alpha = _evaluate_jagged(cset["cb_params"], pt, abs(safe_eta), safe_layers, 3)
        uniform = _evaluate_jagged(
            cset["RandomSmearing"], pt, arrays["event"], arrays["luminosityBlock"], safe_phi,
        )
        random_value = _crystal_ball_invcdf(uniform, mean, sigma, alpha, power)
        k_data = _evaluate_jagged(cset["k_data"], pt, abs(safe_eta), "nom")
        k_mc = _evaluate_jagged(cset["k_mc"], pt, abs(safe_eta), "nom")
        residual_k = np.sqrt(np.maximum(k_data * k_data - k_mc * k_mc, 0.0))
        # The official resolution function receives the already scaled pT.
        # Its unphysical-result fallback must therefore retain that scale.
        # Keep the established 2024 behavior unless an era explicitly opts in.
        resolution_base = scaled if preserve_scaled_fallback else pt
        resolution_valid = valid & (scaled >= 26.0) & (scaled <= 200.0) if preserve_scaled_fallback else valid
        nominal = _valid_scaled_pt(resolution_base, scaled * (1.0 + residual_k * resolution * random_value), resolution_valid)
        corrected_pt = nominal
        if shift in {"muonResolutionUp", "muonResolutionDown"}:
            k_unc = _evaluate_jagged(cset["k_mc"], pt, abs(safe_eta), "stat")
            response = (nominal / ak.where(scaled > 0, scaled, 1.0) - 1.0) / ak.where(k_mc > 0, k_mc, 1.0)
            varied_k = k_mc + k_unc if shift.endswith("Up") else k_mc - k_unc
            candidate = scaled * (1.0 + varied_k * response)
            corrected_pt = _valid_scaled_pt(resolution_base, ak.where(k_mc > 0, candidate, nominal), resolution_valid)
        elif shift in {"muonScaleUp", "muonScaleDown"}:
            stat_a = _evaluate_jagged(cset["a_mc"], pt, safe_eta, safe_phi, "stat")
            stat_m = _evaluate_jagged(cset["m_mc"], pt, safe_eta, safe_phi, "stat")
            stat_rho = _evaluate_jagged(cset["m_mc"], pt, safe_eta, safe_phi, "rho_stat")
            variance = stat_m * stat_m / (nominal * nominal) + stat_a * stat_a
            variance = variance + 2.0 * safe_charge * stat_rho * stat_m / nominal * stat_a
            uncertainty = nominal * nominal * np.sqrt(np.maximum(variance, 0.0))
            candidate = nominal + uncertainty if shift.endswith("Up") else nominal - uncertainty
            corrected_pt = _valid_scaled_pt(pt, candidate, valid)
        status.update({
            "mode": "mc_scale_and_crystal_ball_resolution",
            "random_source": "correctionlib RandomSmearing(event,lumi,phi)",
            "unphysical_resolution_fallback": "scaled_pt" if preserve_scaled_fallback else "original_pt",
        })
    mass = arrays["Muon_mass"]
    corrected_mass = mass * corrected_pt / ak.where(pt > 0, pt, 1.0)
    return corrected_pt, corrected_mass, status


def _calibrate_taus(arrays: Any, is_data: bool, shift: str, root: Path,
                    *, correction_set: Any = None) -> tuple[Any, Any, dict[str, Any]]:
    pt = arrays["Tau_pt"]
    mass = arrays["Tau_mass"]
    status = {
        "payload": "tau.json.gz",
        "selection_wp": "DeepTau2018v2p5 VSjet Medium",
        "vse_wp_for_tes": "VVLoose (least restrictive payload category; baseline veto has no VSe cut)",
        "supported_decay_modes": [0, 1, 10, 11],
        "shift": shift,
    }
    if is_data:
        status["mode"] = "not_applicable_data"
        return pt, mass, status
    if correction_set is None:
        raise ValueError("An internal, hash-verified correction set is required")
    cset = correction_set
    eta = arrays["Tau_eta"]
    decay_mode = arrays["Tau_decayMode"]
    genmatch = arrays["Tau_genPartFlav"]
    valid_dm = (decay_mode == 0) | (decay_mode == 1) | (decay_mode == 10) | (decay_mode == 11)
    valid = (pt >= 20.0) & (abs(eta) < 2.5) & valid_dm & (genmatch >= 0) & (genmatch <= 6)
    safe_pt = ak.where(valid, pt, 25.0)
    safe_eta = ak.where(valid, eta, 0.0)
    safe_dm = ak.where(valid, decay_mode, 0)
    safe_genmatch = ak.where(valid, genmatch, 0)
    variation = "up" if shift == "tauEnergyScaleUp" else "down" if shift == "tauEnergyScaleDown" else "nom"
    tes = _evaluate_jagged(
        cset["tau_energy_scale"], pt, safe_pt, safe_eta, safe_dm, safe_genmatch,
        "DeepTau2018v2p5", "Medium", "VVLoose", variation,
    )
    corrected_pt = _valid_scaled_pt(pt, pt * tes, valid)
    corrected_mass = mass * corrected_pt / ak.where(pt > 0, pt, 1.0)
    status.update({"mode": "mc_tau_energy_scale", "variation": variation})
    return corrected_pt, corrected_mass, status
