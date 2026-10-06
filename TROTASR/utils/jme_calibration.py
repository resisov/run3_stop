"""Internal calibrated JME p4 evaluation, migrated from the adopted backend.

Returns CMS-calibrated p4 only; does NOT select a mapping to the historical
NanoAOD-p4 TROTA nominal. No ROOT writes and no external analysis imports.
"""
from __future__ import annotations
from typing import Any
from functools import lru_cache
import hashlib
import math
import numpy as np
import awkward as ak
import correctionlib
from .paths import ROOT
from .io import sha256, read_json

TAGS = {2024: ('Summer24Prompt24_V5', 'Summer24Prompt24_JRV2'),
        2025: ('Summer24Prompt25_V3', 'Summer24Prompt25_JRV2')}
ENDPOINTS = ('nominal', 'jesTotalUp', 'jesTotalDown', 'jerUp', 'jerDown')


@lru_cache(maxsize=4)
def correction_set(year, prefix):
    if year not in TAGS or prefix not in ('Jet', 'FatJet'):
        raise ValueError('Unsupported JME year/collection')
    manifest = read_json(ROOT/'scales/JME/manifest.json')['years'][str(year)]
    record = manifest[prefix]
    relative = record['path']
    if (TAGS[year] != (manifest['jec_tag'], manifest['jer_tag'])
            or sha256(ROOT/relative) != record['sha256']):
        raise ValueError('JME payload provenance mismatch: ' + relative)
    return correctionlib.CorrectionSet.from_file(str(ROOT/relative))


def calibrated_endpoints(chunk, year):
    """Evaluate nominal-correlated CMS AK4/AK8 endpoints once per read slice.

    No convention ratio to a nearly-zero nominal jet is ever taken here.
    The caller must validate nominal storage and explicitly choose inference
    convention before these values are used for production histograms.
    """
    if np.any(np.asarray(chunk['is_data'], dtype=bool)):
        raise ValueError('JES/JER MC variations cannot be evaluated for data')
    jec, jer = TAGS[year]
    arrays = jet_inputs(chunk)
    result = {shift: {} for shift in ENDPOINTS}
    for prefix, stored, gen, radius in (('Jet', 'jet', 'GenJet', 'AK4PFPuppi'),
                                       ('FatJet', 'fatjet', 'GenJetAK8', 'AK8PFPuppi')):
        cset = correction_set(year, prefix)
        pt_nano, mass_nano = arrays[prefix+'_pt'], arrays[prefix+'_mass']
        raw_factor = arrays[prefix+'_rawFactor']
        eta, phi, area = [arrays[prefix+'_'+field] for field in ('eta', 'phi', 'area')]
        rho = arrays['Rho_fixedGridRhoFastjetAll']
        raw_pt, raw_mass = pt_nano*(1-raw_factor), mass_nano*(1-raw_factor)
        correction = _evaluate_jagged(cset.compound[f'{jec}_MC_L1L2L3Res_{radius}'],
                                      raw_pt, area, eta, raw_pt, rho, phi)
        pt, mass = raw_pt*correction, raw_mass*correction
        resolution = _evaluate_jagged(cset[f'{jer}_MC_PtResolution_{radius}'], pt, eta, pt, rho)
        sf = _evaluate_jagged(cset[f'{jer}_MC_ScaleFactor_{radius}'], pt, eta, pt)
        unc = _evaluate_jagged(cset[f'{jer}_MC_SFUncertainty_{radius}'], pt, eta, pt)
        index = prefix+'_genJetIdx' if prefix == 'Jet' else 'FatJet_genJetAK8Idx'
        for shift, scale in (('nominal', sf), ('jerUp', sf+unc), ('jerDown', sf-unc)):
            smear = _jer_smear(pt, eta, resolution, scale, arrays[index], arrays[gen+'_pt'],
                np.asarray(arrays['run']), np.asarray(arrays['luminosityBlock']),
                np.asarray(arrays['event']), prefix, corrected_mass=mass)
            result[shift][stored+'_corrected_pt'] = pt*smear
            result[shift][stored+'_corrected_mass'] = mass*smear
        nominal_pt = result['nominal'][stored+'_corrected_pt']
        jes = _evaluate_jagged(cset[f'{jec}_MC_Total_{radius}'], nominal_pt, eta, nominal_pt)
        for shift, sign in (('jesTotalUp', 1), ('jesTotalDown', -1)):
            for field in ('pt', 'mass'):
                name = stored+'_corrected_'+field
                result[shift][name] = result['nominal'][name]*(1+sign*jes)
        for shift in ENDPOINTS:
            for field in ('pt', 'mass'):
                values = result[shift][stored+'_corrected_'+field]
                if not np.isfinite(np.asarray(ak.flatten(values))).all():
                    raise ValueError('Nonfinite calibrated JME endpoint')
    return result

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


def _jer_smear(
    corrected_pt: Any,
    eta: Any,
    resolution: Any,
    scale_factor: Any,
    gen_index: Any,
    gen_pt: Any,
    run: np.ndarray,
    lumi: np.ndarray,
    event: np.ndarray,
    prefix: str,
    *,
    corrected_mass: Any,
) -> Any:
    """Hybrid JER with the CMSSW 0.01 GeV minimum *energy* prescription."""
    out = []
    for iev, (pts, etas, masses, resolutions, sfs, indices) in enumerate(
        zip(
            ak.to_list(corrected_pt),
            ak.to_list(eta),
            ak.to_list(corrected_mass),
            ak.to_list(resolution),
            ak.to_list(scale_factor),
            ak.to_list(gen_index),
        )
    ):
        gen_pts = ak.to_list(gen_pt[iev])
        event_out = []
        for ijet, (pt, jet_eta, mass, res, sf, index) in enumerate(zip(pts, etas, masses, resolutions, sfs, indices)):
            pt = float(pt)
            if not all(math.isfinite(float(x)) for x in (pt, jet_eta, mass, res, sf)):
                raise ValueError("Nonfinite JER input")
            if pt < 0:
                raise ValueError("Negative JER input pT")
            if pt == 0:
                # CMSSW copies an input jet with zero pT without smearing.
                event_out.append(1.0)
                continue
            # ROOT's PtEtaPhiM convention permits tiny negative stored masses
            # for spacelike numerical roundoff: M2 = M * abs(M).
            energy2 = (pt * math.cosh(float(jet_eta))) ** 2 + float(mass) * abs(float(mass))
            if not math.isfinite(energy2) or energy2 <= 0:
                raise ValueError("Invalid JER input energy")
            energy = math.sqrt(energy2)
            res = max(0.0, float(res))
            sf = max(0.0, float(sf))
            matched = 0 <= int(index) < len(gen_pts)
            if matched:
                matched_pt = float(gen_pts[int(index)])
                matched = abs(pt - matched_pt) < 3.0 * res * pt
            if matched:
                factor = 1.0 + (sf - 1.0) * (pt - matched_pt) / max(pt, 1e-6)
            else:
                random_value = _deterministic_normal(int(run[iev]), int(lumi[iev]), int(event[iev]), prefix, ijet)
                factor = 1.0 + random_value * res * math.sqrt(max(sf * sf - 1.0, 0.0))
            # Scale the entire four-vector; this is not a pT floor.
            # PhysicsTools/PatUtils/interface/SmearedJetProducerT.h
            event_out.append(max(0.01 / energy, factor))
        out.append(event_out)
    return ak.Array(out)


def jet_inputs(chunk):
    values = {k: chunk[k] for k in ("run", "luminosityBlock", "event")}
    values["Rho_fixedGridRhoFastjetAll"] = chunk["rho"]
    fields = {"pt": "nanoaod_pt", "mass": "nanoaod_mass", "eta": "eta_all",
              "phi": "phi_all", "rawFactor": "raw_factor", "area": "area"}
    for prefix, stored, gen in (("Jet", "jet", "GenJet"), ("FatJet", "fatjet", "GenJetAK8")):
        for nano, field in fields.items():
            values[f"{prefix}_{nano}"] = chunk[f"{stored}_{field}"]
        index = "genJetIdx" if prefix == "Jet" else "genJetAK8Idx"
        source = "jet_genjet_index_all" if prefix == "Jet" else "fatjet_genjetak8_index_all"
        values[f"{prefix}_{index}"] = chunk[source]
        values[f"{gen}_pt"] = chunk["genjet_pt_all" if prefix == "Jet" else "genjetak8_pt_all"]
    return values
