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
from .paths import ROOT, era_year, analysis_era, EARLY_ERAS
from .io import sha256, read_json

TAGS = {'2022': ('Summer22_22Sep2023_V4', 'Summer22_22Sep2023_JRV2'),
        '2022EE': ('Summer22EE_22Sep2023_V4', 'Summer22EE_22Sep2023_JRV2'),
        '2023': ('Summer23Prompt23_V4', 'Summer23Prompt23_RunCv1234_JRV3'),
        '2023BPix': ('Summer23BPixPrompt23_V4', 'Summer23BPixPrompt23_RunD_JRV3'),
        2024: ('Summer24Prompt24_V5', 'Summer24Prompt24_JRV2'),
        2025: ('Summer24Prompt25_V3', 'Summer24Prompt25_JRV2')}
ENDPOINTS = ('nominal', 'jesTotalUp', 'jesTotalDown', 'jerUp', 'jerDown')


@lru_cache(maxsize=12)
def correction_set(year, prefix):
    year = era_year(year)
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
    year = era_year(year)
    early = analysis_era(year) in EARLY_ERAS
    jec, jer = TAGS[year]
    arrays = jet_inputs(chunk, early=early)
    result = {shift: {} for shift in ENDPOINTS}
    for prefix, stored, gen, radius in (('Jet', 'jet', 'GenJet', 'AK4PFPuppi'),
                                       ('FatJet', 'fatjet', 'GenJetAK8', 'AK8PFPuppi')):
        cset = correction_set(year, prefix)
        pt_nano, mass_nano = arrays[prefix+'_pt'], arrays[prefix+'_mass']
        raw_factor = arrays[prefix+'_rawFactor']
        eta, phi, area = [arrays[prefix+'_'+field] for field in ('eta', 'phi', 'area')]
        rho = arrays['Rho_fixedGridRhoFastjetAll']
        raw_pt, raw_mass = pt_nano*(1-raw_factor), mass_nano*(1-raw_factor)
        inputs = dict(JetA=area, JetEta=eta, JetPhi=phi, JetPt=raw_pt,
                      Rho=rho, run=arrays['run'])
        correction = _evaluate_named(cset.compound[f'{jec}_MC_L1L2L3Res_{radius}'], raw_pt, inputs)
        pt, mass = raw_pt*correction, raw_mass*correction
        inputs['JetPt'] = pt
        resolution = _evaluate_named(cset[f'{jer}_MC_PtResolution_{radius}'], pt, inputs)
        sf = _evaluate_named(cset[f'{jer}_MC_ScaleFactor_{radius}'], pt, inputs)
        unc = _evaluate_named(cset[f'{jer}_MC_SFUncertainty_{radius}'], pt, inputs)
        if early:
            unc = _early_supported_jer_uncertainty(eta, correction, resolution, sf, unc)
        index = prefix+'_genJetIdx' if prefix == 'Jet' else 'FatJet_genJetAK8Idx'
        for shift, scale in (('nominal', sf), ('jerUp', sf+unc), ('jerDown', sf-unc)):
            if early:
                smear = _early_jer_smear(pt, eta, phi, resolution, scale,
                    arrays[index], arrays[gen+'_pt'], arrays[gen+'_eta'], arrays[gen+'_phi'],
                    arrays['run'], arrays['luminosityBlock'], arrays['event'], prefix,
                    .4 if prefix == 'Jet' else .8)
            else:
                smear = _jer_smear(pt, eta, resolution, scale, arrays[index], arrays[gen+'_pt'],
                    np.asarray(arrays['run']), np.asarray(arrays['luminosityBlock']),
                    np.asarray(arrays['event']), prefix, corrected_mass=mass)
            result[shift][stored+'_corrected_pt'] = pt*smear
            result[shift][stored+'_corrected_mass'] = mass*smear
        nominal_pt = result['nominal'][stored+'_corrected_pt']
        inputs['JetPt'] = nominal_pt
        jes = _evaluate_named(cset[f'{jec}_MC_Total_{radius}'], nominal_pt, inputs)
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


def _evaluate_named(correction, like, values):
    """Use the frozen payload's declared inputs, including BPix JetPhi."""
    missing = [item.name for item in correction.inputs if item.name not in values]
    if missing:
        raise ValueError(correction.name + ': missing correction inputs ' + repr(missing))
    result = _evaluate_jagged(correction, like, *(values[item.name] for item in correction.inputs))
    if not bool(ak.all(np.isfinite(result))):
        raise ValueError(correction.name + ': nonfinite correction')
    return result


def _early_supported_jer_uncertainty(eta, jec, resolution, sf, uncertainty):
    """Ignore only the exact forward payload sentinel outside all consumers.

    Type-1 MET uses |eta| < 5.2; all jet/TROTA selections are tighter.
    These identity-calibrated jets cannot migrate into that angular support.
    This extends only a formerly fatal domain, not valid JER uncertainties.
    Invalid values inside the support (or any other sentinel) still fail.
    """
    outside = ((abs(eta) >= 5.2) & (jec == 1.) & (resolution == 1.)
               & (sf == 1.) & (uncertainty == -999.))
    return ak.where(outside, 0., uncertainty)


def _early_jer_smear(pt, eta, phi, resolution, sf, gen_index, gen_pt, gen_eta,
                     gen_phi, run, lumi, event, prefix, radius):
    """Exact retained early-era hybrid JER (dR<R/2, dpt<3 sigma, zero floor).

    Migrated from ObjectCorrections2022; the 2024/25 energy-floor prescription
    is deliberately not substituted for this frozen production convention.
    """
    output = []
    vectors = [ak.to_list(x) for x in (pt, eta, phi, resolution, sf, gen_index,
                                     gen_pt, gen_eta, gen_phi)]
    for iev, rows in enumerate(zip(*vectors)):
        pts, etas, phis, resolutions, sfs, indices, gpts, getas, gphis = rows
        factors = []
        for i, (p, e, f, res, scale, index) in enumerate(zip(pts, etas, phis, resolutions, sfs, indices)):
            if not all(np.isfinite(x) for x in (p, e, f, res, scale)) or p <= 0 or res < 0 or scale < 0:
                raise ValueError('Invalid JER input')
            matched = 0 <= int(index) < len(gpts)
            if matched:
                dphi = np.arctan2(np.sin(f-gphis[index]), np.cos(f-gphis[index]))
                matched = np.hypot(e-getas[index], dphi) < radius/2 and abs(p-gpts[index]) < 3*res*p
            if matched:
                factor = 1 + (scale-1)*(p-gpts[index])/p
            else:
                z = _deterministic_normal(int(run[iev]), int(lumi[iev]), int(event[iev]), prefix, i)
                factor = 1 + z*res*np.sqrt(max(scale*scale-1, 0))
            factors.append(max(0.0, factor))
        output.append(factors)
    return ak.Array(output)


def jet_inputs(chunk, early=False):
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
        if early:
            for field in ('eta', 'phi'):
                values[f'{gen}_{field}'] = chunk[('genjet_' if prefix == 'Jet' else 'genjetak8_')+field+'_all']
    return values
