"""Built-in physics profiles; user JSON overrides them without editing code."""

from __future__ import annotations

import copy
from typing import Any, Mapping

FILTERS = [
    "Flag_goodVertices",
    "Flag_globalSuperTightHalo2016Filter",
    "Flag_HBHENoiseFilter",
    "Flag_HBHENoiseIsoFilter",
    "Flag_EcalDeadCellTriggerPrimitiveFilter",
    "Flag_BadPFMuonFilter",
    "Flag_BadPFMuonDzFilter",
    "Flag_eeBadScFilter",
    "Flag_ecalBadCalibFilter",
]


def _base() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "input": {
            "tree": "Events",
            "event_id": ["run", "luminosityBlock", "event"],
            "event_filters": FILTERS,
        },
        "weights": {"mc_nominal": "genWeight", "mc_variations": {}},
        "fit": {
            "mass_bins": 60,
            "signal_model": "crystal_ball",
            "background_model": "exponential",
            "alternate_signal_model": "double_gaussian",
            "alternate_background_model": "linear",
            "rebin_factors": [1, 2],
            "window_shrink_fraction": 0.05,
            "min_fail_significance": 0.0,
        },
        "correction": {"flow": "clamp"},
    }


def _electron_fields() -> list[str]:
    return [
        "pt",
        "eta",
        "phi",
        "mass",
        "charge",
        "deltaEtaSC",
        "cutBased",
        "miniPFRelIso_all",
        "convVeto",
        "lostHits",
    ]


def _muon_fields() -> list[str]:
    return [
        "pt",
        "eta",
        "phi",
        "mass",
        "charge",
        "looseId",
        "tightId",
        "isTracker",
        "miniPFRelIso_all",
    ]


def _z_fit() -> dict[str, Any]:
    return {
        "peak_bounds_gev": [86.0, 96.0],
        "mass_bins": 60,
        "signal_model": "voigt",
        "natural_width_gev": 1.2476,
        "background_model": "exponential",
        "alternate_signal_model": "double_gaussian",
    }


PROFILES: dict[str, dict[str, Any]] = {}

electron_jpsi = _base()
electron_jpsi.update(
    {
        "tag": {
            "collection": "Electron",
            "fields": _electron_fields(),
            "selection": "(pt > 5) & ((abs(eta + deltaEtaSC) < 1.4442) | ((abs(eta + deltaEtaSC) > 1.566) & (abs(eta + deltaEtaSC) < 2.5))) & (cutBased >= 4) & (miniPFRelIso_all < 0.1)",
        },
        "probe": {
            "collection": "Electron",
            "fields": _electron_fields(),
            "selection": "(pt > 5) & (pt < 10) & ((abs(eta + deltaEtaSC) < 1.4442) | ((abs(eta + deltaEtaSC) > 1.566) & (abs(eta + deltaEtaSC) < 2.5))) & convVeto & (lostHits <= 1)",
            "pass": "cutBased >= 1",
            "pt": "pt",
            "eta": "eta + deltaEtaSC",
        },
        "pair": {
            "selection": "tag_charge * probe_charge < 0",
            "mass_window_gev": [2.0, 4.0],
        },
        "axes": {"pt_edges_gev": [5, 7, 10], "abseta_edges": [0.0, 0.8, 1.4442, 2.5]},
        "fit": {
            **electron_jpsi["fit"],
            "peak_bounds_gev": [2.85, 3.30],
            "mass_bins": 100,
            "background_model": "chebyshev2",
        },
        "reference_trigger": {
            "paths": ["HLT_Mu9_Barrel_L1HP10_IP6", "HLT_Mu10_Barrel_L1HP11_IP6"],
            "apply_to_data": True,
            "apply_to_mc": False,
            "match_tag": False,
        },
    }
)
PROFILES["electron_jpsi_lowpt"] = electron_jpsi

muon_jpsi = _base()
muon_jpsi.update(
    {
        "tag": {
            "collection": "Muon",
            "fields": _muon_fields(),
            "selection": "(pt > 5) & (abs(eta) < 2.4) & tightId",
        },
        "probe": {
            "collection": "Muon",
            "fields": _muon_fields(),
            "selection": "(pt > 5) & (pt < 10) & (abs(eta) < 2.4) & isTracker",
            "pass": "looseId",
            "pt": "pt",
            "eta": "eta",
        },
        "spectator": {
            "collection": "Muon",
            "fields": _muon_fields(),
            "selection": "(pt > 12) & (abs(eta) < 1.5) & tightId",
            "distinct_from_pair": True,
        },
        "pair": {
            "selection": "tag_charge * probe_charge < 0",
            "mass_window_gev": [2.6, 3.6],
        },
        "axes": {
            "pt_edges_gev": [5, 6, 7, 8, 9, 10],
            "abseta_edges": [0.0, 0.9, 1.2, 2.1, 2.4],
        },
        "fit": {**muon_jpsi["fit"], "peak_bounds_gev": [2.85, 3.30], "mass_bins": 50},
        "reference_trigger": {
            "paths": ["HLT_Mu9_Barrel_L1HP10_IP6", "HLT_Mu10_Barrel_L1HP11_IP6"],
            "apply_to_data": True,
            "apply_to_mc": False,
            "match_tag": False,
        },
    }
)
PROFILES["muon_jpsi_lowpt"] = muon_jpsi

electron_z = _base()
electron_z.update(
    {
        "tag": {
            "collection": "Electron",
            "fields": _electron_fields(),
            "selection": "(pt > 35) & ((abs(eta + deltaEtaSC) < 1.4442) | ((abs(eta + deltaEtaSC) > 1.566) & (abs(eta + deltaEtaSC) < 2.5))) & (cutBased >= 4)",
        },
        "probe": {
            "collection": "Electron",
            "fields": _electron_fields(),
            "selection": "(pt > 10) & ((abs(eta + deltaEtaSC) < 1.4442) | ((abs(eta + deltaEtaSC) > 1.566) & (abs(eta + deltaEtaSC) < 2.5))) & convVeto & (lostHits <= 1)",
            "pass": "cutBased >= 4",
            "pt": "pt",
            "eta": "eta + deltaEtaSC",
        },
        "pair": {
            "selection": "tag_charge * probe_charge < 0",
            "mass_window_gev": [60, 120],
        },
        "axes": {
            "pt_edges_gev": [10, 20, 35, 50, 100, 200, 500],
            "abseta_edges": [0.0, 0.8, 1.4442, 2.5],
        },
        "fit": {**electron_z["fit"], **_z_fit()},
        "reference_trigger": {
            "paths": ["HLT_Ele32_WPTight_Gsf"],
            "apply_to_data": True,
            "apply_to_mc": True,
            "match_tag": True,
            "object_id": 11,
            "filter_bits": 2,
            "max_delta_r": 0.1,
        },
    }
)
PROFILES["electron_z"] = electron_z

muon_z = _base()
muon_z.update(
    {
        "tag": {
            "collection": "Muon",
            "fields": _muon_fields(),
            "selection": "(pt > 26) & (abs(eta) < 2.4) & tightId & (miniPFRelIso_all < 0.1)",
        },
        "probe": {
            "collection": "Muon",
            "fields": _muon_fields(),
            "selection": "(pt > 10) & (abs(eta) < 2.4) & isTracker",
            "pass": "tightId",
            "pt": "pt",
            "eta": "eta",
        },
        "pair": {
            "selection": "tag_charge * probe_charge < 0",
            "mass_window_gev": [60, 120],
        },
        "axes": {
            "pt_edges_gev": [10, 20, 30, 50, 100, 200, 500],
            "abseta_edges": [0.0, 0.9, 1.2, 2.1, 2.4],
        },
        "fit": {**muon_z["fit"], **_z_fit()},
        "reference_trigger": {
            "paths": ["HLT_IsoMu24"],
            "apply_to_data": True,
            "apply_to_mc": True,
            "match_tag": True,
            "object_id": 13,
            "filter_bits": 2,
            "max_delta_r": 0.1,
        },
    }
)
PROFILES["muon_z"] = muon_z

photon_z = _base()
photon_z.update(
    {
        "tag": {
            "collection": "Electron",
            "fields": _electron_fields(),
            "selection": "(pt > 35) & ((abs(eta + deltaEtaSC) < 1.4442) | ((abs(eta + deltaEtaSC) > 1.566) & (abs(eta + deltaEtaSC) < 2.5))) & (cutBased >= 4)",
        },
        "probe": {
            "collection": "Photon",
            "fields": [
                "pt",
                "eta",
                "phi",
                "mass",
                "deltaEtaSC",
                "cutBased",
                "pixelSeed",
                "electronVeto",
            ],
            "selection": "(pt > 20) & ((abs(eta + deltaEtaSC) < 1.4442) | ((abs(eta + deltaEtaSC) > 1.566) & (abs(eta + deltaEtaSC) < 2.5)))",
            "pass": "cutBased >= 3",
            "pt": "pt",
            "eta": "eta + deltaEtaSC",
        },
        "pair": {"selection": "delta_r > 0.2", "mass_window_gev": [60, 120]},
        "axes": {
            "pt_edges_gev": [20, 35, 50, 100, 200, 500, 1000],
            "abseta_edges": [0.0, 1.4442, 2.5],
        },
        "fit": {**photon_z["fit"], **_z_fit()},
        "reference_trigger": {
            "paths": ["HLT_Ele32_WPTight_Gsf"],
            "apply_to_data": True,
            "apply_to_mc": True,
            "match_tag": True,
            "object_id": 11,
            "filter_bits": 2,
            "max_delta_r": 0.1,
        },
    }
)
PROFILES["photon_z"] = photon_z


def _merge(target: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            target[key] = _merge(dict(target[key]), value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def resolve_profile(user: Mapping[str, Any]) -> dict[str, Any]:
    """Expand one compact user config into the complete immutable run config."""

    if "profile" not in user:
        return copy.deepcopy(dict(user))
    name = str(user["profile"])
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}")
    resolved = copy.deepcopy(PROFILES[name])
    direct = {
        key: value
        for key, value in user.items()
        if key not in {"profile", "id", "pt_edges_gev", "abseta_edges"}
    }
    _merge(resolved, direct)
    identity = user.get("id", {})
    if identity:
        if "fields" in identity:
            resolved["probe"]["fields"] = list(
                dict.fromkeys(
                    list(resolved["probe"]["fields"])
                    + [str(value) for value in identity["fields"]]
                )
            )
        if "denominator" in identity:
            resolved["probe"]["selection"] = str(identity["denominator"])
        if "pass" in identity:
            resolved["probe"]["pass"] = str(identity["pass"])
    if "pt_edges_gev" in user:
        resolved["axes"]["pt_edges_gev"] = user["pt_edges_gev"]
    if "abseta_edges" in user:
        resolved["axes"]["abseta_edges"] = user["abseta_edges"]
    resolved["profile"] = name
    resolved.setdefault("correction", {})
    resolved["correction"].setdefault(
        "name", str(resolved.get("measurement", "tnp_sf"))
    )
    resolved["correction"].setdefault(
        "description", f"Data/MC scale factor for {resolved.get('measurement', name)}"
    )
    return resolved
