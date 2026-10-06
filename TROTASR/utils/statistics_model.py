"""Pure statistical-model rules from the user-approved build_combine_inputs.py.

Only mathematical transformations and official nuisance names are retained.
There is no file I/O, legacy import, environment-selected physics, or global year.
"""
from __future__ import annotations
import math
import re
from typing import Any
import numpy as np

ANALYSIS_NUISANCE_PREFIX = "CMS_NPS26012"
RZ_CATEGORIES = ("highdm_Nb1", "highdm_Nb2plus", "lowdm_Nb1", "lowdm_Nb2plus")

def nps_nuisance_name(name: str, year: str) -> str:
    if str(year) not in ('2024', '2025'):
        raise ValueError('Only 2024/2025 are supported')
    if name == "jesTotal":
        return f"CMS_scale_j_Total_{year}"
    # A name is shared between years only when the producing POG explicitly
    # prescribes an across-year correlation.  The BTV fixed-WP payload does so
    # for its ``*_correlated`` variations.  EGM, MUO and PU correctionlib
    # payloads used here are year-specific and do not publish an across-year
    # correlation prescription, so their nuisance names retain the year.
    common = {
        "btagSF_bc_correlated": "CMS_btag_fixedWP_bc_correlated",
        "btagSF_light_correlated": "CMS_btag_fixedWP_light_correlated",
        "electronScale": "CMS_scale_e_13p6TeV",
        "electronSmear": "CMS_res_e_13p6TeV",
        "jer": "CMS_res_j_13p6TeV",
        "jesAbsolute": "CMS_scale_j_Absolute",
        "jesBBEC1": "CMS_scale_j_BBEC1",
        "jesEC2": "CMS_scale_j_EC2",
        "jesFlavorQCD": "CMS_scale_j_FlavorQCD",
        "jesHF": "CMS_scale_j_HF",
        "jesRelativeBal": "CMS_scale_j_RelativeBal",
        "metUnclustered": (
            f"{ANALYSIS_NUISANCE_PREFIX}_scale_met_unclustered_energy_13p6TeV"
        ),
        "muonResolution": f"{ANALYSIS_NUISANCE_PREFIX}_res_m_13p6TeV",
        "muonScale": "CMS_scale_m_13p6TeV",
        "photonScale": "CMS_scale_g_13p6TeV",
        "photonSmear": "CMS_res_g_13p6TeV",
        "tauEnergyScale": "CMS_scale_t_13p6TeV",
    }
    yearly = {
        "btagSF_bc_uncorrelated": "CMS_btag_fixedWP_bc_uncorrelated",
        "btagSF_light_uncorrelated": "CMS_btag_fixedWP_light_uncorrelated",
        "electron_reco": "CMS_eff_e_reco_13p6TeV",
        "electron_hlt": f"{ANALYSIS_NUISANCE_PREFIX}_trigger_e",
        "electron_id": f"{ANALYSIS_NUISANCE_PREFIX}_eff_e_id",
        "loose_muon_5to10": f"{ANALYSIS_NUISANCE_PREFIX}_eff_m_loose_5to10",
        "met_trigger": f"{ANALYSIS_NUISANCE_PREFIX}_trigger_met",
        "muon_hlt": f"{ANALYSIS_NUISANCE_PREFIX}_trigger_m",
        "muon_id": f"{ANALYSIS_NUISANCE_PREFIX}_eff_m_id",
        "muon_iso": "CMS_eff_m_iso_syst",
        "photon_csev": "CMS_eff_g_CSEV_13p6TeV",
        "photon_id": f"{ANALYSIS_NUISANCE_PREFIX}_eff_g_id",
        "photon_trigger": f"{ANALYSIS_NUISANCE_PREFIX}_trigger_g",
        "pileup": "CMS_pileup",
        "veto_electron_5to10": f"{ANALYSIS_NUISANCE_PREFIX}_eff_e_veto_5to10",
    }
    yearly_jes = {
        "jesAbsolute": "CMS_scale_j_Absolute",
        "jesBBEC1": "CMS_scale_j_BBEC1",
        "jesEC2": "CMS_scale_j_EC2",
        "jesHF": "CMS_scale_j_HF",
        "jesRelativeSample": "CMS_scale_j_RelativeSample",
    }
    if name in common:
        return common[name]
    if name in yearly:
        return f"{yearly[name]}_{year}"
    for source, cms_name in yearly_jes.items():
        if name == f"{source}{year}":
            return f"{cms_name}_{year}"
    topw = re.fullmatch(r"topw_(top|w)_(tp[123]|other)_pt(\d+)to(\d+)", name)
    if topw:
        tag, category, low, high = topw.groups()
        cells = {"top": {(400, 480), (480, 600), (600, 1200)},
                 "w": {(200, 300), (300, 400), (400, 800)}}
        if (int(low), int(high)) in cells[tag]:
            return f"{ANALYSIS_NUISANCE_PREFIX}_eff_{tag}_{category}_pt{low}to{high}_{year}"
    raise ValueError(f"no CMS nuisance-name mapping for canonical variation {name!r}")

def require_positive(record: dict[str, Any], label: str, key: str = "value") -> float:
    if record.get("status") != "complete":
        raise ValueError(f"{label} is not complete: {record}")
    value = float(record[key])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} is nonpositive/nonfinite: {value}")
    return value

def scaled_record(
    record: dict[str, Any] | None, scale: float
) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "nominal": record["nominal"] * scale,
        "sumw2": record["sumw2"] * scale * scale,
        "variations": {
            nuisance: {
                "up": pair["up"] * scale,
                "down": pair["down"] * scale,
            }
            for nuisance, pair in record["variations"].items()
        },
    }

def sum_one_bin_backgrounds(
    records: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    retained = [record for record in records if record is not None]
    if not retained:
        return None
    nominal = sum(
        (record["nominal"] for record in retained), np.zeros(1)
    )
    sumw2 = sum((record["sumw2"] for record in retained), np.zeros(1))
    nuisances = sorted(
        {
            nuisance
            for record in retained
            for nuisance in record["variations"]
        }
    )
    variations = {}
    for nuisance in nuisances:
        up = np.zeros(1)
        down = np.zeros(1)
        for record in retained:
            pair = record["variations"].get(nuisance)
            up += pair["up"] if pair else record["nominal"]
            down += pair["down"] if pair else record["nominal"]
        variations[nuisance] = {"up": up, "down": down}
    return {"nominal": nominal, "sumw2": sumw2, "variations": variations}

def build_rz_covariance(
    high_summary: dict[str, Any], low_summary: dict[str, Any], year: str
) -> dict[str, Any]:
    low_rz = low_summary.get("rz_low") or low_summary.get("rz_low_feature")
    if not isinstance(low_rz, dict):
        raise ValueError("Low-dM RZ measurement is missing")
    records = {
        "highdm_Nb1": high_summary["rz_high"]["combined"]["Nb1"],
        "highdm_Nb2plus": high_summary["rz_high"]["combined"]["Nb2plus"],
        "lowdm_Nb1": low_rz["combined"]["Nb1"],
        "lowdm_Nb2plus": low_rz["combined"]["Nb2plus"],
    }
    central = np.asarray(
        [require_positive(records[key], f"RZ/{key}", "RZ") for key in RZ_CATEGORIES]
    )
    errors = np.asarray([float(records[key]["RZ_stat"]) for key in RZ_CATEGORIES])
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError(f"invalid RZ statistical errors: {errors}")
    covariance_r = np.diag(errors * errors)
    covariance_eta = covariance_r / np.outer(central, central)
    cholesky = np.linalg.cholesky(covariance_eta)
    nuisances = []
    for index, category in enumerate(RZ_CATEGORIES):
        nuisances.append(
            {
                "name": f"{ANALYSIS_NUISANCE_PREFIX}_RZstat_{category}_{year}",
                "category": category,
                "log_coefficient": float(cholesky[index, index]),
            }
        )
    return {
        "schema_version": f"rz_covariance_{year}_v1",
        "status": "temporary_statistical_only",
        "warning": (
            "Diagonal statistical-only covariance. Replace with the final "
            "documented cross-category covariance before the final result."
        ),
        "categories": list(RZ_CATEGORIES),
        "central": central.tolist(),
        "statistical_errors": errors.tolist(),
        "covariance_r": covariance_r.tolist(),
        "covariance_log_r": covariance_eta.tolist(),
        "cholesky_log_r": cholesky.tolist(),
        "nuisances": nuisances,
    }

def rz_value(covariance: dict[str, Any], category: str) -> float:
    return float(covariance["central"][covariance["categories"].index(category)])

def rz_nuisances(covariance: dict[str, Any], category: str) -> list[dict[str, Any]]:
    index = covariance["categories"].index(category)
    output = []
    for column, nuisance in enumerate(covariance["nuisances"]):
        coefficient = float(covariance["cholesky_log_r"][index][column])
        if coefficient == 0.0:
            continue
        output.append(
            {
                "name": nuisance["name"],
                "down": math.exp(-coefficient),
                "up": math.exp(coefficient),
            }
        )
    return output

def closure_record(
    double_ratio: dict[str, Any], regime: str, low: float, high: float, year: str
) -> tuple[str, float, list[int]]:
    selected = []
    for index, record in enumerate(double_ratio[regime]["bins"]):
        if record.get("status") != "complete":
            continue
        if float(record["high"]) <= low or float(record["low"]) >= high:
            continue
        value = float(record["double_ratio"])
        if np.isfinite(value):
            selected.append((index, abs(value - 1.0)))
    if not selected:
        return "", 0.0, []
    delta = max(value for _, value in selected)
    high_label = "Inf" if not np.isfinite(high) else str(int(high))
    name = (
        f"{ANALYSIS_NUISANCE_PREFIX}_zgammaNonclosure_{regime}_"
        f"u{int(low)}to{high_label}_{year}"
    )
    return name, delta, [index for index, _ in selected]

def rate_parameter(
    kind: str, regime: str, group: str, bin_index: int | str, year: str
) -> str:
    suffix = f"bin{bin_index}" if isinstance(bin_index, int) else str(bin_index)
    return (
        f"{ANALYSIS_NUISANCE_PREFIX}_{kind}_{regime}_{group}_"
        f"{suffix}_{year}"
    )

def add_extra(
    channel: dict[str, Any], process: str, records: list[dict[str, Any]]
) -> None:
    if records:
        channel.setdefault("extra_lnN", {}).setdefault(process, []).extend(records)
