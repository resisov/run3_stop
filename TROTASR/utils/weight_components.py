from __future__ import annotations
import math
import json
import os
import contextlib
from pathlib import Path
from typing import Any, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import awkward as ak
from .corrections import load_analysis_corrections, analysis_workdir
from .private_scales import (DEFAULT_ANALYSIS_SF_COMPONENTS, REQUIRED_ANALYSIS_SF_COMPONENTS, loose_muon_lowpt_triplet, met_trigger_triplet, photon_trigger_triplet, veto_electron_lowpt_triplet)


_BTAG_CORRECTOR_CACHE: dict[tuple[Path, str, str, str, str], Any] = {}


def campaign_year(year: str) -> str:
    return year if year in {"2022pre", "2022post", "2023pre", "2023post", "2024", "2025"} else "2024"


def analysis_year(year: str) -> str:
    return campaign_year(year)


def np_filled(values: Any, n: int, fill: float = 1.0) -> np.ndarray:
    try:
        out = np.asarray(ak.to_numpy(ak.fill_none(values, fill)), dtype=float)
    except Exception:
        out = np.asarray(values, dtype=float)
    if out.shape == ():
        out = np.full(n, float(out), dtype=float)
    if len(out) != n:
        return np.full(n, fill, dtype=float)
    return np.where(np.isfinite(out), out, fill).astype(float)


def jagged_prod(values: Any, n: int, fill: float = 1.0) -> np.ndarray:
    try:
        return np_filled(ak.prod(ak.fill_none(values, fill), axis=1), n, fill)
    except Exception:
        return np.full(n, fill, dtype=float)


def finite_float(value: Any, fill: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fill)
    return out if math.isfinite(out) else float(fill)


def finite_weight_array(values: Any, n: int, fill: float = 0.0) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=float)
    except Exception:
        return np.full(n, fill, dtype=float)
    if out.shape == ():
        out = np.full(n, finite_float(out, fill), dtype=float)
    if len(out) != n:
        return np.full(n, fill, dtype=float)
    return np.where(np.isfinite(out), out, fill).astype(float)


def replace_component(base: dict[str, np.ndarray], name: str, varied: np.ndarray) -> np.ndarray:
    n = len(next(iter(base.values())))
    out = np.ones(n, dtype=float)
    for comp_name, comp in base.items():
        factor = finite_weight_array(varied if comp_name == name else comp, n, 1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            out = out * factor
        out = np.where(np.isfinite(out), out, 0.0)
    return out


def get_btag_corrector(repo: Path, year: str, caller: str, tagger: str = "UParTAK4", workingpoint: str = "medium") -> Any:
    key = (repo.resolve(), analysis_year(year), caller, tagger, workingpoint)
    if key not in _BTAG_CORRECTOR_CACHE:
        corrections = load_analysis_corrections(repo)
        with analysis_workdir(repo):
            _BTAG_CORRECTOR_CACHE[key] = corrections["get_btag_weight"](tagger, analysis_year(year), workingpoint, caller)
    return _BTAG_CORRECTOR_CACHE[key]


def compute_weight_bundle(
    arrays: dict[str, Any],
    repo: Path,
    dataset: str,
    process: str,
    year: str,
    n: int,
    jet_pt: Any,
    jet_eta: Any,
    jet_hadflav: Any,
    b_med: Any,
    e_eta: Any,
    e_delta_eta_sc: Any,
    e_pt: Any,
    e_phi: Any,
    e_veto: Any,
    e_med: Any,
    n_e_veto: np.ndarray,
    n_e_med: np.ndarray,
    m_eta: Any,
    m_pt: Any,
    m_phi: Any,
    m_loose: Any,
    m_med: Any,
    n_m_loose: np.ndarray,
    n_m_med: np.ndarray,
    p_eta: Any,
    p_pt: Any,
    p_phi: Any,
    p_med: Any,
    gcr_mask: np.ndarray,
    p_r9: Any | None = None,
    photon_id_wp: str = "Medium",
    met_pt: Any | None = None,
    met_trigger_mask: np.ndarray | None = None,
    analysis_sf_components: tuple[str, ...] | list[str] | None = None,
    include_variations: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    if is_data_process(process):
        ones = np.ones(n, dtype=float)
        return ones, {"nominal": ones}, {"applied": False, "reason": "data", "available_variations": ["nominal"], "components": {}}

    corrections = load_analysis_corrections(repo)
    y = analysis_year(year)
    gen = np_filled(arr(arrays, "genWeight", np.ones(n)), n, 1.0)
    one = np.ones(n, dtype=float)
    components: dict[str, np.ndarray] = {}
    alternates: dict[str, tuple[str, np.ndarray]] = {}
    status: dict[str, Any] = {"applied": True, "available_variations": ["nominal"], "components": {}}
    enabled_analysis_sf = set(
        DEFAULT_ANALYSIS_SF_COMPONENTS
        if analysis_sf_components is None
        else analysis_sf_components
    )
    unknown_analysis_sf = enabled_analysis_sf - set(REQUIRED_ANALYSIS_SF_COMPONENTS)
    if unknown_analysis_sf:
        raise ValueError(
            "unknown analysis-owned SF components: "
            + ", ".join(sorted(unknown_analysis_sf))
        )

    def record(name: str, applied: bool, source: str, error: str | None = None) -> None:
        item = {"applied": applied, "source": source}
        if error:
            item["error"] = error[:400]
        status["components"][name] = item

    if has_field(arrays, "Pileup_nTrueInt"):
        try:
            with analysis_workdir(repo):
                pu_nom, pu_up, pu_down = corrections["get_pu_weight"](y, arrays["Pileup_nTrueInt"])
            components["pileup"] = np_filled(pu_nom, n, 1.0)
            if include_variations:
                alternates["pileupUp"] = ("pileup", np_filled(pu_up, n, 1.0))
                alternates["pileupDown"] = ("pileup", np_filled(pu_down, n, 1.0))
            record("pileup", True, "analysis.utils.corrections.get_pu_weight")
        except Exception as exc:
            components["pileup"] = one
            record("pileup", False, "unity_fallback", f"{type(exc).__name__}: {exc}")
    else:
        components["pileup"] = one
        record("pileup", False, "unity_fallback_missing_Pileup_nTrueInt")

    top_pt_sf = one.copy()
    if "TTto" in dataset and has_field(arrays, "GenPart_pt") and has_field(arrays, "GenPart_pdgId") and has_field(arrays, "GenPart_statusFlags"):
        try:
            gen_pt = arrays["GenPart_pt"]
            gen_pdg = arrays["GenPart_pdgId"]
            gen_flags = ak.values_astype(arrays["GenPart_statusFlags"], np.int64)
            is_top = (abs(gen_pdg) == 6) & ((gen_flags & (1 << 8)) != 0) & ((gen_flags & (1 << 13)) != 0)
            tops = gen_pt[is_top]
            t1 = ak.fill_none(ak.pad_none(tops, 2, axis=1, clip=False)[:, 0], 0.0)
            t2 = ak.fill_none(ak.pad_none(tops, 2, axis=1, clip=False)[:, 1], 0.0)
            both = ak.num(tops, axis=1) >= 2
            with analysis_workdir(repo):
                vals = np.sqrt(corrections["get_top_pt_reweight"](t1) * corrections["get_top_pt_reweight"](t2))
            top_pt_sf = np.where(ak.to_numpy(both), np_filled(vals, n, 1.0), 1.0)
            record("top_pt_reweight", True, "analysis.utils.corrections.get_top_pt_reweight")
        except Exception as exc:
            record("top_pt_reweight", False, "unity_fallback", f"{type(exc).__name__}: {exc}")
    else:
        reason = "not_TTto_dataset" if "TTto" not in dataset else "missing_GenPart_inputs"
        record("top_pt_reweight", False, f"unity_fallback_{reason}")
    components["top_pt_reweight"] = top_pt_sf

    if has_field(arrays, "Jet_hadronFlavour"):
        try:
            caller = dataset.split("____")[0]
            btag_corrector = get_btag_corrector(repo, y, caller)
            btag = btag_corrector.btag_weight(jet_pt, jet_eta, jet_hadflav, b_med,
                                            include_variations=include_variations)
            names = [
                "btagSF", "btagSF_bc_correlatedUp", "btagSF_bc_correlatedDown", "btagSF_bc_uncorrelatedUp", "btagSF_bc_uncorrelatedDown",
                "btagSF_light_correlatedUp", "btagSF_light_correlatedDown", "btagSF_light_uncorrelatedUp", "btagSF_light_uncorrelatedDown",
            ]
            if not include_variations:
                names = names[:1]
            bvals = {name: np_filled(val, n, 1.0) for name, val in zip(names, btag)}
            components["btagSF"] = bvals["btagSF"]
            for name in names[1:]:
                alternates[name] = ("btagSF", bvals[name])
            record("btagSF", True, "analysis.utils.corrections.BTagCorrector")
        except Exception as exc:
            components["btagSF"] = one
            record("btagSF", False, "unity_fallback", f"{type(exc).__name__}: {exc}")
    else:
        components["btagSF"] = one
        record("btagSF", False, "unity_fallback_missing_Jet_hadronFlavour")

    def add_triplet(component: str, nominal: np.ndarray, up: np.ndarray, down: np.ndarray, source: str) -> None:
        components[component] = nominal
        if include_variations:
            alternates[f"{component}Up"] = (component, up)
            alternates[f"{component}Down"] = (component, down)
        record(component, True, source)

    try:
        with analysis_workdir(repo):
            ev_nom, ev_up, ev_down = corrections["get_ele_veto_id_sf"](y, e_eta + e_delta_eta_sc, e_pt, e_phi)
            em_nom, em_up, em_down = corrections["get_ele_medium_id_sf"](y, e_eta + e_delta_eta_sc, e_pt, e_phi)
        # The standard EGM payload begins at 10 GeV.  Do not retain the old
        # 10 GeV edge extrapolation for 5--10 GeV veto electrons.
        electron_lowpt = e_veto & (e_pt > 5.0) & (e_pt < 10.0)
        ev_nom = ak.where(electron_lowpt, ak.ones_like(ev_nom), ev_nom)
        ev_up = ak.where(electron_lowpt, ak.ones_like(ev_up), ev_up)
        ev_down = ak.where(electron_lowpt, ak.ones_like(ev_down), ev_down)
        ele_nom = one.copy(); ele_up = one.copy(); ele_down = one.copy()
        mask_one = np.asarray(n_e_veto == 1, dtype=bool)
        mask_two = np.asarray(n_e_med == 2, dtype=bool)
        vals = [
            (ele_nom, jagged_prod(ak.where(e_veto, ev_nom, ak.ones_like(e_pt)), n), jagged_prod(ak.where(e_med, em_nom, ak.ones_like(e_pt)), n)),
            (ele_up, jagged_prod(ak.where(e_veto, ev_up, ak.ones_like(e_pt)), n), jagged_prod(ak.where(e_med, em_up, ak.ones_like(e_pt)), n)),
            (ele_down, jagged_prod(ak.where(e_veto, ev_down, ak.ones_like(e_pt)), n), jagged_prod(ak.where(e_med, em_down, ak.ones_like(e_pt)), n)),
        ]
        for target, veto_vals, med_vals in vals:
            target[mask_one] = veto_vals[mask_one]
            target[mask_two] = med_vals[mask_two]
        add_triplet("electron_id", ele_nom, ele_up, ele_down, "analysis.utils.corrections electron ID SF")
    except Exception as exc:
        components["electron_id"] = one
        record("electron_id", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    try:
        with analysis_workdir(repo):
            er_nom, er_up, er_down = corrections["get_ele_reco_sf"](
                y,
                e_eta + e_delta_eta_sc,
                e_pt,
                e_phi,
            )
        ele_nom = one.copy(); ele_up = one.copy(); ele_down = one.copy()
        mask_one = np.asarray(n_e_veto == 1, dtype=bool)
        mask_two = np.asarray(n_e_med == 2, dtype=bool)
        vals = [
            (ele_nom, jagged_prod(ak.where(e_veto, er_nom, ak.ones_like(e_pt)), n), jagged_prod(ak.where(e_med, er_nom, ak.ones_like(e_pt)), n)),
            (ele_up, jagged_prod(ak.where(e_veto, er_up, ak.ones_like(e_pt)), n), jagged_prod(ak.where(e_med, er_up, ak.ones_like(e_pt)), n)),
            (ele_down, jagged_prod(ak.where(e_veto, er_down, ak.ones_like(e_pt)), n), jagged_prod(ak.where(e_med, er_down, ak.ones_like(e_pt)), n)),
        ]
        for target, veto_vals, med_vals in vals:
            target[mask_one] = veto_vals[mask_one]
            target[mask_two] = med_vals[mask_two]
        add_triplet(
            "electron_reco",
            ele_nom,
            ele_up,
            ele_down,
            "analysis.utils.corrections.get_ele_reco_sf",
        )
    except Exception as exc:
        components["electron_reco"] = one
        record("electron_reco", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    try:
        if "veto_electron_5to10" not in enabled_analysis_sf:
            raise RuntimeError("disabled by analysis SF configuration")
        low_nom, low_up, low_down = veto_electron_lowpt_triplet(
            repo,
            e_eta,
            e_pt,
            year=y,
        )
        lowpt_mask = e_veto & (e_pt > 5.0) & (e_pt < 10.0)
        mask_one = np.asarray(n_e_veto == 1, dtype=bool)
        low_events = [
            jagged_prod(ak.where(lowpt_mask, values, ak.ones_like(e_pt)), n)
            for values in (low_nom, low_up, low_down)
        ]
        event_nom = one.copy(); event_up = one.copy(); event_down = one.copy()
        for target, values in zip((event_nom, event_up, event_down), low_events):
            target[mask_one] = values[mask_one]
        add_triplet(
            "veto_electron_5to10",
            event_nom,
            event_up,
            event_down,
            f"analysis/data/AnalysisSF/{y}/veto_electron_5to10_sf.json.gz:veto_electron_id_5to10_sf:raw_eta",
        )
    except Exception as exc:
        components["veto_electron_5to10"] = one
        record("veto_electron_5to10", False, "unity_missing_measurement", f"{type(exc).__name__}: {exc}")

    try:
        with analysis_workdir(repo):
            eh_nom, eh_up, eh_down = corrections["get_ele_hlt_sf"](y, e_eta, e_pt, e_phi)
        mask_two = np.asarray(n_e_med == 2, dtype=bool)
        nom = one.copy(); up = one.copy(); down = one.copy()
        nom_vals = jagged_prod(ak.where(e_med, eh_nom, ak.ones_like(e_pt)), n)
        up_vals = jagged_prod(ak.where(e_med, eh_up, ak.ones_like(e_pt)), n)
        down_vals = jagged_prod(ak.where(e_med, eh_down, ak.ones_like(e_pt)), n)
        nom[mask_two] = nom_vals[mask_two]; up[mask_two] = up_vals[mask_two]; down[mask_two] = down_vals[mask_two]
        add_triplet("electron_hlt", nom, up, down, "analysis.utils.corrections.get_ele_hlt_sf")
    except Exception as exc:
        components["electron_hlt"] = one
        record("electron_hlt", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    try:
        with analysis_workdir(repo):
            ml_nom, ml_up, ml_down = corrections["get_mu_loose_id_sf"](y, m_eta, m_pt)
            mm_nom, mm_up, mm_down = corrections["get_mu_medium_id_sf"](y, m_eta, m_pt)
        # The standard correction used here begins at 10 GeV.  Remove its
        # edge extrapolation in 5--10 GeV and apply the measured ID-only SF
        # below as an independent component.
        muon_lowpt = m_loose & (m_pt > 5.0) & (m_pt < 10.0)
        ml_nom = ak.where(muon_lowpt, ak.ones_like(ml_nom), ml_nom)
        ml_up = ak.where(muon_lowpt, ak.ones_like(ml_up), ml_up)
        ml_down = ak.where(muon_lowpt, ak.ones_like(ml_down), ml_down)
        mu_nom = one.copy(); mu_up = one.copy(); mu_down = one.copy()
        mask_one = np.asarray(n_m_loose == 1, dtype=bool)
        mask_two = np.asarray(n_m_med == 2, dtype=bool)
        vals = [
            (mu_nom, jagged_prod(ak.where(m_loose, ml_nom, ak.ones_like(m_pt)), n), jagged_prod(ak.where(m_med, mm_nom, ak.ones_like(m_pt)), n)),
            (mu_up, jagged_prod(ak.where(m_loose, ml_up, ak.ones_like(m_pt)), n), jagged_prod(ak.where(m_med, mm_up, ak.ones_like(m_pt)), n)),
            (mu_down, jagged_prod(ak.where(m_loose, ml_down, ak.ones_like(m_pt)), n), jagged_prod(ak.where(m_med, mm_down, ak.ones_like(m_pt)), n)),
        ]
        for target, loose_vals, med_vals in vals:
            target[mask_one] = loose_vals[mask_one]
            target[mask_two] = med_vals[mask_two]
        add_triplet("muon_id", mu_nom, mu_up, mu_down, "analysis.utils.corrections muon ID SF")
    except Exception as exc:
        components["muon_id"] = one
        record("muon_id", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    try:
        with analysis_workdir(repo):
            ml_nom, ml_up, ml_down = corrections["get_mu_loose_miniiso_sf"](y, m_eta, m_pt)
            mm_nom, mm_up, mm_down = corrections["get_mu_medium_miniiso_sf"](y, m_eta, m_pt)
        # No official mini-isolation SF is defined below 10 GeV.  The
        # dedicated 5--10 GeV payload is ID-only, so mini-isolation stays
        # exactly unity in that interval rather than inheriting the 10 GeV bin.
        muon_lowpt = m_loose & (m_pt > 5.0) & (m_pt < 10.0)
        ml_nom = ak.where(muon_lowpt, ak.ones_like(ml_nom), ml_nom)
        ml_up = ak.where(muon_lowpt, ak.ones_like(ml_up), ml_up)
        ml_down = ak.where(muon_lowpt, ak.ones_like(ml_down), ml_down)
        mu_nom = one.copy(); mu_up = one.copy(); mu_down = one.copy()
        mask_one = np.asarray(n_m_loose == 1, dtype=bool)
        mask_two = np.asarray(n_m_med == 2, dtype=bool)
        vals = [
            (mu_nom, jagged_prod(ak.where(m_loose, ml_nom, ak.ones_like(m_pt)), n), jagged_prod(ak.where(m_med, mm_nom, ak.ones_like(m_pt)), n)),
            (mu_up, jagged_prod(ak.where(m_loose, ml_up, ak.ones_like(m_pt)), n), jagged_prod(ak.where(m_med, mm_up, ak.ones_like(m_pt)), n)),
            (mu_down, jagged_prod(ak.where(m_loose, ml_down, ak.ones_like(m_pt)), n), jagged_prod(ak.where(m_med, mm_down, ak.ones_like(m_pt)), n)),
        ]
        for target, loose_vals, med_vals in vals:
            target[mask_one] = loose_vals[mask_one]
            target[mask_two] = med_vals[mask_two]
        add_triplet(
            "muon_iso",
            mu_nom,
            mu_up,
            mu_down,
            "analysis.utils.corrections muon mini-isolation SF",
        )
    except Exception as exc:
        components["muon_iso"] = one
        record("muon_iso", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    try:
        if "loose_muon_5to10" not in enabled_analysis_sf:
            raise RuntimeError("disabled by analysis SF configuration")
        low_nom, low_up, low_down = loose_muon_lowpt_triplet(
            repo,
            m_eta,
            m_pt,
            year=y,
        )
        lowpt_mask = m_loose & (m_pt > 5.0) & (m_pt < 10.0)
        mask_one = np.asarray(n_m_loose == 1, dtype=bool)
        low_events = [
            jagged_prod(ak.where(lowpt_mask, values, ak.ones_like(m_pt)), n)
            for values in (low_nom, low_up, low_down)
        ]
        event_nom = one.copy(); event_up = one.copy(); event_down = one.copy()
        for target, values in zip((event_nom, event_up, event_down), low_events):
            target[mask_one] = values[mask_one]
        add_triplet(
            "loose_muon_5to10",
            event_nom,
            event_up,
            event_down,
            f"analysis/data/AnalysisSF/{y}/loose_muon_5to10_sf.json.gz:loose_muon_id_5to10_sf",
        )
    except Exception as exc:
        components["loose_muon_5to10"] = one
        record("loose_muon_5to10", False, "unity_missing_measurement", f"{type(exc).__name__}: {exc}")

    try:
        with analysis_workdir(repo):
            mh_nom, mh_up, mh_down = corrections["get_mu_hlt_sf"](y, m_eta, m_pt)
        mask_two = np.asarray(n_m_med == 2, dtype=bool)
        nom = one.copy(); up = one.copy(); down = one.copy()
        nom_vals = jagged_prod(ak.where(m_med, mh_nom, ak.ones_like(m_pt)), n)
        up_vals = jagged_prod(ak.where(m_med, mh_up, ak.ones_like(m_pt)), n)
        down_vals = jagged_prod(ak.where(m_med, mh_down, ak.ones_like(m_pt)), n)
        nom[mask_two] = nom_vals[mask_two]; up[mask_two] = up_vals[mask_two]; down[mask_two] = down_vals[mask_two]
        add_triplet("muon_hlt", nom, up, down, "analysis.utils.corrections.get_mu_hlt_sf")
    except Exception as exc:
        components["muon_hlt"] = one
        record("muon_hlt", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    try:
        with analysis_workdir(repo):
            ph_nom, ph_up, ph_down = corrections["get_photon_id_sf"](
                y,
                photon_id_wp,
                p_eta,
                p_pt,
                p_phi,
            )
        nom = one.copy(); up = one.copy(); down = one.copy()
        mask_g = np.asarray(gcr_mask, dtype=bool)
        nom_vals = jagged_prod(ak.where(p_med, ph_nom, ak.ones_like(p_pt)), n)
        up_vals = jagged_prod(ak.where(p_med, ph_up, ak.ones_like(p_pt)), n)
        down_vals = jagged_prod(ak.where(p_med, ph_down, ak.ones_like(p_pt)), n)
        nom[mask_g] = nom_vals[mask_g]; up[mask_g] = up_vals[mask_g]; down[mask_g] = down_vals[mask_g]
        add_triplet(
            "photon_id",
            nom,
            up,
            down,
            f"analysis.utils.corrections.get_photon_id_sf:{photon_id_wp}",
        )
    except Exception as exc:
        components["photon_id"] = one
        record("photon_id", False, "unity_fallback", f"{type(exc).__name__}: {exc}")

    try:
        if p_r9 is None:
            raise RuntimeError("missing Photon_r9 input")
        # The flat ntuple keeps all NanoAOD photons, including unselected
        # objects whose R9 sentinel can lie outside the CSEV payload domain.
        # Evaluate those irrelevant objects at an in-domain neutral point;
        # only p_med entries are multiplied into the event weight below.
        csev_eta = ak.where(p_med, p_eta, ak.zeros_like(p_eta))
        csev_r9 = ak.where(p_med, p_r9, ak.ones_like(p_r9))
        csev_pt = ak.where(p_med, p_pt, ak.ones_like(p_pt) * 250.0)
        with analysis_workdir(repo):
            pc_nom, pc_up, pc_down = corrections["get_photon_csev_sf"](
                y,
                photon_id_wp,
                csev_eta,
                csev_r9,
                csev_pt,
            )
        nom = one.copy(); up = one.copy(); down = one.copy()
        mask_g = np.asarray(gcr_mask, dtype=bool)
        nom_vals = jagged_prod(ak.where(p_med, pc_nom, ak.ones_like(p_pt)), n)
        up_vals = jagged_prod(ak.where(p_med, pc_up, ak.ones_like(p_pt)), n)
        down_vals = jagged_prod(ak.where(p_med, pc_down, ak.ones_like(p_pt)), n)
        nom[mask_g] = nom_vals[mask_g]; up[mask_g] = up_vals[mask_g]; down[mask_g] = down_vals[mask_g]
        add_triplet(
            "photon_csev",
            nom,
            up,
            down,
            f"analysis.utils.corrections.get_photon_csev_sf:{photon_id_wp}",
        )
    except Exception as exc:
        components["photon_csev"] = one
        record("photon_csev", False, "unity_unavailable", f"{type(exc).__name__}: {exc}")

    try:
        if "photon_trigger" not in enabled_analysis_sf:
            raise RuntimeError("disabled by analysis SF configuration")
        ph_nom, ph_up, ph_down = photon_trigger_triplet(
            repo,
            p_eta,
            p_pt,
            year=y,
        )
        mask_g = np.asarray(gcr_mask, dtype=bool)
        event_values = [
            jagged_prod(ak.where(p_med, values, ak.ones_like(p_pt)), n)
            for values in (ph_nom, ph_up, ph_down)
        ]
        event_nom = one.copy(); event_up = one.copy(); event_down = one.copy()
        for target, values in zip((event_nom, event_up, event_down), event_values):
            target[mask_g] = values[mask_g]
        add_triplet(
            "photon_trigger",
            event_nom,
            event_up,
            event_down,
            f"analysis/data/AnalysisSF/{y}/photon_trigger_sf.json.gz",
        )
    except Exception as exc:
        components["photon_trigger"] = one
        record("photon_trigger", False, "unity_missing_measurement", f"{type(exc).__name__}: {exc}")

    if met_pt is None or met_trigger_mask is None:
        components["met_trigger"] = one
        record("met_trigger", False, "unity_missing_region_mask")
    else:
        try:
            if "met_trigger" not in enabled_analysis_sf:
                raise RuntimeError("disabled by analysis SF configuration")
            mt_nom, mt_up, mt_down = met_trigger_triplet(
                repo,
                met_pt,
                qcd=(process == "QCD" or dataset.startswith("QCD")),
                year=y,
            )
            mask_met = np.asarray(met_trigger_mask, dtype=bool)
            event_nom = one.copy(); event_up = one.copy(); event_down = one.copy()
            event_nom[mask_met] = mt_nom[mask_met]
            event_up[mask_met] = mt_up[mask_met]
            event_down[mask_met] = mt_down[mask_met]
            add_triplet(
                "met_trigger",
                event_nom,
                event_up,
                event_down,
                f"analysis/data/AnalysisSF/{y}/met_trigger_sf.json.gz",
            )
        except Exception as exc:
            components["met_trigger"] = one
            record("met_trigger", False, "unity_missing_measurement", f"{type(exc).__name__}: {exc}")

    # Evaluate the common bundle above, then explicitly remove every
    # analysis-owned component that this production did not adopt.  This also
    # removes its Up/Down shapes.  For disabled low-pT lepton components the
    # official POG 10 GeV edge extrapolation remains suppressed, so the net
    # 5--10 GeV lepton factor is exactly unity.
    for component in REQUIRED_ANALYSIS_SF_COMPONENTS:
        if component in enabled_analysis_sf:
            continue
        components[component] = one
        alternates.pop(f"{component}Up", None)
        alternates.pop(f"{component}Down", None)
        record(component, False, "disabled_by_analysis_sf_configuration")

    protection_counts: dict[str, int] = {}

    def clean_weight(name: str, values: Any) -> np.ndarray:
        try:
            raw = np.asarray(values, dtype=float)
        except Exception:
            protection_counts[name] = protection_counts.get(name, 0) + n
            return np.zeros(n, dtype=float)
        if raw.shape == ():
            raw = np.full(n, finite_float(raw, 0.0), dtype=float)
        if len(raw) != n:
            protection_counts[name] = protection_counts.get(name, 0) + n
            return np.zeros(n, dtype=float)
        bad = ~np.isfinite(raw)
        if np.any(bad):
            protection_counts[name] = protection_counts.get(name, 0) + int(np.sum(bad))
        return np.where(bad, 0.0, raw).astype(float)

    nominal_sf = np.ones(n, dtype=float)
    for component_name, comp in components.items():
        factor = finite_weight_array(comp, n, 1.0)
        with np.errstate(over="ignore", invalid="ignore"):
            nominal_sf = nominal_sf * factor
        bad = ~np.isfinite(nominal_sf)
        if np.any(bad):
            protection_counts[f"nominal_after_{component_name}"] = protection_counts.get(f"nominal_after_{component_name}", 0) + int(np.sum(bad))
            nominal_sf = np.where(bad, 0.0, nominal_sf)
    with np.errstate(over="ignore", invalid="ignore"):
        nominal = gen * nominal_sf
    variations = {"nominal": clean_weight("nominal", nominal)}
    for variation, (component, varied) in alternates.items():
        with np.errstate(over="ignore", invalid="ignore"):
            raw_variation = gen * replace_component(components, component, varied)
        variations[variation] = clean_weight(variation, raw_variation)
    if protection_counts:
        status["nonfinite_weight_protection"] = protection_counts
    status["available_variations"] = sorted(variations)
    return gen, variations, status


def is_data_process(process: str) -> bool:
    return process in {"JetMET", "EGamma", "Muon"}


def has_field(arrays: Any, name: str) -> bool:
    return name in getattr(arrays, "fields", [])


def arr(arrays: dict[str, Any], name: str, default: Any = None) -> Any:
    return arrays[name] if has_field(arrays, name) else default
