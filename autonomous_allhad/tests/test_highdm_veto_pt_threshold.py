from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import awkward as ak
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "workflow" / "build_flat_boosted_recoil_hists.py"
sys.path.insert(0, str(PROJECT))
SPEC = importlib.util.spec_from_file_location("highdm_veto_pt_study", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def jagged(values: list[list[float]]) -> ak.Array:
    return ak.Array(values)


def study_chunk() -> dict[str, object]:
    n = 6
    electron_pt = jagged([[7.0], [12.0], [], [], [8.0], [6.0]])
    electron_eta = jagged([[0.1], [0.2], [], [], [0.3], [0.4]])
    electron_phi = jagged([[0.0], [0.0], [], [], [0.0], [0.0]])
    muon_pt = jagged([[], [], [8.0], [], [], []])
    muon_eta = jagged([[], [], [0.2], [], [], []])
    muon_phi = jagged([[], [], [0.0], [], [], []])
    return {
        "dataset_id": np.arange(n),
        "electron_veto_pt": electron_pt,
        "electron_veto_eta": electron_eta,
        "electron_veto_eta_sc": electron_eta,
        "electron_veto_phi": electron_phi,
        "muon_loose_pt": muon_pt,
        "muon_loose_eta": muon_eta,
        "muon_loose_phi": muon_phi,
        "n_e_veto": np.asarray([1, 1, 0, 0, 1, 1]),
        "n_m_loose": np.asarray([0, 0, 1, 0, 0, 0]),
        "feature_SR": np.asarray([False, False, False, True, False, False]),
        "feature_LLCR": np.asarray([True, True, True, False, False, False]),
        "feature_QCDCR": np.zeros(n, dtype=bool),
        "feature_GCR": np.zeros(n, dtype=bool),
        "pass_base_common": np.ones(n, dtype=bool),
        "pass_zero_tau": np.ones(n, dtype=bool),
        "pass_signal_trigger": np.asarray([True, True, True, True, False, True]),
        "pass_photon_trigger": np.asarray([False, False, False, False, True, False]),
        "njet": np.full(n, 5),
        "nb_medium": np.ones(n, dtype=int),
        "pass_met_250": np.asarray([True, True, True, True, False, True]),
        "pass_ht_300": np.ones(n, dtype=bool),
        "pass_open_high": np.asarray([True, True, True, True, True, False]),
        "pass_qcd_open": np.asarray([False, False, False, False, False, True]),
        "pass_dphi123_0p1": np.asarray([False, False, False, False, False, True]),
        "met": np.asarray([300.0, 300.0, 300.0, 300.0, 100.0, 300.0]),
        "met_phi": np.zeros(n),
        "n_photon_medium": np.asarray([0, 0, 0, 0, 1, 0]),
        "photon_medium_eta": jagged([[], [], [], [], [0.3], []]),
        "photon_medium_phi": jagged([[], [], [], [], [2.5], []]),
        "good_jet_eta": jagged([[1.0] * 5] * n),
        "good_jet_phi": jagged([[0.7, 0.8, 0.9, 1.0, 1.1]] * n),
        "njet_photon_clean": np.asarray([0, 0, 0, 0, 5, 0]),
        "nb_photon_clean": np.asarray([0, 0, 0, 0, 1, 0]),
        "recoil_gcr": np.asarray([0.0, 0.0, 0.0, 0.0, 300.0, 0.0]),
        "recoil_gcr_phi": np.zeros(n),
        "pass_gcr_open_high": np.ones(n, dtype=bool),
        "pass_ht_photon_300": np.ones(n, dtype=bool),
        "nboosted_top": np.zeros(n, dtype=int),
    }


def test_nominal_threshold_is_a_strict_noop() -> None:
    chunk = {"dataset_id": np.arange(3)}
    audit = MODULE.apply_highdm_veto_pt_thresholds(chunk, 5.0, 5.0)
    assert audit == {"events": 3, "recomputed": 0}
    assert set(chunk) == {"dataset_id"}


def test_trota_light_input_rebuilds_veto_before_candidate_eligibility(monkeypatch) -> None:
    chunk = study_chunk()
    for key in MODULE.TROTA_LOWDM_SELECTION_BRANCHES:
        chunk.setdefault(key, np.zeros(6))
    chunk.update({"run": np.ones(6, dtype=int), "luminosityBlock": np.ones(6, dtype=int),
                  "event": np.arange(6), "file_id": np.zeros(6, dtype=int),
                  "entry": np.arange(6), "nboosted_w": np.zeros(6, dtype=int),
                  "nboosted_total": np.zeros(6, dtype=int), "lowdm_mtb": np.full(6, 300.0)})
    for key in MODULE.TROTA_LOWDM_OVERLAP_BRANCHES:
        chunk[key] = jagged([[]] * 6)
    candidates = {key: np.asarray([0]) for key in MODULE.TROTA_PRIMARY_BRANCHES}
    candidates.update({"TopResolved1pct_sourceJetIdx1": np.asarray([1]),
                       "TopResolved1pct_sourceJetIdx2": np.asarray([2]),
                       "TopResolved1pct_mass": np.asarray([170.0]),
                       "TopResolved1pct_QCDDiscriminant": np.asarray([0.9])})

    class Tree:
        def __init__(self, values, count):
            self.values, self.num_entries = values, count

        def keys(self):
            return self.values.keys()

        def arrays(self, names, library):
            return ak.Array({name: self.values[name] for name in names})

    def lowdm_blocks(arrays):
        assert ak.to_list(arrays.electron_veto_pt)[0] == []
        assert bool(arrays.pass_no_veto_leptons[0])
        return {}, {}

    monkeypatch.setattr(MODULE, "broad_lowdm_blocks", lowdm_blocks)
    monkeypatch.setattr(MODULE, "selected_an17_recoil60_indices",
                        lambda arrays, n, sr: np.where(sr, 0, -1))
    monkeypatch.setattr(MODULE, "map60_indices_to_adopted55", lambda values: values)
    for include_lowdm in (False, True):
        counts, audit = MODULE.compute_trota_nres(
            Tree(chunk, 6), Tree(candidates, 1), include_lowdm=include_lowdm,
            highdm_configuration={"mtb_min": 175.0},
        )
        assert counts.tolist() == [1, 0, 0, 0, 0, 0]
        assert audit["nres_positive_events"] == 1


def test_ten_gev_threshold_reassigns_only_low_pt_veto_leptons() -> None:
    chunk = study_chunk()
    audit = MODULE.apply_highdm_veto_pt_thresholds(chunk, 10.0, 10.0)

    assert ak.to_list(chunk["electron_veto_pt"]) == [[], [12.0], [], [], [], []]
    assert ak.to_list(chunk["muon_loose_pt"]) == [[], [], [], [], [], []]
    assert np.asarray(chunk["feature_SR"]).tolist() == [True, False, True, True, False, False]
    assert np.asarray(chunk["feature_LLCR"]).tolist() == [False, True, False, False, False, False]
    assert np.asarray(chunk["feature_QCDCR"]).tolist() == [False, False, False, False, False, True]
    assert np.asarray(chunk["feature_GCR"]).tolist() == [False, False, False, False, True, False]
    assert audit["events_with_removed_electrons"] == 3
    assert audit["events_with_removed_muons"] == 1
    assert audit["SR_gained"] == 2
    assert audit["LLCR_lost"] == 2
    assert audit["QCDCR_gained"] == 1
    assert audit["GCR_gained"] == 1
    assert np.asarray(chunk["pass_no_veto_leptons"]).tolist() == [True, False, True, True, True, True]
    assert np.asarray(chunk["pass_one_veto_lepton"]).tolist() == [False, True, False, False, False, False]
    assert np.asarray(chunk["pass_mt_100"]).all()


def test_shared_veto_threshold_is_strict_and_updates_awkward_in_place() -> None:
    from gnn_lowdm._implementation.region_io import update_veto_leptons

    arrays = ak.Array({
        "electron_veto_pt": [[10.0], [10.1], [7.0, 20.0]],
        "electron_veto_phi": [[0.0], [np.pi], [np.pi, 0.0]],
        "muon_loose_pt": [[], [], [8.0]],
        "muon_loose_phi": [[], [], [np.pi]],
        "n_e_veto": [1, 1, 2], "n_m_loose": [0, 0, 1],
        "met": [300.0] * 3, "met_phi": [0.0] * 3,
    })
    audit = update_veto_leptons(arrays)
    assert ak.to_list(arrays.electron_veto_pt) == [[], [10.1], [20.0]]
    assert ak.to_list(arrays.pass_no_veto_leptons) == [True, False, False]
    assert ak.to_list(arrays.pass_one_veto_lepton) == [False, True, True]
    assert ak.to_list(arrays.pass_mt_100) == [True, False, True]
    assert audit == {"events_with_removed_electrons": 2, "events_with_removed_muons": 1}
    assert update_veto_leptons(arrays) == {"events_with_removed_electrons": 0, "events_with_removed_muons": 0}


def test_electron_and_muon_thresholds_can_be_varied_independently() -> None:
    electron_only = study_chunk()
    MODULE.apply_highdm_veto_pt_thresholds(electron_only, 10.0, 5.0)
    assert np.asarray(electron_only["feature_SR"]).tolist() == [
        True, False, False, True, False, False
    ]
    assert np.asarray(electron_only["feature_LLCR"]).tolist() == [
        False, True, True, False, False, False
    ]

    muon_only = study_chunk()
    MODULE.apply_highdm_veto_pt_thresholds(muon_only, 5.0, 10.0)
    assert np.asarray(muon_only["feature_SR"]).tolist() == [
        False, False, True, True, False, False
    ]
    assert np.asarray(muon_only["feature_LLCR"]).tolist() == [
        True, True, False, False, False, False
    ]


def test_legacy_flat_input_reconstructs_gcr_open_high_from_vectors() -> None:
    chunk = study_chunk()
    del chunk["pass_gcr_open_high"]
    MODULE.apply_highdm_veto_pt_thresholds(chunk, 10.0, 10.0)
    assert np.asarray(chunk["feature_GCR"]).tolist() == [
        False, False, False, False, True, False
    ]


def test_removed_objects_are_absent_from_scale_factor_inputs() -> None:
    chunk = study_chunk()
    MODULE.apply_highdm_veto_pt_thresholds(chunk, 10.0, 10.0)

    chunk.update(
        {
            "gen_weight": np.ones(6),
            "good_jet_pt": jagged([[]] * 6),
            "good_jet_eta": jagged([[]] * 6),
            "good_jet_hadron_flavour": jagged([[]] * 6),
            "good_jet_b_medium": jagged([[]] * 6),
            "electron_medium_pt": jagged([[]] * 6),
            "electron_medium_eta": jagged([[]] * 6),
            "electron_medium_eta_sc": jagged([[]] * 6),
            "electron_medium_phi": jagged([[]] * 6),
            "n_e_medium": np.zeros(6, dtype=int),
            "muon_medium_pt": jagged([[]] * 6),
            "muon_medium_eta": jagged([[]] * 6),
            "muon_medium_phi": jagged([[]] * 6),
            "n_m_medium": np.zeros(6, dtype=int),
            "photon_medium_pt": jagged([[]] * 6),
            "photon_medium_eta": jagged([[]] * 6),
            "photon_medium_phi": jagged([[]] * 6),
            "gen_top_pt": jagged([[]] * 6),
            "pu_ntrueint": np.zeros(6),
            "feature_lowdm_GCR": np.zeros(6, dtype=bool),
            "feature_lowdm_LLCR": np.zeros(6, dtype=bool),
            "feature_lowdm_QCDCR": np.zeros(6, dtype=bool),
            "feature_lowdm_SR": np.zeros(6, dtype=bool),
        }
    )
    _, inputs = MODULE.flat_arrays_for_weights(chunk)
    assert ak.to_list(inputs["e_pt"]) == [[], [12.0], [], [], [], []]
    assert ak.to_list(inputs["m_pt"]) == [[]] * 6
    assert inputs["n_e_veto"].tolist() == [0, 1, 0, 0, 0, 0]
    assert inputs["n_m_loose"].tolist() == [0] * 6
