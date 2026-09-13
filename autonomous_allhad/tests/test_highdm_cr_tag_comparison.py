from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from test_highdm_veto_pt_threshold import MODULE, study_chunk, jagged


def test_tag_union_and_strict_missing_counts():
    chunk = {"nboosted_top": [0, 1, 0, 0], "nboosted_w": [0, 0, 1, 0],
             MODULE.DERIVED_NRES_BRANCH: [0, 0, 0, 1]}
    assert MODULE.highdm_cr_tag_mask(chunk, 4).tolist() == [False, True, True, True]
    for invalid in ([-1, 0, 0, 1], [0, np.nan, 0, 1], [0, 0.5, 0, 1], [0]):
        chunk[MODULE.DERIVED_NRES_BRANCH] = invalid
        with pytest.raises(RuntimeError, match="Invalid"):
            MODULE.highdm_cr_tag_mask(chunk, 4)
    del chunk[MODULE.DERIVED_NRES_BRANCH]
    with pytest.raises(RuntimeError, match="requires"):
        MODULE.highdm_cr_tag_mask(chunk, 4)


def test_cr_only_selection_preserves_weights_variations_and_sr(monkeypatch):
    chunk = {"nboosted_top": [0, 1, 0, 0], "nboosted_w": [0, 0, 1, 0],
             MODULE.DERIVED_NRES_BRANCH: [0, 0, 0, 1], "met": [300.] * 4}
    monkeypatch.setattr(MODULE, "highdm_distribution_masks",
                        lambda c, n: {"LLCR": np.ones(n, bool), "SR_test": np.ones(n, bool)})
    monkeypatch.setattr(MODULE, "HIGHDM_DISTRIBUTION_VARIABLE_SPECS",
                        {"met": {"branch": "met", "bins": [250., 400.]}})
    weights = np.asarray([2., -1., 3., 4.])
    variations = {"nominal": weights, "testUp": 2 * weights, "testDown": .5 * weights}
    out, audit = {}, {}
    MODULE.fill_highdm_distribution_histograms(
        chunk, variations, np.ones(4), "TT", "TT", False, out, audit,
        require_highdm_cr_tag=True)
    leaf = out["LLCR"]["met"]["TT"]
    assert leaf["nominal"]["sumw"] == [6.]
    assert leaf["nominal"]["sumw2"] == [26.]
    assert leaf["nominal"]["entries"] == [3]
    assert leaf["testUp"]["sumw"] == [12.]
    assert leaf["testDown"]["sumw"] == [3.]
    assert out["SR_test"]["met"]["TT"]["nominal"]["sumw"] == [8.]
    assert audit["highdm_cr_tag_audit"]["LLCR"]["TT"]["baseline_entries"] == 4
    assert audit["highdm_cr_tag_audit"]["LLCR"]["TT"]["rejected_entries"] == 1
    del chunk[MODULE.DERIVED_NRES_BRANCH]
    default, default_audit = {}, {}
    MODULE.fill_highdm_distribution_histograms(
        chunk, variations, np.ones(4), "TT", "TT", False, default, default_audit)
    assert default["LLCR"]["met"]["TT"]["nominal"]["sumw"] == [8.]
    assert "highdm_cr_tag_audit" not in default_audit


def test_trota_candidates_in_cr_are_not_limited_to_sr(monkeypatch):
    chunk = study_chunk()
    for key in MODULE.BROAD_LOWDM_SELECTION_BRANCHES:
        chunk.setdefault(key, np.zeros(6))
    for key in ("nboosted_w", "pass_dy2e_open_high", "pass_dy2m_open_high",
                "njet_lepton_clean", "nb_lepton_clean", "ht_lepton_clean"):
        chunk.setdefault(key, np.zeros(6))
    chunk.update({"run": np.ones(6, int), "luminosityBlock": np.ones(6, int),
                  "event": np.arange(6), "file_id": np.zeros(6, int), "entry": np.arange(6)})
    for key in MODULE.TROTA_LOWDM_OVERLAP_BRANCHES:
        chunk[key] = jagged([[]] * 6)
    candidates = {key: np.asarray([0]) for key in MODULE.TROTA_PRIMARY_BRANCHES}
    candidates.update({"entry": np.asarray([1]), "TopResolved1pct_sourceJetIdx1": np.asarray([1]),
                       "TopResolved1pct_sourceJetIdx2": np.asarray([2]),
                       "TopResolved1pct_mass": np.asarray([170.]),
                       "TopResolved1pct_QCDDiscriminant": np.asarray([.9])})

    class Tree:
        def __init__(self, values, count):
            self.values, self.num_entries = values, count

        def keys(self):
            return self.values.keys()

        def arrays(self, names, library):
            return ak.Array({name: self.values[name] for name in names})

    def cr_mask(c, region, flag, n):
        assert not c["feature_SR"][1]
        return np.arange(n) == 1 if region == "LLCR" else np.zeros(n, bool)

    monkeypatch.setattr(MODULE, "region_mask", cr_mask)
    counts, audit = MODULE.compute_trota_nres(
        Tree(chunk, 6), Tree(candidates, 1), include_lowdm=False,
        highdm_configuration=None, include_highdm_cr=True)
    assert counts.tolist() == [0, 1, 0, 0, 0, 0]
    assert audit["eligible_highdm_cr_events"] == 1
    assert audit["eligible_highdm_events"] == 0
