from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
SPEC = importlib.util.spec_from_file_location(
    "dycr_window_histograms", PROJECT / "workflow/build_flat_boosted_recoil_hists.py"
)
HIST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HIST)


def dileptons(channel):
    mass = np.array([70., 71., 71.001, 80., 81., 91., 101., 110., 110.999, 111., 112.])
    n = len(mass)
    chunk = {}
    for flavor in ("electron", "muon", "photon"):
        chosen = (flavor == "electron" and channel == "DY2E") or (flavor == "muon" and channel == "DY2M")
        def jag(values):
            flat = np.tile(np.asarray(values), n) if chosen else np.asarray(values)[:0]
            return ak.unflatten(flat, np.full(n, len(values) if chosen else 0))
        chunk[flavor + "_pt_all"] = jag([70., 30.])
        chunk[flavor + "_eta_all"] = jag([0.1, -0.2])
        chunk[flavor + "_charge_all"] = jag([1, -1])
        chunk[flavor + "_cutbased_all"] = jag([3, 3])
        chunk[flavor + "_mini_iso_all"] = jag([0., 0.])
        for suffix in ("loose_id_all", "medium_id_all", "electron_veto_all"):
            chunk[flavor + "_" + suffix] = jag([True, True])
    for name in ("n_e_medium", "n_e_veto", "n_m_medium", "n_m_loose"):
        selected = (name.startswith("n_e") and channel == "DY2E") or (name.startswith("n_m") and channel == "DY2M")
        chunk[name] = np.full(n, 2 if selected else 0)
    for name in ("pass_base_common", "pass_zero_tau", "pass_electron_trigger", "pass_muon_trigger", "pass_dy2e_open_high", "pass_dy2m_open_high"):
        chunk[name] = np.ones(n, dtype=bool)
    for name in ("mee", "mmm"):
        chunk[name] = mass.copy()
    for name in ("pee", "pmm", "recoil_dy2e", "recoil_dy2m", "ht_lepton_clean"):
        chunk[name] = np.full(n, 400.)
    chunk["njet_lepton_clean"] = np.full(n, 5)
    chunk["nb_lepton_clean"] = np.ones(n, dtype=int)
    chunk["nboosted_top"] = np.zeros(n, dtype=int)
    chunk["feature_" + channel] = (mass > 81.) & (mass < 101.)
    return chunk


@pytest.mark.parametrize("channel", ["DY2E", "DY2M"])
def test_twenty_gev_window_reselects_old_false_flags(channel):
    chunk = dileptons(channel)
    expected = [False, False, True, True, True, True, True, True, True, False, False]
    assert HIST.dycr_lepton_mask(chunk, channel).tolist() == expected
    assert HIST.region_mask(chunk, channel, "feature_" + channel, 11).tolist() == expected
    assert HIST.region_mask(chunk, channel + "_Nt0", "feature_" + channel, 11).tolist() == expected
    assert not HIST.region_mask(chunk, channel + "_Nt1", "feature_" + channel, 11).any()


@pytest.mark.parametrize("channel", ["DY2E", "DY2M"])
def test_non_mass_requirements_are_preserved(channel):
    chunk = dileptons(channel)
    chunk["pass_zero_tau"][2] = False
    chunk["pass_base_common"][3] = False
    chunk["pass_electron_trigger" if channel == "DY2E" else "pass_muon_trigger"][4] = False
    chunk["njet_lepton_clean"][6] = 4
    chunk["nb_lepton_clean"][7] = 0
    chunk["pass_" + channel.lower() + "_open_high"][8] = False
    assert HIST.region_mask(chunk, channel, "feature_" + channel, 11).tolist() == [i == 5 for i in range(11)]


@pytest.mark.parametrize("channel", ["DY2E", "DY2M"])
def test_inconsistent_stored_leptons_fail_explicitly(channel):
    chunk = dileptons(channel)
    chunk["n_e_medium" if channel == "DY2E" else "n_m_medium"][5] = 1
    with pytest.raises(RuntimeError, match="medium-lepton counts differ"):
        HIST.dycr_lepton_mask(chunk, channel)
