import awkward as ak
import numpy as np
import uproot

from cms_tnp.count import count_files
from cms_tnp.profiles import resolve_profile


def _base_config(profile):
    config = resolve_profile(
        {
            "schema_version": 1,
            "profile": profile,
            "measurement": f"test_{profile}",
            "year": "2025",
            "samples": {"data": ["data"], "mc": ["mc"]},
            "input": {
                "tree": "Events",
                "event_id": ["run", "luminosityBlock", "event"],
                "event_filters": [],
            },
            "reference_trigger": {
                "paths": [],
                "apply_to_data": False,
                "apply_to_mc": False,
                "match_tag": False,
            },
        }
    )
    config.pop("lumimask", None)
    return config


def _events(size):
    return {
        "run": np.full(size, 1, dtype=np.int64),
        "luminosityBlock": np.full(size, 1, dtype=np.int64),
        "event": np.arange(size, dtype=np.int64),
        "genWeight": np.ones(size),
    }


def test_photon_tag_and_probe_uses_separate_collections(tmp_path):
    size = 10
    branches = _events(size)
    branches.update(
        {
            "Electron_pt": ak.Array([[45.0]] * size),
            "Electron_eta": ak.Array([[0.0]] * size),
            "Electron_phi": ak.Array([[0.0]] * size),
            "Electron_mass": ak.Array([[0.0005]] * size),
            "Electron_charge": ak.Array([[-1]] * size),
            "Electron_deltaEtaSC": ak.Array([[0.0]] * size),
            "Electron_cutBased": ak.Array([[4]] * size),
            "Electron_miniPFRelIso_all": ak.Array([[0.01]] * size),
            "Electron_convVeto": ak.Array([[True]] * size),
            "Electron_lostHits": ak.Array([[0]] * size),
            "Photon_pt": ak.Array([[45.0]] * size),
            "Photon_eta": ak.Array([[0.0]] * size),
            "Photon_phi": ak.Array([[np.pi]] * size),
            "Photon_mass": ak.Array([[0.0]] * size),
            "Photon_deltaEtaSC": ak.Array([[0.0]] * size),
            "Photon_cutBased": ak.Array(
                [[3 if index % 2 else 0] for index in range(size)]
            ),
            "Photon_pixelSeed": ak.Array([[False]] * size),
            "Photon_electronVeto": ak.Array([[True]] * size),
        }
    )
    path = tmp_path / "photon.root"
    with uproot.recreate(path) as target:
        target["Events"] = branches
    config = _base_config("photon_z")
    output = count_files(config, [str(path)], "mc", base_dir=tmp_path)
    assert output["processing"]["files_processed"] == 1
    assert output["processing"]["pairs_selected"] == 10
    assert np.sum(output["samples"]["mc"]["pass_sumw"]) == 5
    assert np.sum(output["samples"]["mc"]["fail_sumw"]) == 5


def test_lowpt_muon_profile_requires_distinct_spectator(tmp_path):
    size = 8
    branches = _events(size)
    branches.update(
        {
            "Muon_pt": ak.Array([[6.0, 6.0, 13.0]] * size),
            "Muon_eta": ak.Array([[0.0, 0.0, 0.0]] * size),
            "Muon_phi": ak.Array([[0.0, 0.522, 2.0]] * size),
            "Muon_mass": ak.Array([[0.105, 0.105, 0.105]] * size),
            "Muon_charge": ak.Array([[-1, 1, 1]] * size),
            "Muon_looseId": ak.Array(
                [[True, index % 2 == 0, True] for index in range(size)]
            ),
            "Muon_tightId": ak.Array([[True, False, True]] * size),
            "Muon_isTracker": ak.Array([[True, True, True]] * size),
            "Muon_miniPFRelIso_all": ak.Array([[0.0, 0.0, 0.0]] * size),
        }
    )
    path = tmp_path / "muon.root"
    with uproot.recreate(path) as target:
        target["Events"] = branches
    config = _base_config("muon_jpsi_lowpt")
    output = count_files(config, [str(path)], "mc", base_dir=tmp_path)
    assert output["processing"]["files_processed"] == 1
    assert output["processing"]["pairs_selected"] == 8
    assert np.sum(output["samples"]["mc"]["pass_sumw"]) == 4
    assert np.sum(output["samples"]["mc"]["fail_sumw"]) == 4


def test_lowpt_electron_profile_counts_jpsi_pairs(tmp_path):
    size = 8
    branches = _events(size)
    branches.update(
        {
            "Electron_pt": ak.Array([[6.0, 6.0]] * size),
            "Electron_eta": ak.Array([[0.0, 0.0]] * size),
            "Electron_phi": ak.Array([[0.0, 0.522]] * size),
            "Electron_mass": ak.Array([[0.0005, 0.0005]] * size),
            "Electron_charge": ak.Array([[-1, 1]] * size),
            "Electron_deltaEtaSC": ak.Array([[0.0, 0.0]] * size),
            "Electron_cutBased": ak.Array(
                [[4, 1 if index % 2 == 0 else 0] for index in range(size)]
            ),
            "Electron_miniPFRelIso_all": ak.Array([[0.01, 0.5]] * size),
            "Electron_convVeto": ak.Array([[True, True]] * size),
            "Electron_lostHits": ak.Array([[0, 0]] * size),
        }
    )
    path = tmp_path / "electron.root"
    with uproot.recreate(path) as target:
        target["Events"] = branches
    output = count_files(
        _base_config("electron_jpsi_lowpt"), [str(path)], "mc", base_dir=tmp_path
    )
    assert output["processing"]["pairs_selected"] == 8
    assert np.sum(output["samples"]["mc"]["pass_sumw"]) == 4
    assert np.sum(output["samples"]["mc"]["fail_sumw"]) == 4


def test_highpt_muon_profile_counts_z_pairs(tmp_path):
    size = 8
    branches = _events(size)
    branches.update(
        {
            "Muon_pt": ak.Array([[50.0, 45.0]] * size),
            "Muon_eta": ak.Array([[0.0, 0.0]] * size),
            "Muon_phi": ak.Array([[0.0, np.pi]] * size),
            "Muon_mass": ak.Array([[0.105, 0.105]] * size),
            "Muon_charge": ak.Array([[-1, 1]] * size),
            "Muon_looseId": ak.Array([[True, True]] * size),
            "Muon_tightId": ak.Array([[True, index % 2 == 0] for index in range(size)]),
            "Muon_isTracker": ak.Array([[True, True]] * size),
            "Muon_miniPFRelIso_all": ak.Array([[0.01, 0.5]] * size),
        }
    )
    path = tmp_path / "muon_z.root"
    with uproot.recreate(path) as target:
        target["Events"] = branches
    output = count_files(_base_config("muon_z"), [str(path)], "mc", base_dir=tmp_path)
    assert output["processing"]["pairs_selected"] == 8
    assert np.sum(output["samples"]["mc"]["pass_sumw"]) == 4
    assert np.sum(output["samples"]["mc"]["fail_sumw"]) == 4
