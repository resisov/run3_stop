import json

import awkward as ak
import correctionlib
import numpy as np
import uproot

from cms_tnp.count import count_files
from cms_tnp.fit import fit_payload
from cms_tnp.payload import build_payload, write_payload
from cms_tnp.profiles import resolve_profile
from cms_tnp.reduce import merge


def _root(path, efficiency, seed):
    rng = np.random.default_rng(seed)
    size = 12_000
    signal = rng.random(size) < 0.85
    masses = np.where(
        signal, rng.normal(3.096, 0.04, size), rng.uniform(2.6, 3.6, size)
    )
    passing = rng.random(size) < np.where(signal, efficiency, 0.55)
    pt = np.full(size, 6.0)
    delta_phi = np.arccos(np.clip(1.0 - masses * masses / (2.0 * pt * pt), -1.0, 1.0))
    branches = {
        "run": np.full(size, 1, dtype=np.int64),
        "luminosityBlock": np.full(size, 1, dtype=np.int64),
        "event": np.arange(size, dtype=np.int64),
        "genWeight": np.ones(size),
        "Electron_pt": ak.Array(np.column_stack([pt, pt]).tolist()),
        "Electron_eta": ak.Array(np.zeros((size, 2)).tolist()),
        "Electron_phi": ak.Array(np.column_stack([np.zeros(size), delta_phi]).tolist()),
        "Electron_mass": ak.Array(np.full((size, 2), 0.0005).tolist()),
        "Electron_charge": ak.Array(
            np.column_stack(
                [-np.ones(size, dtype=int), np.ones(size, dtype=int)]
            ).tolist()
        ),
        "Electron_deltaEtaSC": ak.Array(np.zeros((size, 2)).tolist()),
        "Electron_cutBased": ak.Array(
            np.column_stack([np.full(size, 4), passing.astype(int)]).tolist()
        ),
        "Electron_miniPFRelIso_all": ak.Array(
            np.column_stack([np.full(size, 0.01), np.full(size, 0.5)]).tolist()
        ),
        "Electron_convVeto": ak.Array(np.ones((size, 2), dtype=bool).tolist()),
        "Electron_lostHits": ak.Array(np.zeros((size, 2), dtype=int).tolist()),
    }
    with uproot.recreate(path) as target:
        target["Events"] = branches


def test_root_to_correctionlib(tmp_path):
    data_path = tmp_path / "data.root"
    mc_path = tmp_path / "mc.root"
    _root(data_path, 0.80, 1)
    _root(mc_path, 0.85, 2)
    (tmp_path / "golden.json").write_text(json.dumps({"1": [[1, 1]]}))
    config = resolve_profile(
        {
            "schema_version": 1,
            "profile": "electron_jpsi_lowpt",
            "measurement": "root_e2e_sf",
            "year": "2025",
            "pt_edges_gev": [5, 10],
            "abseta_edges": [0.0, 2.5],
            "lumimask": "golden.json",
            "samples": {"data": [str(data_path)], "mc": [str(mc_path)]},
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
            "fit": {
                "signal_model": "gaussian",
                "alternate_signal_model": "double_gaussian",
                "background_model": "exponential",
                "alternate_background_model": "linear",
                "rebin_factors": [1, 2],
            },
            "correction": {
                "name": "root_e2e_sf",
                "description": "root e2e",
                "flow": "clamp",
            },
        }
    )
    data = count_files(config, [str(data_path)], "data", base_dir=tmp_path)
    mc = count_files(config, [str(mc_path)], "mc", base_dir=tmp_path)
    result = fit_payload(merge([data, mc]))
    assert result["bins"][0]["valid"]
    output = tmp_path / "scale_factors.json.gz"
    write_payload(output, build_payload(result))
    value = correctionlib.CorrectionSet.from_file(str(output))["root_e2e_sf"].evaluate(
        "nominal", 1.0, 7.0
    )
    assert 0.85 < value < 1.05
