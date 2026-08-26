import awkward as ak
import numpy as np

from cms_tnp.payload import build_payload, write_payload
from cms_tnp.weights import WeightSet, required_fields


def test_correctionlib_weight_variations(tmp_path):
    result = {
        "adoption_blockers": [],
        "probe_abseta_edges": [0.0, 100.0],
        "probe_pt_edges_gev": [0.0, 1.0],
        "correction": {"name": "pileup", "description": "pileup", "flow": "clamp"},
        "bins": [
            {
                "flat_index": 0,
                "valid": True,
                "scale_factor": 2.0,
                "scale_factor_uncertainty": 0.2,
            }
        ],
    }
    correction_path = tmp_path / "pileup.json.gz"
    write_payload(correction_path, build_payload(result))
    config = {
        "mc_nominal": "genWeight",
        "mc_variations": {},
        "corrections": [
            {
                "file": correction_path.name,
                "name": "pileup",
                "inputs": [
                    {"variation": True},
                    {"field": "Pileup_nTrueInt"},
                    {"value": 0.5},
                ],
                "nominal": "nominal",
                "variations": {"pileup_up": "up", "pileup_down": "down"},
            }
        ],
    }
    arrays = ak.Array(
        [
            {"genWeight": 1.0, "Pileup_nTrueInt": 10.0},
            {"genWeight": 2.0, "Pileup_nTrueInt": 20.0},
        ]
    )
    weights = WeightSet(config, tmp_path).evaluate(arrays)
    assert required_fields(config) == {"genWeight", "Pileup_nTrueInt"}
    assert np.allclose(weights["nominal"], [2.0, 4.0])
    assert np.allclose(weights["pileup_up"], [2.2, 4.4])
    assert np.allclose(weights["pileup_down"], [1.8, 3.6])
