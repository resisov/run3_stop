import importlib.util
from pathlib import Path

import numpy as np


PLOTTER = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "plot_control_search_bins_style.py"
)
SPEC = importlib.util.spec_from_file_location("plot_control_search_bins_style", PLOTTER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def leaf(sumw, sumw2):
    return {"entries": [1.0, 1.0], "sumw": sumw, "sumw2": sumw2}


def test_combined_dycr_sums_nominal_statistics_and_correlated_shapes():
    electron = {
        "DYJets": {
            "nominal": leaf([10.0, 20.0], [4.0, 9.0]),
            "pileupUp": leaf([12.0, 21.0], [5.0, 10.0]),
        },
        "data_obs": {"nominal": leaf([13.0, 25.0], [13.0, 25.0])},
    }
    muon = {
        "DYJets": {"nominal": leaf([5.0, 7.0], [2.0, 3.0])},
        "data_obs": {"nominal": leaf([6.0, 8.0], [6.0, 8.0])},
    }

    combined = MODULE.combine_histogram_containers(electron, muon)

    assert combined["DYJets"]["nominal"]["sumw"] == [15.0, 27.0]
    assert combined["DYJets"]["nominal"]["sumw2"] == [6.0, 12.0]
    assert combined["data_obs"]["nominal"]["sumw"] == [19.0, 33.0]
    # The muon nominal is the muon contribution to the correlated pileup-up
    # total because that source is absent from the muon input.
    assert combined["DYJets"]["pileupUp"]["sumw"] == [17.0, 28.0]
    variance = MODULE.background_systematic_variance(combined, 2)
    np.testing.assert_allclose(variance, [4.0, 1.0])


def test_combined_dycr_rejects_incompatible_binning():
    electron = {"DYJets": {"nominal": leaf([1.0, 2.0], [1.0, 2.0])}}
    muon = {
        "DYJets": {
            "nominal": {
                "entries": [1.0, 1.0, 1.0],
                "sumw": [1.0, 2.0, 3.0],
                "sumw2": [1.0, 2.0, 3.0],
            }
        }
    }

    try:
        MODULE.combine_histogram_containers(electron, muon)
    except ValueError as error:
        assert "bin counts" in str(error)
    else:
        raise AssertionError("incompatible DYCR binning was accepted")
