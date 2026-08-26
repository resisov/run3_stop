import correctionlib
import numpy as np
import pytest
from scipy.special import voigt_profile

from cms_tnp.fit import fit_payload
from cms_tnp.payload import build_payload, write_payload
from cms_tnp.plot import plot_result


def _spectrum(edges, mean, width, efficiency, signal_yield, background):
    centers = 0.5 * (edges[:-1] + edges[1:])
    signal = np.exp(-0.5 * ((centers - mean) / width) ** 2)
    signal /= np.sum(signal)
    passed = efficiency * signal_yield * signal + background
    failed = (1.0 - efficiency) * signal_yield * signal + 1.2 * background
    return {
        "pass_sumw": [passed.tolist()],
        "pass_sumw2": [np.maximum(passed, 1.0).tolist()],
        "fail_sumw": [failed.tolist()],
        "fail_sumw2": [np.maximum(failed, 1.0).tolist()],
    }


def _payload():
    edges = np.linspace(2.6, 3.6, 51)
    fit = {
        "peak_bounds_gev": [2.85, 3.30],
        "signal_model": "gaussian",
        "background_model": "exponential",
        "alternate_signal_model": "double_gaussian",
        "alternate_background_model": "linear",
        "rebin_factors": [1, 2],
        "window_shrink_fraction": 0.05,
        "min_fail_significance": 0.0,
    }
    return {
        "schema_version": 1,
        "measurement": "test_sf",
        "year": "2025",
        "probe_collection": "Electron",
        "probe_selection": "pt > 5",
        "pass_selection": "privateId",
        "probe_abseta_edges": [0.0, 2.5],
        "probe_pt_edges_gev": [5.0, 10.0],
        "mass_edges_gev": edges.tolist(),
        "fit": fit,
        "correction": {"name": "test_sf", "description": "test", "flow": "clamp"},
        "samples": {
            "data": _spectrum(edges, 3.096, 0.04, 0.80, 50_000, 15),
            "mc": _spectrum(edges, 3.096, 0.035, 0.85, 60_000, 3),
        },
        "adoption_blockers": [],
    }


def test_fit_export_and_plot(tmp_path):
    result = fit_payload(_payload())
    assert result["bins"][0]["valid"]
    assert result["bins"][0]["scale_factor"] == pytest.approx(0.80 / 0.85, rel=0.03)
    output = tmp_path / "sf.json.gz"
    write_payload(output, build_payload(result))
    correction = correctionlib.CorrectionSet.from_file(str(output))["test_sf"]
    assert correction.evaluate("nominal", 1.0, 7.0) == pytest.approx(
        result["bins"][0]["scale_factor"]
    )
    manifest = plot_result(result, tmp_path / "plots")
    assert len(manifest["outputs"]) == 6


def test_highpt_z_fit():
    edges = np.linspace(60.0, 120.0, 61)
    centers = 0.5 * (edges[:-1] + edges[1:])
    signal = voigt_profile(centers - 91.1876, 1.5, 1.2476)
    signal /= np.sum(signal)

    def sample(efficiency):
        passed = 100_000 * efficiency * signal + 25
        failed = 100_000 * (1.0 - efficiency) * signal + 30
        return {
            "pass_sumw": [passed.tolist()],
            "pass_sumw2": [passed.tolist()],
            "fail_sumw": [failed.tolist()],
            "fail_sumw2": [failed.tolist()],
        }

    payload = {
        "schema_version": 1,
        "measurement": "z_sf",
        "year": "2025",
        "probe_collection": "Photon",
        "probe_selection": "pt > 20",
        "pass_selection": "privateId",
        "probe_abseta_edges": [0.0, 2.5],
        "probe_pt_edges_gev": [20.0, 100.0],
        "mass_edges_gev": edges.tolist(),
        "fit": {
            "peak_bounds_gev": [86.0, 96.0],
            "signal_model": "voigt",
            "natural_width_gev": 1.2476,
            "background_model": "exponential",
            "alternate_signal_model": "double_gaussian",
            "alternate_background_model": "linear",
            "rebin_factors": [1, 2],
            "window_shrink_fraction": 0.05,
        },
        "correction": {"name": "z_sf", "description": "z", "flow": "clamp"},
        "samples": {"data": sample(0.90), "mc": sample(0.92)},
        "adoption_blockers": [],
    }
    result = fit_payload(payload)
    assert result["bins"][0]["valid"]
    assert result["bins"][0]["scale_factor"] == pytest.approx(0.90 / 0.92, rel=0.03)
