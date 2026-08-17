from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from autonomous_allhad import photon_fake_2024_worker as worker


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflow"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

import measure_photon_fakes_2024_v2 as measurement  # noqa: E402
import evaluate_photon_fake_datamc_2024_v2 as evaluation  # noqa: E402
import inject_photon_fakes_2024_v2 as injection  # noqa: E402


class _Builder:
    RECOIL_PT_BINS = (250.0, 400.0, 600.0)
    HIGHDM_DISTRIBUTION_VARIABLE_SPECS = {}


def _fill(
    channels: dict,
    probe: str,
    region: str,
    stratum: int,
    yield_value: float,
    recoil: float,
) -> None:
    dataset = {"channels": channels}
    record = worker._channel_origin_record(
        dataset,
        probe,
        "all",
        _Builder,
    )
    transfer = record["region_transfers"][region]["strata"][stratum]
    transfer["sumw"][0] += yield_value
    transfer["sumw2"][0] += yield_value
    transfer["entries"][0] += int(yield_value)
    distribution = record["distributions"][region]["recoil"]["strata"][stratum]
    index = int(
        np.searchsorted(
            np.asarray(distribution["bin_edges"], dtype=float),
            recoil,
            side="right",
        )
        - 1
    )
    distribution["sumw"][index] += yield_value
    distribution["sumw2"][index] += yield_value
    distribution["entries"][index] += int(yield_value)


def test_independent_kappa_closure_and_shared_covariance() -> None:
    data_channels: dict = {}
    yields = {
        "target": 50.0,
        "application": 25.0,
        "measurement_pass": 20.0,
        "measurement_fail": 10.0,
        "plj_other": 5.0,
    }
    for region in (
        measurement.KAPPA_SOURCE_REGION,
        measurement.KAPPA_VALIDATION_REGION,
    ):
        for stratum, recoil in ((0, 300.0), (1, 500.0)):
            for probe, yield_value in yields.items():
                _fill(
                    data_channels,
                    probe,
                    region,
                    stratum,
                    yield_value,
                    recoil,
                )

    kappa = measurement.fit_kappa(
        data_channels,
        {},
        {},
        measurement.KAPPA_SOURCE_REGION,
    )
    assert np.allclose(kappa["factors"][:2], [1.0, 1.0])
    assert kappa["covariance"][0, 1] > 0.0

    closure = measurement.closure_summary(
        data_channels,
        {},
        {},
        kappa,
        _Builder,
    )
    assert np.isclose(closure["target_fake_residual"], 100.0)
    assert np.isclose(closure["prediction"], 100.0)
    assert np.isclose(closure["target_over_prediction"], 1.0)
    assert np.allclose(
        closure["target_fake_residual_histogram"]["sumw"],
        [50.0, 50.0],
    )
    prediction_covariance = np.asarray(
        closure["prediction_audit"]["covariance"],
        dtype=float,
    )
    assert prediction_covariance[0, 1] > 0.0


def test_origin_fraction_replaces_truth_fake_for_every_process() -> None:
    def leaf(values: list[float]) -> dict:
        return {
            "bin_edges": [0.0, 1.0, 2.0],
            "sumw": values,
            "sumw2": values,
            "entries": [int(value) for value in values],
        }

    fraction, variance, audit = evaluation.origin_fraction(
        {
            "all": leaf([100.0, 0.0]),
            "prompt": leaf([80.0, 0.0]),
            "electron": leaf([0.0, 0.0]),
            "fake": leaf([20.0, 0.0]),
        }
    )
    assert np.allclose(fraction, [0.8, 0.8])
    assert np.allclose(variance, [0.0016, 0.0016])
    assert audit["fallback_bins"] == [1]
    assert np.allclose(audit["partition_difference"], [0.0, 0.0])


def test_v2_injection_removes_all_mc_truth_fake_and_adds_fake_once() -> None:
    def leaf(values: list[float]) -> dict:
        return {
            "bin_edges": [0.0, 1.0, 2.0],
            "sumw": values,
            "sumw2": values,
            "entries": [10, 20],
        }

    samples = {
        process: {
            "nominal": leaf([10.0, 20.0]),
            "jesUp": leaf([12.0, 24.0]),
        }
        for process in injection.BACKGROUND_PROCESSES
    }
    origin = {
        "all": leaf([100.0, 100.0]),
        "prompt": leaf([80.0, 80.0]),
        "electron": leaf([0.0, 0.0]),
        "fake": leaf([20.0, 20.0]),
    }
    fake_variations = {
        variation: leaf([5.0, 5.0])
        for variation in injection.REQUIRED_FAKE_VARIATIONS
    }
    fake_variations["photonFakeClosureUp"] = leaf([6.0, 6.0])
    measurement_payload = {
        "fake_prediction": {
            "histograms": {"GCR": fake_variations},
            "highdm_variable_histograms": {},
        },
        "mc_target_origin_histograms": {
            process: {"GCR": {"recoil": origin}}
            for process in injection.BACKGROUND_PROCESSES
        },
    }
    injection.inject_variable(
        samples,
        measurement_payload,
        "GCR",
        "recoil",
    )
    assert np.allclose(samples["GJ"]["nominal"]["sumw"], [8.0, 16.0])
    assert np.allclose(samples["QCD"]["nominal"]["sumw"], [13.0, 21.0])
    assert np.allclose(samples["QCD"]["jesUp"]["sumw"], [14.6, 24.2])
    assert np.allclose(
        samples["QCD"]["photonFakeClosureUp"]["sumw"],
        [14.0, 22.0],
    )
