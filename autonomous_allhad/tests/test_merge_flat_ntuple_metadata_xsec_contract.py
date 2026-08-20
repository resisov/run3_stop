from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "merge_flat_ntuple_metadata.py"
)
SPEC = importlib.util.spec_from_file_location("merge_flat_ntuple_metadata", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def authoritative_payload() -> dict:
    return {
        "schema_version": MODULE.SIGNAL_XSEC_SCHEMA,
        "source_file": "signal_xsec.txt",
        "xsec_table_status": "parsed",
        "parsed": True,
        "records_parsed": 2,
        "records": [
            {
                "mStop": 100,
                "xsec_pb": 2046.0,
                "uncertainty_relative": 0.0461,
                "parsing_status": "parsed",
            },
            {
                "mStop": 105,
                "xsec_pb": 1666.0,
                "uncertainty_relative": 0.0457,
                "parsing_status": "parsed",
            },
        ],
    }


def test_accepts_only_authoritative_stop_pair_xsec_schema(tmp_path: Path) -> None:
    path = write_json(tmp_path / "stop_xsec.json", authoritative_payload())
    loaded = MODULE.load_signal_xsec(path)
    assert loaded == {
        "mStop100": {
            "xsec_pb": 2046.0,
            "xsec_uncertainty_relative": 0.0461,
        },
        "mStop105": {
            "xsec_pb": 1666.0,
            "xsec_uncertainty_relative": 0.0457,
        },
    }


def test_rejects_mass_point_yield_products(tmp_path: Path) -> None:
    payload = authoritative_payload()
    payload["mass_points"] = {
        "mStop100_mLSP1": {"xsec_pb": 2046.0, "normalized_weighted": 1.0}
    }
    path = write_json(tmp_path / "derived_yields.json", payload)
    with pytest.raises(RuntimeError, match="yield-shaped JSON"):
        MODULE.load_signal_xsec(path)


def test_rejects_unknown_schema_even_if_records_look_valid(tmp_path: Path) -> None:
    payload = authoritative_payload()
    payload["schema_version"] = "derived_analysis_product_v1"
    path = write_json(tmp_path / "wrong_schema.json", payload)
    with pytest.raises(RuntimeError, match="refusing non-authoritative"):
        MODULE.load_signal_xsec(path)


def test_rejects_missing_xsec_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        MODULE.load_signal_xsec(tmp_path / "missing.json")


def test_authoritative_stop_pair_xsec_covers_each_supported_topology() -> None:
    datasets = {
        "signal": {
            "dataset": "SMS-2Stop",
            "is_signal": True,
            "signal_sumw_by_genmodel": {
                "GenModel_T2tt_100_1": 10.0,
                "GenModel_T2tb_100_1": 20.0,
                "GenModel_T2bW_100_1": 40.0,
            },
            "signal_event_genweight_sum_by_genmodel": {},
        }
    }
    xsecs = {
        "mStop100": {
            "xsec_pb": 2.0,
            "xsec_uncertainty_relative": 0.1,
        }
    }
    points = MODULE.build_signal_mass_points(datasets, xsecs, lumi_pb=1000.0)
    assert set(points) == {
        "mStop100_mLSP1",
        "T2tb_mStop100_mLSP1",
        "T2bW_mStop100_mLSP1",
    }
    assert points["mStop100_mLSP1"]["normalization_factor"] == 200.0
    assert points["T2tb_mStop100_mLSP1"]["normalization_factor"] == 100.0
    assert points["T2bW_mStop100_mLSP1"]["normalization_factor"] == 50.0
    assert all(
        point["normalization_status"]
        == "normalized_with_signal_xsec_and_runs_mass_point_sumw"
        for point in points.values()
    )
