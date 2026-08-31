from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
WORKFLOW = PROJECT / "workflow"
sys.path.insert(0, str(WORKFLOW))
MODULE_PATH = WORKFLOW / "build_lowdm_only_canonical_inputs.py"
SPEC = importlib.util.spec_from_file_location("lowdm_only_canonical", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def channel(name: str, regime: str, region: str, parameter: str | None = None) -> dict:
    return {
        "name": name,
        "regime": regime,
        "region": region,
        "rate_params": {} if parameter is None else {"Top": parameter},
    }


def test_lowdm_channel_filter_and_rate_parameter_scope() -> None:
    channels = [
        channel("LLCR_highdm_Nb1_bin0", "highdm", "LLCR", "high"),
        channel("SR_highdm_bin0", "highdm", "SR", "high"),
        channel("LLCR_lowdm_bin0", "lowdm", "LLCR", "low"),
        channel("SR_lowdm_bin0", "lowdm", "SR", "low"),
    ]
    selected = MODULE.lowdm_channels(channels)
    assert [item["name"] for item in selected] == [
        "LLCR_lowdm_bin0",
        "SR_lowdm_bin0",
    ]
    parameters, scopes = MODULE.validate_rate_parameter_scopes(selected)
    assert parameters == ["low"]
    assert scopes == {"low": ["cr", "sr"]}


def test_unmatched_lowdm_rate_parameter_is_rejected() -> None:
    channels = [
        channel("LLCR_lowdm_bin0", "lowdm", "LLCR", "orphan"),
        channel("SR_lowdm_bin0", "lowdm", "SR"),
    ]
    with pytest.raises(ValueError, match="unmatched Low-dM rate parameters"):
        MODULE.validate_rate_parameter_scopes(channels)


def test_lowdm_mass_points_require_positive_lowdm_yield(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE.canonical,
        "mass_points",
        lambda *args, **kwargs: ["mStop700_mLSP500", "mStop900_mLSP700"],
    )

    def signal_leaf(hists, regime, mass_key, variation, topology):
        assert regime == "lowdm"
        assert variation == "nominal"
        return (
            np.asarray([1.0, 2.0])
            if mass_key == "mStop700_mLSP500"
            else np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
        )

    monkeypatch.setattr(MODULE.canonical, "signal_leaf", signal_leaf)
    selected = MODULE.lowdm_mass_points({}, "T2tt", None, 1800)
    assert selected == ["mStop700_mLSP500"]
