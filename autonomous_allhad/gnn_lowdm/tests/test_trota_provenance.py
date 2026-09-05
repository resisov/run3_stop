from __future__ import annotations

import pytest

from autonomous_allhad.gnn_lowdm._implementation.region_io import (
    EXPECTED_TROTA_MODEL_SHA256,
    EXPECTED_TROTA_SCHEMAS,
    validate_trota_provenance,
)


def payload(year: int) -> dict:
    return {
        f"trota_topresolved_{year}": {
            "status": "complete",
            "schema_version": EXPECTED_TROTA_SCHEMAS[year],
            "marker": {
                "status": "complete",
                "application_year": year,
                "model_release_year": 2024,
                "model_sha256": EXPECTED_TROTA_MODEL_SHA256,
            },
        }
    }


@pytest.mark.parametrize("year", [2024, 2025])
def test_accepts_supported_year_specific_trota_provenance(year: int) -> None:
    result = validate_trota_provenance(payload(year))
    assert result["application_year"] == year
    assert result["schema_version"] == EXPECTED_TROTA_SCHEMAS[year]
    assert result["model_sha256"] == EXPECTED_TROTA_MODEL_SHA256


def test_rejects_wrong_model_hash() -> None:
    sidecar = payload(2025)
    sidecar["trota_topresolved_2025"]["marker"]["model_sha256"] = "bad"
    with pytest.raises(RuntimeError, match="invalid TROTA provenance"):
        validate_trota_provenance(sidecar)


def test_rejects_ambiguous_year_payloads() -> None:
    sidecar = {**payload(2024), **payload(2025)}
    with pytest.raises(RuntimeError, match="exactly one supported year"):
        validate_trota_provenance(sidecar)
