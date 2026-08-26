import pytest

from cms_tnp.config import Expression, validate_config
from cms_tnp.profiles import PROFILES, resolve_profile


def test_every_profile_resolves():
    for name in PROFILES:
        config = resolve_profile(
            {
                "schema_version": 1,
                "profile": name,
                "measurement": f"test_{name}",
                "year": "2025",
                "samples": {"data": ["data"], "mc": ["mc"]},
            }
        )
        validate_config(config)


def test_user_knobs_override_without_code_changes():
    config = resolve_profile(
        {
            "schema_version": 1,
            "profile": "photon_z",
            "measurement": "private_photon_id",
            "year": "2025",
            "id": {
                "fields": ["mvaID_WP90"],
                "denominator": "(pt > 25) & (abs(eta + deltaEtaSC) < 2.5)",
                "pass": "mvaID_WP90",
            },
            "pt_edges_gev": [25, 40, 100, 500],
            "samples": {"data": ["data"], "mc": ["mc"]},
        }
    )
    validate_config(config)
    assert config["probe"]["pass"] == "mvaID_WP90"
    assert config["axes"]["pt_edges_gev"] == [25, 40, 100, 500]
    assert config["samples"]["mc"] == ["mc"]


def test_expression_language_has_no_python_escape():
    assert Expression("(pt > 5) & (abs(eta) < 2.4)").names == {"pt", "eta"}
    with pytest.raises(ValueError):
        Expression("__import__('os').system('id')").evaluate({})
