from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_2025_data_objectcorr_campaign.py"
SPEC = importlib.util.spec_from_file_location(
    "prepare_2025_data_objectcorr_campaign", SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_main_wrapper_integrates_2025_trota_before_stageout() -> None:
    wrapper = MODULE.wrapper_text(
        "x509up_u147757",
        "model_TopResolved_2024_TROTA2D_ptcut.h5",
    )
    inference = wrapper.index("autonomous_allhad.trota_resolved_2024_inplace")
    stageout = wrapper.index('"$XRDCOPY" -f')
    assert inference < stageout
    assert "LCG_104/x86_64-el9-gcc13-opt/setup.sh" in wrapper
    assert '--target-year 2025' in wrapper
    assert '--input "$WORKDIR/out.root"' in wrapper
    assert "--allow-hadd-repair" in wrapper
    assert 'd["root_trees"]=["Events","TROTA"]' in wrapper
    assert 'd["trota_topresolved_2025"]=t' in wrapper
    assert 't.get("marker", {}).get("application_year") == 2025' in wrapper


def test_submit_transfers_model_and_requires_almalinux9(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "wrapper", "arguments", "logs", "py38", "worker", "payload",
            "shards", "proxy", "model",
        )
    }
    submit = MODULE.submit_text(
        paths["wrapper"],
        paths["arguments"],
        paths["logs"],
        paths["py38"],
        paths["worker"],
        paths["payload"],
        paths["shards"],
        paths["proxy"],
        paths["model"],
    )
    assert str(paths["model"]) in submit
    assert 'requirements = (OpSysAndVer =?= "AlmaLinux9")' in submit
    assert "request_memory = $(memory_mb)MB" in submit
    assert "queue name,shift,shard_name,root_out,memory_mb" in submit


def test_output_schema_is_the_integrated_2025_schema() -> None:
    assert MODULE.OUTPUT_SCHEMA == (
        "flat_ntuple_shard_v8_float32_fullselection_2025_trota"
    )
    assert MODULE.DEFAULT_CAMPAIGN.name == "flat2025_v8"
    assert MODULE.EOS_SCHEDD == "bigbird24.cern.ch"
