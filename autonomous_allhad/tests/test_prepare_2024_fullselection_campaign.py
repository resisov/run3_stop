from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_2024_fullselection_campaign.py"
SPEC = importlib.util.spec_from_file_location("prepare_2024_fullselection_campaign", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def record(path: str, *, role: str = "background") -> dict[str, object]:
    return {
        "sample_name": "Sample",
        "dataset": "Sample-RunIII2024",
        "process_group": "TT",
        "year": "2024",
        "file_path": path,
        "is_data": role == "data",
        "is_background": role == "background",
        "is_signal": role == "signal",
        "sumw_source": "Runs.genEventSumw",
    }


def test_frozen_inputs_are_normalized_audited_and_bad_files_removed(tmp_path: Path) -> None:
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({
        "schema_version": "test",
        "records": [record("/store/good.root"), record("/store/bad.root")],
    }))
    records, summary = MODULE.frozen_input_records(
        path,
        {"root://cms-xrd-global.cern.ch//store/bad.root"},
    )
    assert [item["file_path"] for item in records] == [
        "root://cms-xrd-global.cern.ch//store/good.root"
    ]
    assert summary["records"] == 1
    assert summary["background_records"] == 1
    assert summary["excluded_bad_files"][0]["file_path"].endswith("/store/bad.root")
    assert len(summary["sha256"]) == 64


def test_frozen_inputs_reject_duplicate_files(tmp_path: Path) -> None:
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"records": [record("/store/a.root"), record("/store/a.root")]}))
    with pytest.raises(RuntimeError, match="duplicate frozen input file"):
        MODULE.frozen_input_records(path, set())


def test_normalize_lfn_collapses_alternate_xrootd_endpoints() -> None:
    assert MODULE.normalize_lfn("root://xrootd-cms.infn.it//store/mc/a.root") == (
        "root://cms-xrd-global.cern.ch//store/mc/a.root"
    )


def test_main_wrapper_integrates_trota_before_stageout() -> None:
    wrapper = MODULE.wrapper_text(
        "x509up_u147757",
        "model_TopResolved_2024_TROTA2D_ptcut.h5",
    )
    inference = wrapper.index("autonomous_allhad.trota_resolved_2024_inplace")
    stageout = wrapper.index('"$XRDCOPY" -f')
    assert inference < stageout
    assert "LCG_104/x86_64-el9-gcc13-opt/setup.sh" in wrapper
    assert '--input "$WORKDIR/out.root"' in wrapper
    assert "--allow-hadd-repair" in wrapper
    assert 'd["root_trees"]=["Events","TROTA"]' in wrapper
    assert 't.get("status") == "complete"' in wrapper
    assert "trota.json" in wrapper


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
