import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from autonomous_allhad.intermediate_production import _check_events, _trota_command, _work_path


def _metadata():
    return {"status": "complete", "bad_files": [], "files_processed": 1,
            "record_digest": "abc", "files": [{"file_path": "source"}],
            "events_read": 10, "events_written": 4, "skim_flag": "feature_flat_preselection",
            "shape_shift": "nominal", "physical_datasets": {
                "sample": {"sumw": 12.5, "sumw2": 0, "xsec_pb": 0.7, "files_processed": 1}}}


def test_complete_inputs_and_normalization_are_required():
    shard = {"record_digest": "abc", "records": [{"file_path": "source"}]}
    original = _metadata()
    _check_events(copy.deepcopy(original), shard, original)
    for key, value in (("status", "failed"), ("events_written", 3),
                       ("events_read", 9), ("files_processed", 0),
                       ("files", [{"file_path": "different"}])):
        changed = copy.deepcopy(original)
        changed[key] = value
        with unittest.TestCase().assertRaises(RuntimeError):
            _check_events(changed, shard, original)
    changed = copy.deepcopy(original)
    changed["physical_datasets"]["sample"]["sumw"] = 13
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "normalization changed"):
        _check_events(changed, shard, original)


def test_trota_setup_is_limited_to_child_shell():
    args = [Path("/cvmfs/setup.sh"), Path("/eos/repo"), Path("/eos/event.root"),
            Path("/eos/model.h5"), Path("/eos/trota.json"), 2025]
    command = _trota_command(*args)
    assert command[:2] == ["/bin/bash", "-c"]
    assert 'source "$1"' in command[2]
    assert "autonomous_allhad.trota_resolved_2024_inplace" in command[2]
    assert command[command.index("--target-year") + 1] == "2025"
    assert command[command.index("--model") + 1] == "/eos/model.h5"


def test_laptop_tmp_and_afs_outputs_are_rejected():
    with patch.dict(os.environ, {"_CONDOR_SCRATCH_DIR": ""}):
        for path in ("/tmp/job", "/afs/job", "/Users/example/job", "/eos/tmp/job"):
            with unittest.TestCase().assertRaises(ValueError):
                _work_path(Path(path))
        assert _work_path(Path("/eos/user/example/job")) == Path("/eos/user/example/job")


if __name__ == "__main__":
    test_complete_inputs_and_normalization_are_required()
    test_trota_setup_is_limited_to_child_shell()
    test_laptop_tmp_and_afs_outputs_are_rejected()
    print("3 checks passed")
