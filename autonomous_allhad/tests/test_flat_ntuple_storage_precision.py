from __future__ import annotations

import awkward as ak
import numpy as np
import json
import pytest
import uproot

from autonomous_allhad import flat_ntuple_worker as flat


def test_compact_2024_storage_keeps_only_gen_weight_as_float64() -> None:
    row = {
        **{name: 1 for name in flat.INT64_FIELDS},
        **{name: 1 for name in flat.INT32_FIELDS},
        **{name: 1.25 for name in flat.FLOAT_FIELDS},
        **{name: True for name in flat.BOOL_FIELDS},
        **{name: [1.25, 2.5] for name in flat.VECTOR_FLOAT_FIELDS},
        **{name: [1, 2] for name in flat.VECTOR_INT_FIELDS},
    }
    try:
        flat.configure_float_storage(float32=True, keep_float64={"gen_weight"})
        types = flat.branch_types()
        arrays = flat.rows_to_arrays([row])

        assert types["gen_weight"] is np.float64
        assert types["met"] is np.float32
        assert types["good_jet_pt"] == "var * float32"
        assert arrays["gen_weight"].dtype == np.dtype("float64")
        assert arrays["met"].dtype == np.dtype("float32")
        assert ak.to_numpy(ak.flatten(arrays["good_jet_pt"])).dtype == np.dtype(
            "float32"
        )
    finally:
        flat.configure_float_storage(float32=False)


def test_float_storage_policy_rejects_unknown_branches() -> None:
    try:
        with np.testing.assert_raises_regex(RuntimeError, "unknown branches"):
            flat.configure_float_storage(
                float32=True,
                keep_float64={"not_a_branch"},
            )
    finally:
        flat.configure_float_storage(float32=False)


def _truth_fixture():
    stored = ak.Array({
        "run": [1, 1], "luminosityBlock": [4, 4], "event": [10, 11],
        "entry": [0, 1], "file_id": [5, 5], "year": [2024, 2024], "is_data": [False, False],
        "fatjet_source_index_all": [[0], []], "fatjet_eta_all": [[0.0], []],
        "fatjet_phi_all": [[0.0], []],
    })
    nano = ak.Array({
        "run": [1, 1], "luminosityBlock": [4, 4], "event": [10, 11],
        "FatJet_eta": [[0.0], []], "FatJet_phi": [[0.0], []],
        "GenPart_pdgId": [[6, 24, 5, 2, -1], []],
        "GenPart_genPartIdxMother": [[-1, 0, 0, 1, 1], []],
        "GenPart_eta": [[0.0, 0.0, 0.1, -0.1, 0.1], []],
        "GenPart_phi": [[0.0, 0.0, 0.1, -0.1, -0.1], []],
    })
    return stored, nano


def test_topw_truth_reuses_efficiency_classification():
    from analysis.processors.btageff import decay_flavor
    stored, nano = _truth_fixture()
    result = flat.topw_truth_payload(stored, nano, decay_flavor.py_func)
    assert ak.to_list(result["fatjet_decay_flavor_all"]) == [[4], []]
    assert ak.to_list(result["GenPart_genPartIdxMother"]) == [[-1, 0, 0, 1, 1], []]


@pytest.mark.parametrize("field,values,error", [
    ("event", [99, 11], "event identity"),
    ("FatJet_eta", [[0.1], []], "source identity"),
    ("GenPart_genPartIdxMother", [[99, 0, 0, 1, 1], []], "mother index"),
    ("GenPart_phi", [[0, 0, float('nan'), 0, 0], []], "non-finite"),
])
def test_topw_truth_rejects_mismatched_or_invalid_inputs(field, values, error):
    stored, nano = _truth_fixture()
    nano = ak.with_field(nano, ak.Array(values), field)
    with pytest.raises(ValueError, match=error):
        flat.topw_truth_payload(stored, nano, lambda *args: np.array([4]))


def test_topw_truth_rejects_invalid_jet_index():
    stored, nano = _truth_fixture()
    stored = ak.with_field(stored, ak.Array([[1], []]), "fatjet_source_index_all")
    with pytest.raises(ValueError, match="source index"):
        flat.topw_truth_payload(stored, nano, lambda *args: np.array([4]))


@pytest.mark.parametrize("year", [2024, 2025])
def test_topw_append_preserves_events_trota_and_is_idempotent(tmp_path, monkeypatch, year):
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    stored, nano = _truth_fixture()
    source = str(tmp_path / "mapped_source.root")
    file_id = flat.stable_id(source)
    stored = ak.with_field(stored, ak.Array([file_id, file_id]), "file_id")
    stored = ak.with_field(stored, ak.Array([year, year]), "year")
    original, output = tmp_path / "original.root", tmp_path / "augmented.root"
    types = {name: np.int64 for name in flat.TOPW_ID_FIELDS + ("year",)}
    types.update({"is_data": np.bool_, "fatjet_source_index_all": "var * int32",
                  "fatjet_eta_all": "var * float32", "fatjet_phi_all": "var * float32"})
    with uproot.recreate(original) as root:
        root.mktree("Events", types)
        root["Events"].extend({name: stored[name] for name in types})
        root.mktree("TROTA", {"event": np.int64, "score": np.float32})
        root["TROTA"].extend({"event": np.array([10]), "score": np.array([0.99], dtype=np.float32)})
        root["TROTA_metadata"] = json.dumps({"status": "complete", "application_year": year,
                                             "events_entries": 2, "model_sha256": "test_model"})
    original.with_suffix(".json").write_text(json.dumps({
        "status": "complete", "files": [{"file_path": source, "file_id": file_id,
        "events_written": 2, "read_status": "success", "processed_entry_ranges": [[0, 2]],
        "number_of_entries": 2}],
    }))
    monkeypatch.setattr(flat, "_read_topw_nano_rows", lambda source, entries, expected: nano[entries])
    original_hash = flat._sha256(original)
    result = flat.append_topw_truth(original, output, repo, tmp_path / "scratch", year)
    assert result["status"] == "complete"
    assert result["marker"]["flavor_counts"] == [0, 0, 0, 0, 1]
    assert flat._sha256(original) == original_hash
    assert flat._root_content_digests(original) == flat._root_content_digests(output, exclude_truth=True)
    def forbidden(*args):
        raise AssertionError("already-complete truth must not reread NanoAOD")
    monkeypatch.setattr(flat, "_read_topw_nano_rows", forbidden)
    assert flat.append_topw_truth(original, output, repo, tmp_path / "scratch", year)["status"] == "already_complete"


def test_topw_read_reorders_selected_rows(tmp_path):
    _, nano = _truth_fixture()
    source = tmp_path / "selected_rows.root"
    with uproot.recreate(source) as root:
        root["Events"] = {name: nano[name] for name in flat.TOPW_NANO_FIELDS}
    selected = flat._read_topw_nano_rows(str(source), np.array([1, 0, 1]), 2)
    assert ak.to_list(selected["event"]) == [11, 10, 11]
    with pytest.raises(ValueError, match="entry range"):
        flat._read_topw_nano_rows(str(source), np.array([2]), 2)
