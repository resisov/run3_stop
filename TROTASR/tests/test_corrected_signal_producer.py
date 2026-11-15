"""Corrected UUID signal receipts hydrate only their separate nominal output."""
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from TROTASR.utils import histogramming, io, lhe_weights, systematic_views, trota_inference
from TROTASR.workflows import merge_nominal, produce, signal_grid_inputs, validation_regions as vr


def test_nominal_merge_recovers_completed_output_without_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(merge_nominal, "ROOT", tmp_path)
    monkeypatch.setattr(merge_nominal, "internal_path", Path)
    monkeypatch.setattr(io, "internal_path", Path)
    task = dict(input="mc_1.root", output="hist_1.json.gz", year="2024", entries=3)
    source = dict(schema="trotasr_histograms_v1", status="complete", sr_data_blinded=True,
                  bins_merged=False, highdm_sr_source_axes="native", mixed_sf="not_provided",
                  category_order=[], contract=dict(input="/mirror/mc_1.root", year=2024,
                      modes=["highdm"], variations=["nominal"]),
                  audit=dict(events_read=3), input_metadata={}, histograms={})
    inventory, output = tmp_path / "tasks.json", tmp_path / "merged.json.gz"
    io.write_json(tmp_path / task["output"], source)
    io.write_json(inventory, dict(tasks=[task]))
    merge_nominal.merge(inventory, "2024", output)
    summary = tmp_path / "merged.summary.json"
    summary.unlink()
    original = output.read_bytes()
    merge_nominal.merge(inventory, "2024", output)
    assert output.read_bytes() == original
    assert io.read_json(summary)["coverage"]["expected_events"] == 3
    io.write_json(inventory, dict(tasks=[dict(task, input="mc_2.root")]))
    with pytest.raises(ValueError, match="different histogram inputs"):
        merge_nominal.merge(inventory, "2024", output)


def test_lhe_receipt_matches_events_and_members_instead_of_transfer_size(monkeypatch):
    monkeypatch.setattr(lhe_weights, "_payload_source_coverage", lambda payload: (3, [], set()))
    task = dict(input="inputs/2024/mc_1.root", year="2024", entries=3, fingerprint=dict(size=2))
    payload = dict(status="complete_member_sums", sr_data_blinded=True,
                   scope="source_lhe_member_sums_not_reduced", audit=dict(events_read=3),
                   contract=dict(input="/another/mirror/mc_1.root", year=2024,
                       fingerprint=dict(size=10),
                       lhe_weights=dict(source="original_nanoaod", signal_uncertainties=False)),
                   input_metadata={"1": dict(is_data=False, is_signal=False)}, lhe_histograms={})
    assert lhe_weights._check_task_payload(task, "2024", payload) == ({}, 3, [])
    with pytest.raises(ValueError, match="registered background input"):
        lhe_weights._check_task_payload(dict(task, entries=4), "2024", payload)


def test_replacement_worker_revalidates_root_when_recorded_size_differs(monkeypatch, tmp_path):
    import numpy as np
    import uproot

    monkeypatch.setattr(produce, "ROOT", tmp_path)
    monkeypatch.setattr(produce, "internal_path", Path)
    era = "2024"
    relative = Path("inputs/2024/signal_01234567-abcd-4321-0987-000000001122.root")
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    names = ("nTopMixed1pct", "mStop", "mLSP", "file_id", "entry", "gen_weight")
    with uproot.recreate(path) as root:
        events = root.mktree("Events", {name: np.int64 for name in names})
        events.extend({name: np.zeros(2, dtype=np.int64) for name in names})
        truth = root.mktree("TopWTruth", {"entry": np.int64})
        truth.extend({"entry": np.zeros(2, dtype=np.int64)})
        root.mktree("TROTA", {"entry": np.int64})
    runs = {"GenModel_T2tt_1200_500": -2.5}
    dataset = "/SMS-2Stop-T2tt_Par-mStop-1200/campaign-v2/NANOAODSIM"
    path.with_suffix(".json").write_text(json.dumps(dict(
        status="complete", analysis_era=era, input_nanoaod_version=15, events_read=5, events_written=2,
        datasets={"7": dict(dataset=dataset, is_data=False, is_signal=True, signal_sumw_by_genmodel=runs)})))
    task = dict(year=era, input=str(relative), metadata=str(relative.with_suffix(".json")),
                signal_population=produce.REPLACEMENT_SIGNAL_POPULATION,
                fingerprint={"size": path.stat().st_size - 1}, entries=2, source_events=5,
                dataset=dataset, signal_sumw_by_genmodel=runs)
    assert produce._replacement_task_source(task) == path
    with pytest.raises(ValueError, match="retained events or Runs"):
        produce._replacement_task_source(dict(task, entries=3))
    with pytest.raises(ValueError, match="retained events or Runs"):
        produce._replacement_task_source(dict(task, signal_sumw_by_genmodel={next(iter(runs)): 2.5}))


def test_collect_recovers_invalid_local_pair_before_retiring_eos(monkeypatch, tmp_path):
    from TROTASR.utils import nanoaod_production

    monkeypatch.setattr(produce, "ROOT", tmp_path)
    monkeypatch.setattr(produce, "PRODUCTION", tmp_path / "production")
    monkeypatch.setattr(io, "internal_path", Path)
    monkeypatch.setattr(produce.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30))
    task = dict(era="2024", name="signal_2024", source={"name": "UUID.root", "nevents": 2})
    target = produce._collected_target(task)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bad")
    target.with_suffix(".json").write_text("{}")
    remote = produce.EOS_ROOT + "/production/outputs/2024/signal_2024"
    ready = {remote + ".root", remote + ".json"}
    metadata = {"events_read": 2, "analysis_era": "2024", "status": "complete"}

    def validate(path, era, source):
        if path.read_bytes() != b"verified" or json.loads(path.with_suffix(".json").read_text()) != metadata:
            raise ValueError("Invalid pair")
        return metadata

    retired = []

    def transfer(command):
        if command[0] == "xrdcp" and isinstance(command[-1], Path):
            data = b"verified" if command[-1].suffix == ".root" else json.dumps(metadata).encode()
            command[-1].write_bytes(data)
        elif command[0] == "xrdfs":
            retired.append(command[-1])
            assert produce._collected_receipt(task) is not None

    monkeypatch.setattr(nanoaod_production, "validate_output", validate)
    monkeypatch.setattr(produce, "run", transfer)
    assert produce.collect_file(task, ready) is None
    assert target.read_bytes() == b"verified"
    assert retired == [remote + ".root", remote + ".json"]


def test_corrected_era_merge_preserves_published_input(monkeypatch, tmp_path):
    era = "2024"
    nominal = tmp_path / f"estimations/normalization_{era}.json.gz"
    nominal.parent.mkdir()
    normalized = b"accepted generated Runs table"
    nominal.write_bytes(normalized)
    normalization = dict(path=str(nominal.relative_to(tmp_path)))
    out = tmp_path / f"stats/run3/new_signal/signal_histograms/{era}"
    out.mkdir(parents=True)
    monkeypatch.setattr(produce, "ROOT", tmp_path)
    monkeypatch.setattr(produce, "internal_path", lambda path: path)
    monkeypatch.setattr(signal_grid_inputs, "ROOT", tmp_path)
    monkeypatch.setattr(signal_grid_inputs, "internal_path", Path)
    monkeypatch.setattr(io, "internal_path", lambda path: path)
    monkeypatch.setattr(produce, "REPLACEMENT_FILES_PER_ERA", 2)
    monkeypatch.setattr(vr, "_available_memory", lambda: 100 * 2**30)
    monkeypatch.setattr(produce.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30))
    sources = {}
    monkeypatch.setattr(lhe_weights, "lhe_input_population", lambda path, era: sources[(era, path)])
    monkeypatch.setattr(lhe_weights, "signal_lhe_provenance", lambda source, metadata, era: {
        "source": source, "file_id": metadata["file_id"], "era": era,
    })

    records = []
    for i in (1, 2):
        source = tmp_path / "inputs" / era / f"signal_deadbeef-abcd-0000-1234-{i:012d}.root"
        metadata = source.with_suffix(".json")
        metadata.parent.mkdir(parents=True, exist_ok=True)
        upstream = f"/store/mc/UUID-source-{i}.root"
        sources[(era, source)] = upstream
        io.write_json(metadata, {"file_id": i})
        fp = dict(size=10, mtime_ns=12)
        task = dict(input=str(source.relative_to(tmp_path)), metadata=str(metadata.relative_to(tmp_path)),
                    year=era, fingerprint=fp, entries=10)
        records.append(dict(task=task, normalization=normalization))
        leaf = dict(sumw=[float(i)], sumw2=[float(i*i)], entries=[1], edges=[0., 1.])
        contract = dict(input=str(source), year=2024, fingerprint=fp,
                        shape_mode="cms_trota_jme", nominal_calibrations_only=True,
                        signal_sr_only=True, regions=["SR"],
                        execution=dict(inference_support="all_retained_mc_rows",
                                       physics_definitions_changed=False),
                        registered_signal_source={"source": upstream, "file_id": i, "era": era},
                        variations=["nominal"])
        payload = dict(status="complete", sr_data_blinded=True,
                       highdm_sr_source_axes="exact_nb_exact_topology_native_recoil_v2",
                       contract=contract, audit=dict(events_read=10),
                       histograms={mode: {"nominal": {"SR": {"Nb1": {"mStop1200_mLSP500": {"recoil": leaf}}}}}
                                   for mode in ("lowdm", "highdm")}, physical_histograms={},
                       input_metadata={"7": dict(is_signal=True, is_data=False, events_written=10)})
        io.write_json(out / (source.stem + ".json.gz"), payload)

    first = produce._merge_corrected_signals(era, records)
    second = produce._merge_corrected_signals(era, records)
    assert first == second
    assert json.loads(json.dumps(first)) == first
    assert first["scope"] == "full_cms_trota_jme_production"
    assert first["coverage"]["expected_events"] == 20
    assert first["normalization"] == normalization
    assert "registered_signal_source" not in first["physics_contract"]
    assert "execution" not in first["physics_contract"]
    card_input = tmp_path / first["output"]
    assert signal_grid_inputs.replacement_source_binding(card_input, era, first)["path"] == first["output"]
    without_regions = dict(first, physics_contract={
        key: value for key, value in first["physics_contract"].items() if key != "regions"
    })
    assert signal_grid_inputs.replacement_source_binding(card_input, era, without_regions)["path"] == first["output"]
    with gzip.open(tmp_path / first["output"], "rt") as fh:
        result = json.load(fh)
    assert result["sr_data_blinded"] is True
    assert result["input_metadata"]["7"]["events_written"] == 20
    assert result["histograms"]["highdm"]["nominal"]["SR"]["Nb1"]["mStop1200_mLSP500"]["recoil"]["sumw"] == [3.]
    assert normalized == nominal.read_bytes()
    assert not (tmp_path / "results").exists()

    source = tmp_path / records[-1]["task"]["input"]
    path = out / (source.stem + ".json.gz")
    previous = io.read_json(path)
    merged_path = tmp_path / first["output"]
    published_result = merged_path.read_bytes()
    true_provenance = previous["contract"]["registered_signal_source"]
    previous["contract"]["registered_signal_source"] = dict(true_provenance, file_id=1)
    io.write_json(path, previous)
    produce._merge_corrected_signals(era, records)
    previous["contract"]["registered_signal_source"] = true_provenance
    previous["histograms"]["lowdm"]["nominal"]["DY2M"] = {}
    io.write_json(path, previous)
    with pytest.raises(ValueError, match="misbound"):
        produce._merge_corrected_signals(era, records)
    del previous["histograms"]["lowdm"]["nominal"]["DY2M"]
    io.write_json(path, previous)

    records[-1]["task"]["entries"] = 5
    with pytest.raises(ValueError, match="misbound"):
        produce._merge_corrected_signals(era, records)
    assert merged_path.read_bytes() == published_result


def test_uuid_sr_only_uses_current_normalization_without_legacy_asset_guard(monkeypatch, tmp_path):
    source, output = (tmp_path / name for name in ("signal.root", "output.json"))
    source.touch()
    era = "2024"
    norm = tmp_path / f"estimations/normalization_{era}.json.gz"
    registered = {"source": "/store/mc/registered-signal.root"}
    read_root = Mock(num_entries=0)
    read_root.iterate.return_value = iter(())
    provenance = Mock(return_value=registered)
    monkeypatch.setattr(histogramming, "ROOT", tmp_path)
    monkeypatch.setattr(histogramming, "internal_path", Path)
    monkeypatch.setattr(histogramming, "validate_root", lambda *a: {"datasets": {}})
    read_json = Mock(return_value={"xsec_pb": 1.})
    monkeypatch.setattr(histogramming, "read_json", read_json)
    monkeypatch.setattr(histogramming, "write_json", Mock())
    monkeypatch.setattr(histogramming, "topw_file_input_policy", Mock())
    monkeypatch.setattr(histogramming, "TopWEvents", Mock(return_value=read_root))
    monkeypatch.setattr(histogramming.uproot, "open", lambda *a, **kw: nullcontext({"Events": {}}))
    monkeypatch.setattr(lhe_weights, "lhe_input_population", Mock(return_value=registered))
    monkeypatch.setattr(lhe_weights, "signal_lhe_provenance", provenance)
    monkeypatch.setattr(systematic_views, "event_views", Mock(return_value=iter(())))
    monkeypatch.setattr(trota_inference, "load_models", Mock())

    options = dict(modes=("highdm",), shape_mode="cms_trota_jme", use_trota_friend=False,
                   nominal_calibrations_only=True, normalization_path=norm)
    contract = histogramming.build(source, era, output, signal_sr_only=True, **options)["contract"]
    provenance.assert_called_once()
    assert contract["registered_signal_source"] == registered
    assert any(call.args == (norm,) for call in read_json.call_args_list)
    assert all(Path(call.args[0]).name != "assets.json" for call in read_json.call_args_list)
