"""Corrected UUID signal receipts hydrate only their separate nominal output."""
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from TROTASR.utils import histogramming, io, lhe_weights, systematic_views, trota_inference
from TROTASR.workflows import produce, signal_grid_inputs, validation_regions as vr


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
                       histograms={mode: {"nominal": {"SR": {"signal": leaf}}}
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
    assert result["histograms"]["highdm"]["nominal"]["SR"]["signal"]["sumw"] == [3.]
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
