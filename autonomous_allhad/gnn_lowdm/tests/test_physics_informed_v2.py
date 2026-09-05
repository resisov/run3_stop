from __future__ import annotations

import json

import awkward as ak
import numpy as np
import torch
import uproot

from autonomous_allhad.gnn_lowdm.data import (
    DIAGONAL_V3_GLOBAL_FEATURE_NAMES,
    ENGINEERED_EXPANDED_GLOBAL_FEATURE_NAMES,
    GraphEvents,
    canonical_config,
    lowdm30_category_ids,
    lowdm30_coordinates,
)
from autonomous_allhad.gnn_lowdm.model import (
    PhysicsInformedJetGraphClassifier,
    wrapped_edge_geometry,
)
from autonomous_allhad.gnn_lowdm._implementation.physics_informed_v2 import (
    CORE_GLOBAL_FEATURE_NAMES,
    EXTENDED_NO_RESOLVED_GLOBAL_FEATURE_NAMES,
    RESOLVED_RECONSTRUCTION_FEATURE_NAMES,
    physics_informed_loss_weights,
    rotate_graph_to_met_frame_inplace,
    select_global_features,
)
from autonomous_allhad.gnn_lowdm._implementation.significance import (
    diagonal_v3_category_ids,
    evaluate_binning,
    soft_s_over_sqrt_b_loss,
)
from autonomous_allhad.gnn_lowdm._implementation.build_expanded_feature_cache import (
    lowdm_cache_selection_mask,
    process_source_with_retries,
    worker,
)


def make_events(count: int) -> GraphEvents:
    return GraphEvents(
        node_features=np.zeros((count, 3, 6), dtype=np.float32),
        node_mask=np.ones((count, 3), dtype=bool),
        node_eta=np.zeros((count, 3), dtype=np.float32),
        node_phi=np.zeros((count, 3), dtype=np.float32),
        global_features=np.zeros(
            (count, len(ENGINEERED_EXPANDED_GLOBAL_FEATURE_NAMES)),
            dtype=np.float32,
        ),
        labels=np.zeros(count, dtype=np.float32),
        process_id=np.zeros(count, dtype=np.int64),
        signal_topology_id=np.zeros(count, dtype=np.int32),
        gen_weight=np.ones(count),
        sampling_weight=np.ones(count),
        fold=np.zeros(count, dtype=np.int64),
        physical_dataset_id=np.arange(count, dtype=np.int64),
        run=np.zeros(count, dtype=np.int64),
        luminosity_block=np.zeros(count, dtype=np.int64),
        event=np.arange(count, dtype=np.int64),
        mstop=np.zeros(count, dtype=np.int32),
        mlsp=np.zeros(count, dtype=np.int32),
        lowdm_search_bin=np.full(count, 8, dtype=np.int32),
    )


def test_canonical_sr_has_exactly_thirty_bins() -> None:
    definition = canonical_config()["sr_binning"]
    assert definition["total_bins"] == 30
    assert len(definition["category_labels"]) == 6
    assert all(
        len(definition["edges_by_category"][label]) == 6
        for label in definition["category_labels"]
    )
    nb = np.asarray([1, 1, 1, 2, 2, 3])
    nisr = np.asarray([0, 1, 2, 0, 1, 4])
    np.testing.assert_array_equal(lowdm30_category_ids(nb, nisr), np.arange(6))
    np.testing.assert_array_equal(
        lowdm30_coordinates(nb, nisr, np.ones(6)),
        np.asarray([4, 9, 14, 19, 24, 29]),
    )


def test_wrapped_edge_geometry_distinguishes_pi_from_zero() -> None:
    eta = torch.zeros((1, 2))
    phi = torch.tensor([[0.0, torch.pi]])
    geometry = wrapped_edge_geometry(eta, phi)
    torch.testing.assert_close(geometry[0, 0, 1, 3], torch.tensor(torch.pi))
    phi_same = torch.tensor([[0.0, 2.0 * torch.pi]])
    same_geometry = wrapped_edge_geometry(eta, phi_same)
    assert float(same_geometry[0, 0, 1, 3]) < 2.0e-4


def test_met_frame_rotation_removes_absolute_azimuth() -> None:
    events = make_events(1)
    names = list(ENGINEERED_EXPANDED_GLOBAL_FEATURE_NAMES)
    met_phi = np.float32(2.8)
    events.global_features[0, names.index("sin_met_phi")] = np.sin(met_phi)
    events.global_features[0, names.index("cos_met_phi")] = np.cos(met_phi)
    events.node_phi[0] = np.asarray([-2.9, 2.8, 0.1], dtype=np.float32)
    rotate_graph_to_met_frame_inplace(events)
    expected = np.arctan2(
        np.sin(np.asarray([-2.9, 2.8, 0.1]) - met_phi),
        np.cos(np.asarray([-2.9, 2.8, 0.1]) - met_phi),
    )
    np.testing.assert_allclose(events.node_phi[0], expected, atol=1.0e-6)
    np.testing.assert_allclose(
        events.node_features[0, :, 2], np.sin(expected), atol=2.0e-7
    )
    np.testing.assert_allclose(
        events.node_features[0, :, 3], np.cos(expected), atol=2.0e-7
    )
    selected = select_global_features(events, "core")
    assert selected.global_features.shape == (1, len(CORE_GLOBAL_FEATURE_NAMES))
    assert "sin_met_phi" not in CORE_GLOBAL_FEATURE_NAMES
    assert "cos_met_phi" not in CORE_GLOBAL_FEATURE_NAMES
    assert not set(RESOLVED_RECONSTRUCTION_FEATURE_NAMES) & set(
        EXTENDED_NO_RESOLVED_GLOBAL_FEATURE_NAMES
    )


def test_sparse_signal_points_receive_sqrt_neff_not_equal_budgets() -> None:
    events = make_events(9)
    events.labels[:] = np.asarray([0, 0, 1, 1, 1, 1, 1, 1, 1])
    events.signal_topology_id[:] = np.asarray([0, 0, 1, 1, 1, 1, 1, 2, 2])
    events.mstop[:] = np.asarray([0, 0, 800, 900, 900, 900, 900, 1000, 1000])
    events.mlsp[:] = np.asarray([0, 0, 650, 750, 750, 750, 750, 750, 750])
    physics = np.asarray([1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 6.0])
    weights = physics_informed_loss_weights(
        events, np.arange(len(events.labels)), physics, cap_quantile=1.0
    )
    np.testing.assert_allclose(weights[:2].sum(), 4.5)
    np.testing.assert_allclose(weights[5:].sum() + weights[2:5].sum(), 4.5)
    sparse = float(weights[2])
    dense = float(weights[3:7].sum())
    np.testing.assert_allclose(dense / sparse, 2.0, rtol=1.0e-6)
    np.testing.assert_allclose(weights[1] / weights[0], 3.0)


def test_physics_informed_model_is_jet_permutation_invariant() -> None:
    torch.manual_seed(17)
    model = PhysicsInformedJetGraphClassifier(
        global_features=7, hidden=16, message_layers=2, dropout=0.0
    ).eval()
    nodes = torch.randn(2, 5, 6)
    eta = torch.randn(2, 5)
    phi = torch.randn(2, 5)
    mask = torch.tensor([[1, 1, 1, 1, 0]] * 2, dtype=torch.bool)
    globals_ = torch.randn(2, 7)
    permutation = torch.tensor([2, 0, 3, 1, 4])
    with torch.no_grad():
        nominal = model(nodes, mask, eta, phi, globals_)
        permuted = model(
            nodes[:, permutation],
            mask[:, permutation],
            eta[:, permutation],
            phi[:, permutation],
            globals_,
        )
    torch.testing.assert_close(nominal, permuted, atol=1.0e-6, rtol=1.0e-6)


def test_diagonal_v3_selection_removes_only_met_sqrt_ht_requirement() -> None:
    arrays = {
        "is_background": np.asarray([1, 1, 1, 1], dtype=bool),
        "is_signal": np.zeros(4, dtype=bool),
        "feature_lowdm_preselection": np.asarray([1, 1, 1, 0], dtype=bool),
        "pass_lowdm_topology_veto": np.asarray([1, 1, 0, 1], dtype=bool),
        "pass_lowdm_met_sqrt_ht": np.asarray([1, 0, 1, 1], dtype=bool),
        "nb_medium_lowdm": np.asarray([1, 1, 1, 1], dtype=np.int16),
    }
    nres = np.asarray([0, 0, 0, 0], dtype=np.int16)
    legacy = lowdm_cache_selection_mask(
        arrays, nres, "mc", require_met_sqrt_ht=True
    )
    diagonal = lowdm_cache_selection_mask(
        arrays, nres, "mc", require_met_sqrt_ht=False
    )
    np.testing.assert_array_equal(legacy, [True, False, False, False])
    np.testing.assert_array_equal(diagonal, [True, True, False, False])


def test_cache_source_retry_recovers_through_eos_xrootd(monkeypatch) -> None:
    calls: list[str] = []

    def fake_process(record, kind, *, selection_mode, root_override=None):
        calls.append(str(root_override))
        if len(calls) < 3:
            raise OSError("transient EOS FUSE failure")
        return {"event": np.asarray([7])}, {"selected_events": 1}

    monkeypatch.setattr(
        "autonomous_allhad.gnn_lowdm._implementation.build_expanded_feature_cache.process_source",
        fake_process,
    )
    monkeypatch.setattr(
        "autonomous_allhad.gnn_lowdm._implementation.build_expanded_feature_cache.time.sleep",
        lambda _: None,
    )
    payload, stats, audit = process_source_with_retries(
        {"root": "/eos/user/t/taiwoo/source.root", "sidecar": "unused"},
        "mc",
        selection_mode="diagonal_v3",
    )
    assert payload["event"].tolist() == [7]
    assert stats["selected_events"] == 1
    assert audit["attempts"] == 3
    assert calls == [
        "/eos/user/t/taiwoo/source.root",
        "/eos/user/t/taiwoo/source.root",
        "root://eosuser.cern.ch//eos/user/t/taiwoo/source.root",
    ]


def test_cache_worker_streams_sources_into_one_tree(tmp_path, monkeypatch) -> None:
    def fake_process(record, kind, *, selection_mode):
        start = int(record["start"])
        count = int(record["count"])
        return (
            {
                "event": np.arange(start, start + count, dtype=np.int64),
                "jets": ak.Array([[float(value)] for value in range(count)]),
            },
            {"selected_events": count},
            {"attempts": 1, "access_endpoint": record["root"]},
        )

    monkeypatch.setattr(
        "autonomous_allhad.gnn_lowdm._implementation.build_expanded_feature_cache.process_source_with_retries",
        fake_process,
    )
    output = tmp_path / "streamed.root"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "selection_mode": "diagonal_v3",
                "kind": "mc",
                "batch": 0,
                "inputs": [
                    {"root": "first", "start": 1, "count": 2},
                    {"root": "second", "start": 3, "count": 3},
                ],
                "output": str(output),
            }
        )
    )
    assert worker(request) == 0
    with uproot.open(output) as root_file:
        assert root_file["Events"].num_entries == 5
        np.testing.assert_array_equal(
            root_file["Events"]["event"].array(library="np"), [1, 2, 3, 4, 5]
        )
    sidecar = json.loads(output.with_suffix(".json").read_text())
    assert sidecar["output_write_mode"] == "streaming_ttree_extend_per_source"


def test_soft_significance_loss_prefers_signal_above_background() -> None:
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    weights = torch.ones(4)
    good = soft_s_over_sqrt_b_loss(
        torch.tensor([3.0, 2.0, -2.0, -3.0]), labels, weights
    )
    bad = soft_s_over_sqrt_b_loss(
        torch.tensor([-2.0, -3.0, 3.0, 2.0]), labels, weights
    )
    assert good < bad


def test_soft_significance_loss_can_optimize_physics_categories() -> None:
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    categories = torch.tensor([0, 0, 1, 1])
    weights = torch.ones(4)
    good = soft_s_over_sqrt_b_loss(
        torch.tensor([2.0, -2.0, 3.0, -3.0]),
        labels,
        weights,
        category_ids=categories,
    )
    bad = soft_s_over_sqrt_b_loss(
        torch.tensor([-2.0, 2.0, -3.0, 3.0]),
        labels,
        weights,
        category_ids=categories,
    )
    assert torch.isfinite(good)
    assert good < bad


def test_diagonal_v3_category_ids_decode_nb_and_isr_masks() -> None:
    base = make_events(4)
    payload = {
        field: getattr(base, field) for field in GraphEvents.__dataclass_fields__
    }
    payload["global_features"] = np.zeros(
        (4, len(DIAGONAL_V3_GLOBAL_FEATURE_NAMES)), dtype=np.float32
    )
    events = GraphEvents(**payload)
    names = list(DIAGONAL_V3_GLOBAL_FEATURE_NAMES)
    events.global_features[:, names.index("nb_medium")] = (
        np.asarray([1, 1, 2, 3], dtype=np.float32) / 5.0
    )
    events.global_features[:, names.index("n_lowdm_isr")] = (
        np.asarray([0, 2, 0, 1], dtype=np.float32) / 4.0
    )
    np.testing.assert_array_equal(
        diagonal_v3_category_ids(events, np.arange(4)), [0, 1, 2, 3]
    )


def test_binned_significance_uses_all_four_physics_categories() -> None:
    count = 16
    base = make_events(count)
    payload = {
        field: getattr(base, field) for field in GraphEvents.__dataclass_fields__
    }
    payload["global_features"] = np.zeros(
        (count, len(DIAGONAL_V3_GLOBAL_FEATURE_NAMES)), dtype=np.float32
    )
    payload["labels"] = np.asarray([0] * 8 + [1] * 8, dtype=np.float32)
    payload["signal_topology_id"] = np.asarray([0] * 8 + [1] * 8)
    payload["mstop"] = np.asarray([0] * 8 + [600] * 8)
    payload["mlsp"] = np.asarray([0] * 8 + [400] * 8)
    events = GraphEvents(**payload)
    names = list(DIAGONAL_V3_GLOBAL_FEATURE_NAMES)
    categories = np.tile(np.arange(4), 4)
    events.global_features[:, names.index("nb_medium")] = np.where(
        categories >= 2, 2.0 / 5.0, 1.0 / 5.0
    )
    events.global_features[:, names.index("n_lowdm_isr")] = np.where(
        categories % 2, 1.0 / 4.0, 0.0
    )
    scores = np.tile(
        np.asarray([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]),
        2,
    )
    result = evaluate_binning(
        events,
        np.arange(count),
        scores,
        np.ones(count),
        (0.0, 0.5, 1.0),
        subset_scale=1.0,
        min_background_neff=1.0,
        max_relative_mc_stat=1.0,
    )
    assert result is not None
    assert result["total_bins"] == 8
    assert result["median_s_over_sqrt_b"] > 0.0
