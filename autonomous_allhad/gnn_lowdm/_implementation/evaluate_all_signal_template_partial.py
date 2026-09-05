#!/usr/bin/env python3
"""Evaluate one diagonal-v3 signal-cache shard with the frozen GNN.

Only the deterministic 70% test partition is used.  The output is a compact
per-mass-point SR histogram partial for the validation-frozen
Nb=(1,>=2) x NISR=(0,1,>=2) categorization.  No signal-mass or delta-mass
filter is applied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ..data import (
        DIAGONAL_V3_GLOBAL_FEATURE_NAMES,
        _read_one,
        split_buckets_2_1_7,
    )
    from .rank005_numpy import Rank005Numpy
except ImportError:
    from data import (  # type: ignore[no-redef]
        DIAGONAL_V3_GLOBAL_FEATURE_NAMES,
        _read_one,
        split_buckets_2_1_7,
    )
    from rank005_numpy import Rank005Numpy  # type: ignore[no-redef]


# Canonical event-level contract from autonomous_allhad.signal_models.
TOPOLOGY = {1: "T2tt", 2: "T2tb", 3: "T2bW"}
CATEGORY_LABELS = (
    "Nb1_NISR0",
    "Nb1_NISR1",
    "Nb1_NISR2plus",
    "Nb2plus_NISR0",
    "Nb2plus_NISR1",
    "Nb2plus_NISR2plus",
)


def require_path(path: str, label: str, *, allow_local: bool) -> Path:
    if allow_local:
        return Path(path)
    if not path.startswith("/eos/") or "/afs/" in path or "/tmp/" in path:
        raise ValueError(f"{label} is not an EOS-only path: {path}")
    return Path(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def category_ids(global_features: np.ndarray) -> np.ndarray:
    names = list(DIAGONAL_V3_GLOBAL_FEATURE_NAMES)
    nb = np.rint(global_features[:, names.index("nb_medium")] * 5.0).astype(np.int16)
    nisr = np.rint(global_features[:, names.index("n_lowdm_isr")] * 4.0).astype(np.int16)
    if np.any(nb < 1) or np.any(nisr < 0):
        raise RuntimeError("selected signal contains invalid Nb or NISR")
    nb_group = (nb >= 2).astype(np.int8)
    nisr_group = np.where(nisr == 0, 0, np.where(nisr == 1, 1, 2)).astype(np.int8)
    return (3 * nb_group + nisr_group).astype(np.int8)


def normalization_maps(manifest: dict[str, Any], xsec: dict[str, Any]):
    signal_sumw = {
        (str(row["topology"]), int(row["mStop"]), int(row["mLSP"])): float(row["sumw"])
        for row in manifest["normalization"]["signal_mass_points"]
        if float(row.get("sumw", 0.0)) != 0.0
    }
    stop_xsec = {
        int(row["mStop"]): float(row["xsec_pb"])
        for row in xsec["records"]
        if row.get("parsing_status") == "parsed"
    }
    return signal_sumw, stop_xsec


def counter(scores: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(scores, edges[0] + 1.0e-8, edges[-1] - 1.0e-8)
    return {
        "sumw": np.histogram(clipped, edges, weights=weights)[0].astype(float).tolist(),
        "sumw2": np.histogram(clipped, edges, weights=np.square(weights))[0].astype(float).tolist(),
        "entries": np.histogram(clipped, edges)[0].astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--local-cache", type=Path)
    parser.add_argument("--local-output", type=Path)
    parser.add_argument("--local-manifest", type=Path)
    parser.add_argument("--local-xsec", type=Path)
    parser.add_argument("--local-model", type=Path)
    parser.add_argument("--local-configuration", type=Path)
    opts = parser.parse_args()
    request = json.loads(opts.request.read_text())
    local_overrides = (
        opts.local_cache,
        opts.local_output,
        opts.local_manifest,
        opts.local_xsec,
        opts.local_model,
        opts.local_configuration,
    )
    if any(value is not None for value in local_overrides):
        if not opts.allow_local or not all(value is not None for value in local_overrides):
            raise ValueError("all local path overrides require --allow-local and must be set together")
        basename = Path(str(request["input_root"])).name
        request.update({
            "input_root": str(opts.local_cache / basename),
            "output": str(opts.local_output / (Path(basename).stem + ".json")),
            "campaign_manifest": str(opts.local_manifest),
            "xsec": str(opts.local_xsec),
            "model": str(opts.local_model),
            "configuration": str(opts.local_configuration),
        })
    source = require_path(str(request["input_root"]), "input ROOT", allow_local=opts.allow_local)
    output = require_path(str(request["output"]), "output", allow_local=opts.allow_local)
    manifest_path = require_path(
        str(request["campaign_manifest"]), "campaign manifest", allow_local=opts.allow_local
    )
    xsec_path = require_path(str(request["xsec"]), "cross section", allow_local=opts.allow_local)
    model_path = require_path(str(request["model"]), "model", allow_local=opts.allow_local)
    configuration_path = require_path(
        str(request["configuration"]), "configuration", allow_local=opts.allow_local
    )

    configuration_payload = json.loads(configuration_path.read_text())
    configuration = configuration_payload["sr_binning"]
    if int(configuration.get("total_bins", -1)) != 30:
        raise RuntimeError("canonical Low-dM SR configuration is not 30-bin")
    if tuple(configuration["category_labels"]) != CATEGORY_LABELS:
        raise RuntimeError("validation-frozen category labels changed")
    edges = {
        label: np.asarray(configuration["edges_by_category"][label], dtype=float)
        for label in CATEGORY_LABELS
    }

    events = _read_one(
        source,
        target_mstop=None,
        target_mlsp=None,
        max_jets=10,
        folds=5,
        require_highdm_exclusive=False,
        selection_branch="feature_lowdm_diagonal_v3_SR",
        include_mass_features=False,
        top_targeted_features=False,
        engineered_features_v2=False,
        engineered_features_expanded=False,
        engineered_features_diagonal_v3=True,
        allow_empty=False,
    )
    if events is None or not np.all(events.labels == 1):
        raise RuntimeError("signal cache produced an empty or non-signal event table")
    split = split_buckets_2_1_7(
        events.physical_dataset_id, events.run, events.luminosity_block, events.event
    )
    test_indices = np.flatnonzero(split == 2)
    test = events.take(test_indices)
    if opts.checkpoint is None:
        model = Rank005Numpy(model_path)
        scores = model.predict(
            test.node_features,
            test.node_mask,
            test.node_eta,
            test.node_phi,
            test.global_features,
            batch_size=int(request.get("batch_size", 256)),
        )
    else:
        import torch
        try:
            from ..model import PhysicsInformedJetGraphClassifier
            from .train_oof import predict, tensors
        except ImportError:  # Standalone EOS worker payload.
            from model import PhysicsInformedJetGraphClassifier
            from train_oof import predict, tensors

        device = torch.device(opts.device)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        checkpoint = torch.load(opts.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = PhysicsInformedJetGraphClassifier(
            node_features=test.node_features.shape[2],
            global_features=test.global_features.shape[1],
            hidden=int(config["hidden"]),
            message_layers=int(config["message_layers"]),
            dropout=float(config["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        scores = predict(
            model,
            tensors(test, np.arange(len(test), dtype=np.int64)),
            int(request.get("torch_batch_size", 4096)),
            device,
        )
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("frozen inference returned non-finite scores")

    manifest = json.loads(manifest_path.read_text())
    xsec = json.loads(xsec_path.read_text())
    signal_sumw, stop_xsec = normalization_maps(manifest, xsec)
    luminosity_pb = float(manifest["normalization"]["luminosity_pb"])
    category = category_ids(test.global_features)
    point_rows = np.unique(
        np.stack((test.signal_topology_id, test.mstop, test.mlsp), axis=1), axis=0
    )
    histograms: dict[str, Any] = {}
    records = []
    excluded = []
    for topology_id, mstop, mlsp in point_rows:
        topology = TOPOLOGY.get(int(topology_id))
        if topology is None:
            excluded.append({"reason": "unknown_topology", "topology_id": int(topology_id),
                             "mStop": int(mstop), "mLSP": int(mlsp)})
            continue
        denominator = signal_sumw.get((topology, int(mstop), int(mlsp)))
        cross_section = stop_xsec.get(int(mstop))
        if denominator is None or cross_section is None:
            excluded.append({"reason": "missing_normalization", "topology": topology,
                             "mStop": int(mstop), "mLSP": int(mlsp)})
            continue
        selected_point = (
            (test.signal_topology_id == topology_id)
            & (test.mstop == mstop)
            & (test.mlsp == mlsp)
        )
        weights = (
            test.gen_weight[selected_point]
            * cross_section
            * luminosity_pb
            / denominator
            / 0.70
        )
        sample = f"{topology}_mStop{int(mstop)}_mLSP{int(mlsp)}"
        histograms[sample] = {}
        point_scores = scores[selected_point]
        point_categories = category[selected_point]
        for category_id, label in enumerate(CATEGORY_LABELS):
            local = point_categories == category_id
            histograms[sample][label] = counter(point_scores[local], weights[local], edges[label])
        records.append({
            "sample": sample,
            "topology": topology,
            "mStop": int(mstop),
            "mLSP": int(mlsp),
            "deltaM": int(mstop - mlsp),
            "test_events": int(np.count_nonzero(selected_point)),
            "yield": float(np.sum(weights)),
        })

    payload = {
        "schema_version": "gnn_lowdm_all_signal_template_partial_v1",
        "status": "complete" if not excluded else "complete_with_exclusions",
        "selection": (
            "feature_lowdm_preselection && Nb>=1 && Nt=0 && NW=0 && Nres=0; "
            "no MET/sqrtHT, NISR, ISR-dphi, feature_SR, deltaM, mStop, or mLSP cut"
        ),
        "partition": "deterministic 70% test, normalized by 1/0.70",
        "input_root": str(source),
        "input_events_selected": int(len(events)),
        "test_events": int(len(test)),
        "mass_points": len(records),
        "records": records,
        "exclusions": excluded,
        "configuration": configuration,
        "histograms": histograms,
    }
    write_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "input": str(source), "test_events": len(test),
        "mass_points": len(records), "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
