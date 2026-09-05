#!/usr/bin/env python3
"""Validate portable NumPy inference against the frozen PyTorch checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data import _read_one
from ..model import PhysicsInformedJetGraphClassifier
from .rank005_numpy import Rank005Numpy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--numpy-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--events", type=int, default=181)
    args = parser.parse_args()

    events = _read_one(
        args.cache,
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
    )
    if events is None or not len(events):
        raise RuntimeError("cache shard contains no diagonal-v3 events")
    count = min(int(args.events), len(events))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = PhysicsInformedJetGraphClassifier(
        node_features=events.node_features.shape[2],
        global_features=events.global_features.shape[1],
        hidden=int(config["hidden"]),
        message_layers=int(config["message_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(events.node_features[:count]),
            torch.from_numpy(events.node_mask[:count]),
            torch.from_numpy(events.node_eta[:count]),
            torch.from_numpy(events.node_phi[:count]),
            torch.from_numpy(events.global_features[:count]),
        )
        torch_scores = torch.sigmoid(logits).numpy()
    numpy_scores = Rank005Numpy(args.numpy_model).predict(
        events.node_features[:count],
        events.node_mask[:count],
        events.node_eta[:count],
        events.node_phi[:count],
        events.global_features[:count],
    )
    difference = np.abs(torch_scores.astype(float) - numpy_scores.astype(float))
    payload = {
        "schema_version": "diagonal_v3_numpy_inference_parity_v1",
        "status": "complete" if float(np.max(difference)) < 2.0e-6 else "failed",
        "events": count,
        "maximum_absolute_difference": float(np.max(difference)),
        "mean_absolute_difference": float(np.mean(difference)),
        "mismatches_at_2e_6": int(np.count_nonzero(difference >= 2.0e-6)),
        "torch_score_range": [float(np.min(torch_scores)), float(np.max(torch_scores))],
        "numpy_score_range": [float(np.min(numpy_scores)), float(np.max(numpy_scores))],
        "checkpoint": str(args.checkpoint),
        "numpy_model": str(args.numpy_model),
        "cache": str(args.cache),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
