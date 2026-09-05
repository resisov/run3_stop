#!/usr/bin/env python3
"""Validate SR feature reconstruction against the frozen training loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from . import region_io as base
from ..data import DIAGONAL_V3_GLOBAL_FEATURE_NAMES, _read_one
from .diagonal_v3_region_features import feature_arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    opts = parser.parse_args()
    expected = _read_one(
        opts.cache,
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
    if expected is None:
        raise RuntimeError("training loader returned no events")
    with uproot.open(opts.cache) as root_file:
        arrays = root_file["Events"].arrays(library="ak")
    count = len(arrays)
    block = base.RegionBlock(
        core=np.ones(count, dtype=bool),
        recoil=np.asarray(arrays["met"], dtype=float),
        recoil_phi=np.asarray(arrays["met_phi"], dtype=float),
        ht=np.asarray(arrays["ht"], dtype=float),
        njet=np.asarray(arrays["njet"], dtype=int),
        nb=np.asarray(arrays["nb_medium"], dtype=int),
        nt=np.zeros(count, dtype=int),
        nw=np.zeros(count, dtype=int),
        nisr=np.asarray(arrays["n_lowdm_isr"], dtype=int),
        met_sqrt_ht=np.asarray(arrays["lowdm_met_sqrt_ht"], dtype=float),
        mtb=np.asarray(arrays["lowdm_mtb"], dtype=float),
        ptb=np.asarray(arrays["lowdm_ptb"], dtype=float),
    )
    rebuilt = feature_arrays(
        arrays, block, "SR", np.ones(count, dtype=bool), max_jets=10
    )
    names = ("node_features", "node_mask", "node_eta", "node_phi", "global_features")
    expected_values = (
        expected.node_features,
        expected.node_mask,
        expected.node_eta,
        expected.node_phi,
        expected.global_features,
    )
    comparison = {}
    for name, left, right in zip(names, rebuilt[:5], expected_values):
        difference = np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
        comparison[name] = {
            "shape": list(np.shape(left)),
            "maximum_absolute_difference": float(np.max(difference)) if difference.size else 0.0,
            "mismatched_values_at_1e_6": int(np.count_nonzero(difference > 1.0e-6)),
        }
    global_difference = np.abs(
        np.asarray(rebuilt[4], dtype=float)
        - np.asarray(expected.global_features, dtype=float)
    )
    comparison["global_features"]["by_feature"] = {
        feature: {
            "maximum_absolute_difference": float(np.max(global_difference[:, index])),
            "mismatched_values_at_1e_6": int(
                np.count_nonzero(global_difference[:, index] > 1.0e-6)
            ),
        }
        for index, feature in enumerate(DIAGONAL_V3_GLOBAL_FEATURE_NAMES)
        if np.any(global_difference[:, index] > 1.0e-6)
    }
    status = "complete" if all(
        record["mismatched_values_at_1e_6"] == 0
        for record in comparison.values()
    ) else "failed"
    payload = {
        "schema_version": "diagonal_v3_region_feature_parity_v1",
        "status": status,
        "cache": str(opts.cache),
        "events": count,
        "comparison": comparison,
        "reconstruction_audit": rebuilt[5],
    }
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
