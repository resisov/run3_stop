#!/usr/bin/env python3
"""Audit low-pT J/psi->ee probes in an independently triggered parking file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


PARKING_PATHS = (
    "HLT_Mu9_Barrel_L1HP10_IP6",
    "HLT_Mu10_Barrel_L1HP11_IP6",
)


def invariant_mass(left: ak.Array, right: ak.Array) -> ak.Array:
    px1 = left.pt * np.cos(left.phi)
    py1 = left.pt * np.sin(left.phi)
    pz1 = left.pt * np.sinh(left.eta)
    px2 = right.pt * np.cos(right.phi)
    py2 = right.pt * np.sin(right.phi)
    pz2 = right.pt * np.sinh(right.eta)
    e1 = np.sqrt(np.maximum(0.0, left.mass**2 + px1**2 + py1**2 + pz1**2))
    e2 = np.sqrt(np.maximum(0.0, right.mass**2 + px2**2 + py2**2 + pz2**2))
    mass2 = (e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    return np.sqrt(np.maximum(0.0, mass2))


def audit(path: str, step_size: int) -> dict[str, object]:
    tree = uproot.open(path)["Events"]
    present = set(tree.keys())
    paths = [name for name in PARKING_PATHS if name in present]
    if not paths:
        raise RuntimeError("no configured ParkingSingleMuon reference path is present")
    branches = [
        *paths,
        "Electron_pt",
        "Electron_eta",
        "Electron_phi",
        "Electron_mass",
        "Electron_charge",
        "Electron_cutBased",
        "Electron_miniPFRelIso_all",
        "Electron_convVeto",
        "Electron_lostHits",
    ]
    pt_edges = np.asarray([5.0, 7.0, 10.0, 15.0])
    eta_edges = np.asarray([0.0, 0.8, 1.44, 1.57, 2.5])
    passed = np.zeros((len(eta_edges) - 1, len(pt_edges) - 1), dtype=np.int64)
    failed = np.zeros_like(passed)
    events_read = 0
    events_reference = 0
    pairs = 0
    for arrays in tree.iterate(branches, step_size=step_size, library="ak"):
        events_read += len(arrays[paths[0]])
        reference = np.zeros(len(arrays[paths[0]]), dtype=bool)
        for name in paths:
            reference |= np.asarray(arrays[name], dtype=bool)
        events_reference += int(np.sum(reference))
        if not np.any(reference):
            continue
        arrays = arrays[reference]
        pt = arrays["Electron_pt"]
        eta = arrays["Electron_eta"]
        index = ak.local_index(pt, axis=1)
        fiducial = (abs(eta) < 1.44) | ((abs(eta) > 1.57) & (abs(eta) < 2.5))
        denominator = (
            (pt > 5.0)
            & (pt < 15.0)
            & fiducial
            & arrays["Electron_convVeto"]
            & (arrays["Electron_lostHits"] <= 1)
        )
        tag = (
            denominator
            & (arrays["Electron_cutBased"] >= 4)
            & (arrays["Electron_miniPFRelIso_all"] < 0.1)
        )
        records = ak.zip(
            {
                "pt": pt,
                "eta": eta,
                "phi": arrays["Electron_phi"],
                "mass": arrays["Electron_mass"],
                "charge": arrays["Electron_charge"],
                "index": index,
                "passing": arrays["Electron_cutBased"] >= 1,
            }
        )
        candidates = ak.cartesian({"tag": records[tag], "probe": records[denominator]}, axis=1)
        candidates = candidates[
            (candidates.tag.index != candidates.probe.index)
            & (candidates.tag.charge * candidates.probe.charge < 0)
        ]
        mass = invariant_mass(candidates.tag, candidates.probe)
        selected = (mass >= 2.0) & (mass <= 4.0)
        candidates = candidates[selected]
        flat_pt = np.asarray(ak.to_numpy(ak.flatten(candidates.probe.pt, axis=1)))
        flat_eta = np.abs(np.asarray(ak.to_numpy(ak.flatten(candidates.probe.eta, axis=1))))
        flat_pass = np.asarray(ak.to_numpy(ak.flatten(candidates.probe.passing, axis=1)), dtype=bool)
        pairs += len(flat_pt)
        passed += np.histogram2d(flat_eta[flat_pass], flat_pt[flat_pass], bins=(eta_edges, pt_edges))[0].astype(np.int64)
        failed += np.histogram2d(flat_eta[~flat_pass], flat_pt[~flat_pass], bins=(eta_edges, pt_edges))[0].astype(np.int64)
    return {
        "file": path,
        "reference_paths": paths,
        "electron_trigger_requirement": False,
        "probe_definition": "veto_id_only",
        "pt_edges_gev": pt_edges.tolist(),
        "abseta_edges": eta_edges.tolist(),
        "events_read": events_read,
        "events_reference": events_reference,
        "pairs_2to4_gev": pairs,
        "pass": passed.tolist(),
        "fail": failed.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--step-size", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.input, args.step_size)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
