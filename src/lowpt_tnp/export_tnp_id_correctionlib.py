#!/usr/bin/env python3
"""Export electron or muon low-pT ID-only TnP results to correctionlib."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .sf_payload import (
    correction,
    correction_set,
    ensure_adopted,
    write_json_gz,
)


DEFINITIONS = {
    "electron": {
        "probe_definition": "veto_id_only",
        "correction": "veto_electron_id_5to10_sf",
        "description": "{year} data/MC SF for the analysis veto-electron ID, excluding isolation, 5 < pT < 10 GeV",
        "set_description": "{year} Run-3 all-hadronic stop low-pT veto-electron ID-only scale factor",
    },
    "muon": {
        "probe_definition": "loose_id_only",
        "correction": "loose_muon_id_5to10_sf",
        "description": "{year} data/MC SF for analysis LooseID relative to tracker muons, excluding isolation, 5 < pT < 10 GeV",
        "set_description": "{year} Run-3 all-hadronic stop low-pT loose-muon ID-only scale factor",
    },
}


def _lowpt_bins(result: dict[str, Any]) -> tuple[list[float], list[dict[str, Any]]]:
    """Select exactly 5--10 GeV from an optional wider validation fit."""

    eta_edges = [float(value) for value in result["probe_abseta_edges"]]
    pt_edges = [float(value) for value in result["probe_pt_edges_gev"]]
    bins = list(result.get("bins") or [])
    try:
        pt_low = pt_edges.index(5.0)
        pt_high = pt_edges.index(10.0)
    except ValueError as exc:
        raise ValueError("ID-only TnP result must contain exact 5 and 10 GeV edges") from exc
    if pt_low >= pt_high:
        raise ValueError("invalid 5--10 GeV TnP edge ordering")
    n_eta = len(eta_edges) - 1
    n_pt = len(pt_edges) - 1
    if len(bins) != n_eta * n_pt:
        raise ValueError("TnP bin count does not match eta/pT axes")
    selected = [
        bins[eta_index * n_pt + pt_index]
        for eta_index in range(n_eta)
        for pt_index in range(pt_low, pt_high)
    ]
    return pt_edges[pt_low : pt_high + 1], selected


def _unity_fallback_uncertainty(entry: dict[str, Any]) -> float:
    """Return a measurement-derived symmetric uncertainty about a unity fallback.

    The central value is fixed to one by analysis policy.  The uncertainty covers
    both the fitted Data/MC ratio's displacement from unity and its propagated
    statistical uncertainty, added in quadrature.
    """

    nominal = entry.get("fits", {}).get("nominal", {})
    data = nominal.get("data", {})
    simulation = nominal.get("mc", {})
    data_efficiency = float(data["efficiency"])
    simulation_efficiency = float(simulation["efficiency"])
    data_uncertainty = float(data["efficiency_stat_uncertainty"])
    simulation_uncertainty = float(simulation["efficiency_stat_uncertainty"])
    values = (
        data_efficiency,
        simulation_efficiency,
        data_uncertainty,
        simulation_uncertainty,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("electron endcap unity fallback requires finite nominal fit values")
    if data_efficiency <= 0.0 or simulation_efficiency <= 0.0:
        raise ValueError("electron endcap unity fallback requires positive fitted efficiencies")
    if data_uncertainty < 0.0 or simulation_uncertainty < 0.0:
        raise ValueError("electron endcap unity fallback requires non-negative fit uncertainties")
    raw_scale_factor = data_efficiency / simulation_efficiency
    raw_statistical_uncertainty = raw_scale_factor * math.hypot(
        data_uncertainty / data_efficiency,
        simulation_uncertainty / simulation_efficiency,
    )
    return math.hypot(raw_scale_factor - 1.0, raw_statistical_uncertainty)


def electron_unity_policy_uncertainty(entry: dict[str, Any]) -> float:
    """Cover the measured ratio and its total uncertainty about a unity central value."""

    if entry.get("valid"):
        raw_scale_factor = float(entry["scale_factor"])
        raw_uncertainty = float(entry["scale_factor_uncertainty"])
        if not math.isfinite(raw_scale_factor) or raw_scale_factor <= 0.0:
            raise ValueError("electron endcap unity policy requires a positive fitted scale factor")
        if not math.isfinite(raw_uncertainty) or raw_uncertainty < 0.0:
            raise ValueError("electron endcap unity policy requires a finite total uncertainty")
        return math.hypot(raw_scale_factor - 1.0, raw_uncertainty)
    return _unity_fallback_uncertainty(entry)


def build_payload(
    result: dict[str, Any],
    kind: str,
    *,
    electron_endcap_unity_fallback: bool = False,
) -> dict[str, Any]:
    definition = DEFINITIONS[kind]
    if result.get("probe_definition") != definition["probe_definition"]:
        raise ValueError(
            f"{kind} result has probe_definition={result.get('probe_definition')!r}; "
            f"expected {definition['probe_definition']!r}"
        )
    pt_edges, bins = _lowpt_bins(result)
    if not bins:
        raise ValueError("5--10 GeV ID-only TnP bins are missing")
    if electron_endcap_unity_fallback and kind != "electron":
        raise ValueError("the endcap unity fallback is defined only for electron payloads")

    n_pt = len(pt_edges) - 1
    n_eta = len(result["probe_abseta_edges"]) - 1
    forward_start = (n_eta - 1) * n_pt
    nominal: list[float] = []
    uncertainty: list[float] = []
    unity_policy_indices: list[int] = []
    for index, entry in enumerate(bins):
        if electron_endcap_unity_fallback and index >= forward_start:
            nominal.append(1.0)
            uncertainty.append(electron_unity_policy_uncertainty(entry))
            unity_policy_indices.append(index)
            continue
        if entry.get("valid"):
            nominal.append(float(entry["scale_factor"]))
            uncertainty.append(float(entry["scale_factor_uncertainty"]))
            continue
        if not electron_endcap_unity_fallback:
            raise ValueError("all 5--10 GeV ID-only TnP bins must be present and valid before export")
        if index < forward_start:
            raise ValueError("electron endcap unity fallback cannot replace an invalid barrel bin")
        raise AssertionError("unreachable endcap unity-policy branch")

    year = str(result.get("year") or "2024")
    description = definition["description"].format(year=year)
    set_description = definition["set_description"].format(year=year)
    if unity_policy_indices:
        policy = (
            "; highest-|eta| bins use a unity central value with a symmetric "
            "measurement-derived uncertainty from the nominal fitted Data/MC ratio"
        )
        description += policy
        set_description += policy
    item = correction(
        name=definition["correction"],
        description=description,
        axes=[
            ("abseta", result["probe_abseta_edges"]),
            ("pt", pt_edges),
        ],
        nominal=nominal,
        uncertainty=uncertainty,
    )
    return correction_set(set_description, [item])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--kind", choices=tuple(DEFINITIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="write a validation-pending candidate without requiring adopted status",
    )
    parser.add_argument(
        "--electron-endcap-unity-fallback",
        action="store_true",
        help=(
            "for electron payloads only, set every bin in the highest-|eta| interval "
            "to nominal 1.0 with a measurement-derived symmetric uncertainty"
        ),
    )
    args = parser.parse_args(argv)
    result = json.loads(args.result.read_text())
    if not args.candidate:
        ensure_adopted(result, args.result)
    payload = build_payload(
        result,
        args.kind,
        electron_endcap_unity_fallback=args.electron_endcap_unity_fallback,
    )
    digest = write_json_gz(args.output, payload)
    print(json.dumps({
        "status": "candidate" if args.candidate else "installed",
        "source_result": str(args.result),
        "output": str(args.output),
        "sha256": digest,
        "correction": DEFINITIONS[args.kind]["correction"],
        "bins": len(_lowpt_bins(result)[1]),
        "electron_endcap_unity_fallback": args.electron_endcap_unity_fallback,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
