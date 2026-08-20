#!/usr/bin/env python3
"""Measure and plot AN-style S_gamma factors versus U_T for 2024."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import build_an_zinv_factors_2024 as zinv


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nominal_arrays(payload: dict[str, Any], nbin: int) -> tuple[np.ndarray, np.ndarray]:
    nominal = (payload or {}).get("nominal") or {}
    values = np.asarray(nominal.get("sumw") or [], dtype=float)
    variances = np.asarray(nominal.get("sumw2") or [], dtype=float)
    if len(values) != nbin or len(variances) != nbin:
        raise ValueError(
            f"data_obs has {len(values)}/{len(variances)} bins, expected {nbin}"
        )
    return values, variances


def data_leaf(value: float, variance: float) -> dict[str, float]:
    return {"sumw": float(value), "sumw2": float(variance)}


def split_data_and_mc(exact: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract GCR data_obs and return an MC-only copy for the factor code."""
    mc_exact = copy.deepcopy(exact)
    high_nbin = len(exact["highdm"]["recoil_edges"]) - 1
    high_yields: dict[str, dict[str, dict[str, float]]] = {}
    for group in zinv.HIGH_GROUPS:
        source = mc_exact["highdm"]["recoil"]["GCR"][group]
        if "data_obs" not in source:
            raise ValueError(f"missing EGamma data_obs in High-dM GCR/{group}")
        values, variances = nominal_arrays(source.pop("data_obs"), high_nbin)
        high_yields[group] = {
            str(index): data_leaf(values[index], variances[index])
            for index in range(high_nbin)
        }

    low_nbin = len(exact["lowdm"]["search_bin_labels"])
    low_values = np.zeros(low_nbin, dtype=float)
    low_variances = np.zeros(low_nbin, dtype=float)
    for group in zinv.LOW_GROUPS:
        source = mc_exact["lowdm"]["search_components"]["GCR"][group]
        if "data_obs" not in source:
            raise ValueError(f"missing EGamma data_obs in Low-dM GCR/{group}")
        values, variances = nominal_arrays(source.pop("data_obs"), low_nbin)
        low_values += values
        low_variances += variances

    measurement = {
        "gcr_data": {
            "highdm": {"yields": high_yields},
            "lowdm": {
                "yields": {
                    str(index): data_leaf(low_values[index], low_variances[index])
                    for index in range(low_nbin)
                }
            },
        }
    }
    return measurement, mc_exact


def validate_samples(
    exact: dict[str, Any], campaign_year: str = "2024"
) -> dict[str, Any]:
    datasets = sorted((exact.get("summary") or {}).get("datasets") or {})
    forbidden = [name for name in datasets if "PTLL" in name.upper()]
    if forbidden:
        raise ValueError(f"forbidden PTLL DY inputs found: {forbidden[:5]}")
    required_dy = {
        "DYto2E-4Jets": any("DYto2E-4Jets" in name for name in datasets),
        "DYto2Mu-4Jets": any("DYto2Mu-4Jets" in name for name in datasets),
        "DYto2Tau-4Jets": any("DYto2Tau-4Jets" in name for name in datasets),
    }
    missing = [name for name, present in required_dy.items() if not present]
    if missing:
        raise ValueError(f"required DY2x samples absent: {missing}")
    data = [name for name in datasets if f"Run{campaign_year}" in name]
    gj = [name for name in datasets if name.startswith("GJ") or "GJets" in name]
    qcd = [name for name in datasets if name.startswith("QCD")]
    allowed_data_prefixes = (
        ("EGamma0-", "EGamma1-")
        if campaign_year == "2024"
        else ("EGamma0-", "EGamma1-", "EGamma2-", "EGamma3-")
    )
    gcr_data = [name for name in data if name.startswith(allowed_data_prefixes)]
    regions = set((exact.get("provenance") or {}).get("regions") or [])
    auxiliary_prefixes: tuple[str, ...] = ()
    if "LLCR" in regions:
        auxiliary_prefixes += ("Muon0-", "Muon1-")
    if "QCDCR" in regions or "SR" in regions:
        auxiliary_prefixes += (
            "JetMET0-",
            "JetMET1-",
            "JetMET2-",
            "JetMET3-",
        )
    unexpected_data = [
        name
        for name in data
        if not name.startswith(allowed_data_prefixes + auxiliary_prefixes)
    ]
    if not gcr_data or unexpected_data:
        raise ValueError(
            "data inputs do not match the region-gated streams for "
            f"{campaign_year}: {unexpected_data[:5]}"
        )
    if not gj or any(not name.startswith("GJ-4Jets_Bin-HT-") for name in gj):
        raise ValueError("GJ inputs are not exclusively the adopted GJ-4Jets HTxPTG family")
    if not qcd or any(not name.startswith("QCD-4Jets_Bin-HT-") for name in qcd):
        raise ValueError("QCD inputs are not exclusively the adopted QCD-4Jets HT family")
    return {
        "dataset_count": len(datasets),
        "data_dataset_records": len(data),
        "gcr_data_dataset_records": len(gcr_data),
        "auxiliary_region_data_dataset_records": len(data) - len(gcr_data),
        "data_family": "/".join(
            prefix[:-1] if prefix.endswith("-") else prefix
            for prefix in allowed_data_prefixes
        )
        + " only",
        "gj_physical_datasets": len(gj),
        "gj_family": "GJ-4Jets HTxPTG, dRGJ>0.25",
        "qcd_physical_datasets": len(qcd),
        "qcd_family": "QCD-4Jets HT",
        "forbidden_ptll_count": len(forbidden),
        "required_dy2x_present": required_dy,
    }


def closure_checks(factors: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {"highdm": {}, "lowdm": {}}
    for group, payload in factors["highdm"].items():
        q = payload["Q"]
        numerator = sum(item["Sgamma"]["numerator"] for item in payload["bins"])
        denominator = float(q["value"]) * sum(
            item["gamma_mc"] for item in payload["bins"]
        )
        checks["highdm"][group] = {
            "summed_numerator": float(numerator),
            "Q_times_summed_gamma": float(denominator),
            "relative_residual": float(
                (numerator - denominator) / denominator if denominator else 0.0
            ),
        }
    for group, q in factors["lowdm_Q_groups"].items():
        selected = [
            payload
            for payload in factors["lowdm"].values()
            if payload["group"] == group
        ]
        numerator = sum(
            item["Sgamma"]["numerator"]
            for payload in selected
            for item in payload["bins"]
        )
        gamma = sum(
            item["gamma_mc"]
            for payload in selected
            for item in payload["bins"]
        )
        denominator = float(q["value"]) * gamma
        checks["lowdm"][group] = {
            "summed_numerator": float(numerator),
            "Q_times_summed_gamma": float(denominator),
            "relative_residual": float(
                (numerator - denominator) / denominator if denominator else 0.0
            ),
        }
    return checks


def write_csv(path: Path, factors: dict[str, Any], high_edges: list[float]) -> None:
    fields = [
        "regime",
        "category",
        "ut_low_gev",
        "ut_high_gev",
        "last_bin_includes_overflow",
        "Q",
        "Q_stat",
        "Sgamma",
        "Sgamma_stat",
        "status",
        "data_minus_other",
        "Q_times_gamma",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in zinv.HIGH_GROUPS:
            payload = factors["highdm"][group]
            q = payload["Q"]
            for index, item in enumerate(payload["bins"]):
                factor = item["Sgamma"]
                writer.writerow(
                    {
                        "regime": "highdm",
                        "category": group,
                        "ut_low_gev": high_edges[index],
                        "ut_high_gev": high_edges[index + 1],
                        "last_bin_includes_overflow": index == len(payload["bins"]) - 1,
                        "Q": q["value"],
                        "Q_stat": q["stat"],
                        "Sgamma": factor["value"],
                        "Sgamma_stat": factor["stat"],
                        "status": factor["status"],
                        "data_minus_other": factor["numerator"],
                        "Q_times_gamma": factor["denominator"],
                    }
                )
        for family, payload in factors["lowdm"].items():
            q = payload["Q"]
            isr_group = (
                "PISR300to500"
                if "PISR300to500" in family
                else "PISR500plus"
            )
            centers, widths = zinv.low_ut_geometry(
                isr_group, len(payload["bins"])
            )
            for index, item in enumerate(payload["bins"]):
                factor = item["Sgamma"]
                writer.writerow(
                    {
                        "regime": "lowdm",
                        "category": family,
                        "ut_low_gev": centers[index] - widths[index],
                        "ut_high_gev": centers[index] + widths[index],
                        "last_bin_includes_overflow": index == len(payload["bins"]) - 1,
                        "Q": q["value"],
                        "Q_stat": q["stat"],
                        "Sgamma": factor["value"],
                        "Sgamma_stat": factor["stat"],
                        "status": factor["status"],
                        "data_minus_other": factor["numerator"],
                        "Q_times_gamma": factor["denominator"],
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--campaign-year", choices=("2024", "2025"), default="2024"
    )
    args = parser.parse_args()
    zinv.CMS_LABEL["rlabel"] = f"{args.campaign_year} (13.6 TeV)"

    exact = json.loads(args.exact.read_text())
    if exact.get("status") != "complete":
        raise ValueError(f"exact input is not complete: {exact.get('status')}")
    sample_check = validate_samples(exact, args.campaign_year)
    measurement, mc_exact = split_data_and_mc(exact)
    factors = zinv.build_q_sgamma(measurement, mc_exact)
    low_shared = zinv.aggregate_low_sgamma(factors["lowdm"])
    checks = closure_checks(factors)
    max_residual = max(
        abs(item["relative_residual"])
        for regime in checks.values()
        for item in regime.values()
    )
    if max_residual > 1.0e-10:
        raise ValueError(f"Q/Sgamma closure residual is too large: {max_residual}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    plot_paths.extend(zinv.plot_q(factors, args.output_dir))
    plot_paths.extend(zinv.plot_sgamma(factors["highdm"], "highdm", args.output_dir))
    plot_paths.extend(zinv.plot_sgamma(factors["lowdm"], "lowdm", args.output_dir))
    payload = {
        "schema_version": f"sgamma_ut_{args.campaign_year}_v1",
        "status": "complete",
        "definition": {
            "Q_g": "sum_i(data_i - otherMC_i) / sum_i(gammaJetsMC_i)",
            "Sgamma_g_i": "(data_i - otherMC_i) / (Q_g * gammaJetsMC_i)",
            "gcr_data_stream": "EGamma",
            "other_mc": "all normalized GCR MC except GJ",
            "uncertainty": "diagonal statistical propagation used by the AN-style implementation",
            "lowdm_plot_overflow_cap_gev": 1500.0,
            "lowdm_plot_mode": "all 34 adopted category bins; no pTb/Nj family aggregation",
        },
        "provenance": {
            "exact_input": str(args.exact),
            "exact_sha256": file_sha256(args.exact),
            "exact_provenance": exact.get("provenance"),
            "sample_check": sample_check,
            "nominal_sr_data_recorded": False,
            "campaign_year": args.campaign_year,
        },
        "highdm": factors["highdm"],
        "lowdm_families": factors["lowdm"],
        "lowdm_Q_groups": factors["lowdm_Q_groups"],
        "lowdm_aggregated_diagnostic": low_shared,
        "closure_checks": checks,
        "plots": plot_paths,
    }
    zinv.write_json(args.output_dir / "sgamma_ut.json", payload)
    write_csv(
        args.output_dir / "sgamma_ut.csv",
        factors,
        list(exact["highdm"]["recoil_edges"]),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir),
                "plots": plot_paths,
                "max_closure_residual": max_residual,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
