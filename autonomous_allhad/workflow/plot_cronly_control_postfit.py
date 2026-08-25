#!/usr/bin/env python3
"""Render High-dM CR MET/recoil postfit distributions from FitDiagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import uproot

from background_process_groups import BACKGROUND_DISPLAY_LABELS, BACKGROUND_PROCESS_ORDER
from plot_control_search_bins_style import GROUP_ORDER, draw_flat_blocks


CHANNEL = re.compile(
    r"^y(?P<year>2024|2025)_(?P<region>LLCR|QCDCR|GCR)_highdm_"
    r"(?P<nb>Nb1|Nb2|Nb3plus)_bin(?P<bin>[0-5])$"
)
REGIONS = ("LLCR", "QCDCR", "GCR")
NB_CATEGORIES = ("Nb1", "Nb2", "Nb3plus")
YEARS = ("2024", "2025")
RECOIL_LABELS = ("250–300", "300–350", "350–400", "400–500", "500–800", "≥800")
RECOIL_DISPLAY_EDGES = (250.0, 300.0, 350.0, 400.0, 500.0, 800.0, 1500.0)
LUMINOSITY = {"2024": 109.82, "2025": 110.84, "combined": 220.66}
REGION_LABELS = {
    "LLCR": "Lost-lepton control region",
    "QCDCR": "Multijet control region",
    "GCR": "Photon control region",
}
NB_LABELS = {"Nb1": "$N_b=1$", "Nb2": "$N_b=2$", "Nb3plus": "$N_b\\geq3$"}


def graph_record(graph: object) -> tuple[float, float]:
    _x, y = graph.values()
    low = np.asarray(graph.member("fEYlow"), dtype=float)
    high = np.asarray(graph.member("fEYhigh"), dtype=float)
    return float(y[0]), float(0.5 * (low[0] + high[0]))


def extract_channel(directory: object) -> dict:
    keys = set(directory.keys(cycle=False))
    groups = {group: 0.0 for group in GROUP_ORDER}
    processes = {}
    for process in BACKGROUND_PROCESS_ORDER:
        value = float(directory[process].values()[0]) if process in keys else 0.0
        processes[process] = value
        groups[BACKGROUND_DISPLAY_LABELS[process]] += value
    total_hist = directory["total_background"]
    total = float(total_hist.values()[0])
    variance = float(total_hist.variances()[0])
    data, data_unc = graph_record(directory["data"])
    stack_total = float(sum(processes.values()))
    if not math.isclose(stack_total, total, rel_tol=2.0e-6, abs_tol=1.0e-5):
        raise RuntimeError(f"postfit process stack does not close: {stack_total} != {total}")
    if not all(math.isfinite(value) for value in (*groups.values(), total, variance, data, data_unc)):
        raise RuntimeError("nonfinite postfit content")
    return {
        "groups": groups,
        "processes": processes,
        "total": total,
        "uncertainty": math.sqrt(max(variance, 0.0)),
        "data": data,
        "data_uncertainty": data_unc,
    }


def make_block(records: list[dict], annotation: str, labels: list[str]) -> dict:
    groups = {
        group: np.asarray([record["groups"][group] for record in records], dtype=float)
        for group in GROUP_ORDER
    }
    total = np.asarray([record["total"] for record in records], dtype=float)
    return {
        "groups": groups,
        "background": total,
        "background_unc": np.asarray([record["uncertainty"] for record in records], dtype=float),
        "background_stat_unc": np.asarray([record["uncertainty"] for record in records], dtype=float),
        "data": np.asarray([record["data"] for record in records], dtype=float),
        "data_unc": np.asarray([record["data_uncertainty"] for record in records], dtype=float),
        "signals": {},
        "label": annotation,
        "annotation": annotation,
        "nbin": len(records),
        # Preserve the physical CR histogram geometry.  The final displayed
        # interval is only a plotting width; its content is the >=800 GeV bin
        # with the full overflow already folded into it.
        "edges": list(RECOIL_DISPLAY_EDGES),
        "xlabels": [],
        "blind_data": False,
        "reference_style": True,
        "label_box": False,
        "show_annotation": True,
        "annotation_x": 0.68,
        "annotation_y": 0.68,
        "group_labels": {},
        "significance_panel": False,
    }


def sum_year_records(left: dict, right: dict) -> dict:
    return {
        "groups": {
            group: float(left["groups"][group] + right["groups"][group])
            for group in GROUP_ORDER
        },
        "processes": {
            process: float(left["processes"][process] + right["processes"][process])
            for process in BACKGROUND_PROCESS_ORDER
        },
        "total": float(left["total"] + right["total"]),
        "uncertainty": float(
            math.hypot(left["uncertainty"], right["uncertainty"])
        ),
        "data": float(left["data"] + right["data"]),
        "data_uncertainty": float(
            math.hypot(left["data_uncertainty"], right["data_uncertainty"])
        ),
    }
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-diagnostics", required=True, type=Path)
    parser.add_argument("--fit-status", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    status = json.loads(args.fit_status.read_text())
    if status.get("status") != "complete" or int(status.get("fit_status", -1)) != 0:
        raise SystemExit("CR-only fit is not complete")
    if int(status.get("covariance_quality", -1)) < 3:
        raise SystemExit("CR-only covariance quality is below 3")
    if status.get("vr_observation_in_likelihood") is not False:
        raise SystemExit("VR observation entered the likelihood")
    if status.get("sr_observation_in_likelihood") is not False:
        raise SystemExit("SR observation entered the likelihood")

    source = uproot.open(args.fit_diagnostics)
    shapes = source["shapes_fit_b"]
    extracted: dict[tuple[str, str, str, int], dict] = {}
    ignored = []
    for channel_name in shapes.keys(recursive=False, cycle=False):
        match = CHANNEL.fullmatch(channel_name)
        if not match:
            ignored.append(channel_name)
            continue
        key = (
            match.group("year"),
            match.group("region"),
            match.group("nb"),
            int(match.group("bin")),
        )
        extracted[key] = extract_channel(shapes[channel_name])

    expected = len(YEARS) * len(REGIONS) * len(NB_CATEGORIES) * len(RECOIL_LABELS)
    if len(extracted) != expected:
        raise SystemExit(f"expected {expected} High-dM CR channels, found {len(extracted)}")

    plots = []
    channel_summary = {}
    for region in REGIONS:
        for nb in NB_CATEGORIES:
            annotation = f"{REGION_LABELS[region]}\n{NB_LABELS[nb]} · CR-only postfit"
            axis_label = r"$U_{T}$ (GeV)" if region == "GCR" else r"$p_{T}^{miss}$ (GeV)"
            for year in YEARS:
                records = [extracted[(year, region, nb, index)] for index in range(6)]
                block = make_block(records, annotation, list(RECOIL_LABELS))
                outbase = args.output_dir / year / f"{region}_{nb}_met_postfit"
                record = draw_flat_blocks(
                    [block],
                    outbase,
                    xlabel=axis_label,
                    reference_style=True,
                    show_yields=True,
                    ratio_ylabel="Data/Pred.",
                    uncertainty_label_override="Total postfit unc.",
                    luminosity_fb=LUMINOSITY[year],
                )
                record.update({"scope": year, "region": region, "nb": nb})
                plots.append(record)
                channel_summary[f"{year}/{region}/{nb}"] = records

            combined_records = [
                sum_year_records(
                    extracted[("2024", region, nb, index)],
                    extracted[("2025", region, nb, index)],
                )
                for index in range(6)
            ]
            combined_block = make_block(combined_records, annotation, list(RECOIL_LABELS))
            outbase = args.output_dir / "combined" / f"{region}_{nb}_met_postfit"
            record = draw_flat_blocks(
                [combined_block],
                outbase,
                xlabel=axis_label,
                reference_style=True,
                show_yields=True,
                ratio_ylabel="Data/Pred.",
                uncertainty_label_override="Total postfit unc.",
                luminosity_fb=LUMINOSITY["combined"],
            )
            record.update(
                {
                    "scope": "combined",
                    "region": region,
                    "nb": nb,
                    "combination": "2024 and 2025 yields summed bin-by-bin on the original CR physical-bin axis",
                    "combined_uncertainty": "quadrature sum of the two per-year postfit uncertainty bands",
                }
            )
            plots.append(record)

    if len(plots) != 27:
        raise SystemExit(f"expected 27 plots, produced {len(plots)}")

    digest = hashlib.sha256()
    with args.fit_diagnostics.open("rb") as fit_source:
        for chunk in iter(lambda: fit_source.read(1024 * 1024), b""):
            digest.update(chunk)
    payload = {
        "status": "complete",
        "fit": "2024+2025 observed CR-only background-only fit",
        "fit_status": int(status["fit_status"]),
        "covariance_quality": int(status["covariance_quality"]),
        "likelihood_channel_count": int(status["likelihood_channel_count"]),
        "vr_observation_in_likelihood": False,
        "sr_observation_in_likelihood": False,
        "highdm_cr_channel_count": len(extracted),
        "fit_diagnostics_sha256": digest.hexdigest(),
        "plot_count": len(plots),
        "plots": plots,
        "recoil_bin_labels_gev": list(RECOIL_LABELS),
        "recoil_display_edges_gev": list(RECOIL_DISPLAY_EDGES),
        "overflow_policy": "last bin includes all pTmiss/recoil >= 800 GeV",
        "ignored_non_highdm_cr_shape_directories": sorted(ignored),
        "channels": channel_summary,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "channels": len(extracted), "plots": len(plots)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
