#!/usr/bin/env python3
"""Build an expected-limit model using the six configured Low-dM regions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_boosted_an17_combine_inputs import (  # noqa: E402
    LUMI_LNN,
    LUMI_NAME,
    datacard_text,
    stable_path,
    write_hist,
    write_json,
)
from build_combine_inputs_from_preview import collect_limits, plot_contour  # noqa: E402
from build_flat_recoil_ntop_split_combine_inputs import write_parallel_runner  # noqa: E402
from background_process_groups import (  # noqa: E402
    aggregate_background_processes,
    background_grouping_contract,
    materialize_grouped_background_templates,
)
from build_flat_recoil_sr_combine_inputs import (  # noqa: E402
    SIGNAL_PREFIX,
    aggregate_nominal,
    aggregate_variations,
    hist_arrays,
    parse_mass_key,
    sample_to_mass_key,
    signal_process_name,
)

LOWDM_REGION_MAP = {
    "LLCR": "cat2_LLCR_lowDeltaM",
    "QCDCR": "cat3_QCDCR_lowDeltaM",
    "GCR": "cat4_GCR_lowDeltaM",
    "DY2E": "cat5_DY2E_lowDeltaM",
    "DY2M": "cat6_DY2M_lowDeltaM",
    "SR": "cat7_SR_lowDeltaM",
}
MIN_BIN = 1.0e-9


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def lowdm_channel(flat: dict[str, Any], region: str, scheme: str) -> dict[str, Any]:
    labels = list((((flat.get("search_bin_schemes") or {}).get(scheme) or {}).get("bin_labels") or []))
    if len(labels) not in {34, 42}:
        raise ValueError(
            f"{scheme} has {len(labels)} bins; supported Low-dM schemes have 34 or 42"
        )
    by_sample = ((flat.get("search_bin_histograms") or {}).get(scheme) or {})
    if not by_sample:
        raise ValueError(f"missing Low-dM histogram scheme {scheme}")
    bkg, bkg_s2, data, data_s2, backgrounds = aggregate_nominal(by_sample, len(labels))
    return {
        "name": scheme,
        "source_region": scheme,
        "region": region,
        "kind": (
            f"lowdm_{len(labels)}bin_control"
            if region != "SR"
            else f"lowdm_{len(labels)}bin_signal"
        ),
        "edges": np.arange(len(labels) + 1, dtype=float),
        "background": bkg,
        "background_sumw2": bkg_s2,
        "data": data,
        "data_sumw2": data_s2,
        "variations": aggregate_variations(by_sample, backgrounds, len(labels), bkg),
        "bin_labels": labels,
        "background_samples": backgrounds,
        "background_processes": aggregate_background_processes(
            by_sample, len(labels), hist_arrays, signal_prefix=SIGNAL_PREFIX
        ),
    }


def mass_points(flat: dict[str, Any], only: list[str] | None, max_mstop: int | None) -> list[str]:
    by_sample = ((flat.get("search_bin_histograms") or {}).get(LOWDM_REGION_MAP["SR"]) or {})
    selected: list[tuple[int, int, str]] = []
    for sample in by_sample:
        key = sample_to_mass_key(sample)
        if not key or (only and key not in only):
            continue
        mstop, mlsp = parse_mass_key(key)
        if mlsp >= mstop or (max_mstop is not None and mstop > max_mstop):
            continue
        nominal = ((by_sample.get(sample) or {}).get("nominal") or {}).get("sumw") or []
        if sum(float(value) for value in nominal) <= 0.0:
            continue
        selected.append((mstop, mlsp, key))
    return [key for _, _, key in sorted(set(selected))]


def signal_array(flat: dict[str, Any], scheme: str, mass_key: str) -> tuple[np.ndarray, np.ndarray]:
    nbin = len(
        (((flat.get("search_bin_schemes") or {}).get(scheme) or {}).get("bin_labels") or [])
    )
    if nbin <= 0:
        raise ValueError(f"missing bin labels for Low-dM scheme {scheme}")
    rec = (
        (((flat.get("search_bin_histograms") or {}).get(scheme) or {}).get(SIGNAL_PREFIX + mass_key) or {})
        .get("nominal")
    )
    return hist_arrays(rec, nbin)


def build_root(
    channels: list[dict[str, Any]],
    flat: dict[str, Any],
    selected_masses: list[str],
    output_root: Path,
    data_mode: str,
) -> dict[str, Any]:
    import ROOT

    output_root.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output_root), "RECREATE")
    summary: dict[str, Any] = {
        "channels": {},
        "signals": {},
        "background_shape_nuisances": sorted({name for ch in channels for name in ch["variations"]}),
        "background_grouping_contract": background_grouping_contract(),
    }
    try:
        for channel in channels:
            name = channel["name"]
            directory = root_file.mkdir(name)
            edges = np.asarray(channel["edges"], dtype=float)
            bkg = np.asarray(channel["background"], dtype=float)
            bkg_s2 = np.asarray(channel["background_sumw2"], dtype=float)
            data = bkg if data_mode == "asimov" else np.asarray(channel["data"], dtype=float)
            write_hist(directory, "data_obs", data, np.maximum(data, 0.0), edges)
            process_summary = materialize_grouped_background_templates(
                directory,
                channel.get("background_processes") or {},
                bkg,
                bkg_s2,
                edges,
                write_hist,
                min_bin=MIN_BIN,
            )
            summary["channels"][name] = {
                "region": channel["region"],
                "kind": channel["kind"],
                "bin_count": len(bkg),
                "background_yield": float(np.sum(bkg)),
                "data_yield": float(np.sum(data)),
                "data_mode": data_mode,
                "background_shape_nuisances": sorted(channel["variations"]),
                "background_processes": process_summary,
                "bin_labels": channel["bin_labels"],
            }
            for mass_key in selected_masses:
                values, sumw2 = signal_array(flat, name, mass_key)
                process = signal_process_name(mass_key)
                write_hist(directory, process, values, sumw2, edges)
                signal_summary = summary["signals"].setdefault(
                    mass_key,
                    {"process": process, "channels": {}, "shape_nuisances": {}},
                )
                signal_summary["channels"][name] = float(np.sum(values))
                by_sample = ((flat.get("search_bin_histograms") or {}).get(name) or {})
                signal_variations = aggregate_variations(
                    by_sample,
                    [SIGNAL_PREFIX + mass_key],
                    len(values),
                    values,
                )
                for nuisance, pair in signal_variations.items():
                    write_hist(directory, f"{process}_{nuisance}Up", np.asarray(pair["up"]), sumw2, edges)
                    write_hist(directory, f"{process}_{nuisance}Down", np.asarray(pair["down"]), sumw2, edges)
                    signal_summary["shape_nuisances"].setdefault(nuisance, []).append(name)
    finally:
        root_file.Close()
    return summary


def write_datacards(
    channels: list[dict[str, Any]],
    selected_masses: list[str],
    template_root: Path,
    root_summary: dict[str, Any],
    output_dir: Path,
    auto_mc_stats: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: dict[str, str] = {}
    lowdm_bins = len(channels[0]["background"])
    note = (
        "# Low-dM-only datacard: LLCR, QCDCR, GCR, DY2E, DY2M, and SR each use "
        f"the adopted {lowdm_bins} Nsv-inclusive categories; no high-dM channel is present.\n"
    )
    for mass_key in selected_masses:
        card = output_dir / f"datacard_{mass_key}.txt"
        text = datacard_text(
            template_root,
            channels,
            mass_key,
            root_summary,
            auto_mc_stats,
            lumi_name=LUMI_NAME,
            lumi_lnn=LUMI_LNN,
        )
        text = text.replace(
            "# Boosted AN17 datacard: CR channels use 6-bin recoil/U_T histograms; SR uses 17 boosted top/W tagged search bins.\n"
            "# SR background shape nuisances are reconstructed from shard-level search_bin_variations plus JES/MET unclustered shape shards.\n",
            "",
        )
        card.write_text(note + text)
        cards[mass_key] = str(card)
    return cards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hists", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-mode", choices=["asimov", "observed"], default="asimov")
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--max-mstop", type=int)
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--runner-jobs", type=int, default=12)
    parser.add_argument("--point-timeout", type=int, default=1800)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    flat = read_json(args.hists)
    channels = [
        lowdm_channel(flat, region, scheme)
        for region, scheme in LOWDM_REGION_MAP.items()
    ]
    lowdm_bins = len(channels[0]["background"])
    if any(len(channel["background"]) != lowdm_bins for channel in channels):
        raise ValueError("Low-dM regions do not have a common bin count")
    selected_masses = mass_points(flat, args.only, args.max_mstop)
    if args.max_points is not None:
        selected_masses = selected_masses[: args.max_points]
    if not selected_masses:
        raise SystemExit("no Low-dM signal mass points selected")

    outdir = args.output_dir
    tag = f"lowdm{lowdm_bins}"
    template_root = outdir / f"templates_{tag}.root"
    datacard_dir = outdir / "datacards"
    limit_dir = outdir / "limits"
    runner = outdir / "run_combine_expected.sh"
    if args.collect_only:
        root_summary: dict[str, Any] = {}
    else:
        root_summary = build_root(channels, flat, selected_masses, template_root, args.data_mode)
        cards = write_datacards(
            channels,
            selected_masses,
            template_root,
            root_summary,
            datacard_dir,
            args.auto_mc_stats,
        )
        write_parallel_runner(cards, limit_dir, runner, args.runner_jobs, args.point_timeout)

    limit_payload = collect_limits(limit_dir, selected_masses, outdir / "expected_limits.json")
    contour_png = outdir / f"expected_limit_contour_{tag}.png"
    contour_written = plot_contour(
        limit_payload,
        contour_png,
        run2_contours=None,
        luminosity_label=r"109.82 fb$^{-1}$ (13.6 TeV)",
        analysis_label=rf"Low-$\Delta m$ CR+SR only, $6\times{lowdm_bins}$ bins",
    )
    contour_pdf = contour_png.with_suffix(".pdf")
    manifest = {
        "status": "combine_outputs_complete" if limit_payload["status"] == "complete" else "combine_inputs_ready",
        "schema": f"lowdm_only_6region_{lowdm_bins}bin_v1",
        "hists": str(args.hists),
        "channels": [channel["name"] for channel in channels],
        "bins_per_channel": lowdm_bins,
        "total_analysis_bins": lowdm_bins * len(channels),
        "highdm_included": False,
        "data_mode": args.data_mode,
        "mass_points": selected_masses,
        "mass_point_count": len(selected_masses),
        "template_root": str(template_root),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "expected_limits": str(outdir / "expected_limits.json"),
        "contour_png": str(contour_png) if contour_written else None,
        "contour_pdf": str(contour_pdf) if contour_written and contour_pdf.exists() else None,
        "systematics": {
            "weight_shapes": sorted({name for channel in channels for name in channel["variations"]}),
            "lumi": {"name": LUMI_NAME, "lnN": LUMI_LNN},
            "autoMCStats": args.auto_mc_stats,
            "object_shape_variations": "not present in the current nominal plotting payload",
        },
        "root_summary": root_summary,
        "limit_collection": limit_payload,
    }
    write_json(outdir / "combine_input_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "channels": len(channels),
        "bins": manifest["total_analysis_bins"],
        "mass_points": len(selected_masses),
        "limits_collected": limit_payload["collected_point_count"],
        "contour": manifest["contour_png"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
