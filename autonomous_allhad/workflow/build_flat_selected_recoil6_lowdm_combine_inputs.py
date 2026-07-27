#!/usr/bin/env python3
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
    parse_mass_key,
    unit_edges,
    write_hist,
    write_json,
)
from build_flat_boosted_an17_combine_inputs import (  # noqa: E402
    CONTROL_REGION_MAP,
    SIGNAL_PREFIX,
    aggregate_nominal,
    aggregate_variations,
    cr_channel,
    hist_arrays,
    sample_to_mass_key,
)
from build_flat_recoil_ntop_split_combine_inputs import write_parallel_runner  # noqa: E402
from build_flat_recoil_ntop_split_lowdm_combine_inputs import (  # noqa: E402
    LOWDM_CHANNEL,
    LOWDM_SIGNAL_SCHEME,
    lowdm_channel,
    mass_points_from_lowdm,
    signal_array_from_search,
)
from build_flat_recoil_sr_combine_inputs import signal_process_name  # noqa: E402
from background_process_groups import (  # noqa: E402
    aggregate_background_processes,
    background_grouping_contract,
    materialize_grouped_background_templates,
)


DEFAULT_SELECTED_SCHEME = "boosted_an17_selected_recoil6_SR"
DEFAULT_SELECTED_CHANNEL = "cat7_SR_selected_an17_recoil6"
DEFAULT_OUTPUT_TAG = "selected_an17_recoil6"
MIN_BIN = 1.0e-9


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def selected_recoil6_channel(flat: dict[str, Any], scheme_name: str, channel_name: str) -> dict[str, Any]:
    scheme = ((flat.get("search_bin_schemes") or {}).get(scheme_name) or {})
    labels = list(scheme.get("bin_labels") or [])
    if not labels:
        raise ValueError(f"missing search bin labels for {scheme_name}")
    by_sample = ((flat.get("search_bin_histograms") or {}).get(scheme_name) or {})
    if not by_sample:
        raise ValueError(f"missing search-bin histograms for {scheme_name}")

    nbin = len(labels)
    bkg, bkg_s2, data, data_s2, backgrounds = aggregate_nominal(by_sample, nbin)
    return {
        "name": channel_name,
        "source_region": scheme_name,
        "source_kind": "search_bin_histograms",
        "kind": "signal_region_selected_an17_recoil6",
        "edges": unit_edges(nbin),
        "background": bkg,
        "background_sumw2": bkg_s2,
        "data": data,
        "data_sumw2": data_s2,
        "variations": aggregate_variations(by_sample, backgrounds, nbin, bkg),
        "variable": "selected_an17_recoil6_bin",
        "bin_labels": labels,
        "background_samples": backgrounds,
        "background_processes": aggregate_background_processes(
            by_sample, nbin, hist_arrays, signal_prefix=SIGNAL_PREFIX
        ),
    }


def mass_points_from_selected(flat: dict[str, Any], scheme_name: str, only: list[str] | None, max_mstop: int | None) -> list[str]:
    by_sample = ((flat.get("search_bin_histograms") or {}).get(scheme_name) or {})
    out: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for sample in by_sample:
        mass_key = sample_to_mass_key(sample)
        if not mass_key or mass_key in seen:
            continue
        if only and mass_key not in only:
            continue
        mstop, mlsp = parse_mass_key(mass_key)
        if mlsp >= mstop:
            continue
        if max_mstop is not None and mstop >= max_mstop:
            continue
        seen.add(mass_key)
        out.append((mstop, mlsp, mass_key))
    out.sort()
    return [key for _, _, key in out]


def signal_array_from_selected(flat: dict[str, Any], scheme_name: str, mass_key: str, nbin: int) -> tuple[np.ndarray, np.ndarray]:
    sample = SIGNAL_PREFIX + mass_key
    rec = (((flat.get("search_bin_histograms") or {}).get(scheme_name) or {}).get(sample) or {}).get("nominal")
    return hist_arrays(rec, nbin)


def signal_variations_from_selected(
    flat: dict[str, Any], scheme_name: str, mass_key: str, nbin: int
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    sample = SIGNAL_PREFIX + mass_key
    variations = ((flat.get("search_bin_histograms") or {}).get(scheme_name) or {}).get(sample) or {}
    bases = sorted(
        name[:-2]
        for name in variations
        if name.endswith("Up") and name[:-2] + "Down" in variations
    )
    return {
        base: {
            "up": hist_arrays(variations.get(base + "Up"), nbin),
            "down": hist_arrays(variations.get(base + "Down"), nbin),
        }
        for base in bases
    }


def build_root_from_flat(
    channels: list[dict[str, Any]],
    flat: dict[str, Any],
    mass_keys: list[str],
    output_root: Path,
    data_mode: str,
    selected_scheme: str,
) -> dict[str, Any]:
    import ROOT

    output_root.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output_root), "RECREATE")
    summary: dict[str, Any] = {
        "channels": {},
        "signals": {},
        "background_shape_nuisances": sorted({name for ch in channels for name in ch.get("variations", {})}),
        "background_grouping_contract": background_grouping_contract(),
    }
    try:
        for channel in channels:
            name = channel["name"]
            source_region = channel.get("source_region")
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
                "kind": channel.get("kind"),
                "source_region": source_region,
                "source_kind": channel.get("source_kind", "histograms"),
                "bin_count": int(len(bkg)),
                "background_yield": float(np.sum(bkg)),
                "data_yield": float(np.sum(data)),
                "data_mode": data_mode,
                "background_shape_nuisances": sorted((channel.get("variations") or {}).keys()),
                "background_processes": process_summary,
                "bin_labels": channel.get("bin_labels") or [],
            }
            for mass_key in mass_keys:
                proc = signal_process_name(mass_key)
                signal_shapes: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
                if source_region == selected_scheme:
                    sig, sig_s2 = signal_array_from_selected(flat, selected_scheme, mass_key, len(bkg))
                    signal_shapes = signal_variations_from_selected(flat, selected_scheme, mass_key, len(bkg))
                elif source_region == LOWDM_SIGNAL_SCHEME:
                    sig, sig_s2 = signal_array_from_search(flat, LOWDM_SIGNAL_SCHEME, mass_key, len(bkg))
                else:
                    sig = np.zeros(len(bkg), dtype=float)
                    sig_s2 = np.zeros(len(bkg), dtype=float)
                write_hist(directory, proc, np.maximum(sig, MIN_BIN), sig_s2, edges)
                signal_summary = summary["signals"].setdefault(
                    mass_key, {"process": proc, "channels": {}, "shape_nuisances": {}}
                )
                signal_summary["channels"][name] = float(np.sum(sig))
                for syst_name, pair in signal_shapes.items():
                    up, up_s2 = pair["up"]
                    down, down_s2 = pair["down"]
                    write_hist(directory, f"{proc}_{syst_name}Up", np.maximum(up, MIN_BIN), up_s2, edges)
                    write_hist(directory, f"{proc}_{syst_name}Down", np.maximum(down, MIN_BIN), down_s2, edges)
                    signal_summary["shape_nuisances"].setdefault(syst_name, []).append(name)
    finally:
        root_file.Close()
    return summary


def selected_lowdm_datacard_text(
    template_root: Path,
    channels: list[dict[str, Any]],
    mass_key: str,
    root_summary: dict[str, Any],
    auto_mc_stats: int,
    include_lowdm: bool,
    selected_bin_count: int,
    lumi_name: str = LUMI_NAME,
    lumi_lnn: float = LUMI_LNN,
) -> str:
    text = datacard_text(
        template_root,
        channels,
        mass_key,
        root_summary,
        auto_mc_stats,
        lumi_name=lumi_name,
        lumi_lnn=lumi_lnn,
    )
    lowdm_note = "low-dM contributes one inclusive SR bin" if include_lowdm else "low-dM is not included"
    category_count = selected_bin_count // 6
    return text.replace(
        "# Boosted AN17 datacard: CR channels use 6-bin recoil/U_T histograms; SR uses 17 boosted top/W tagged search bins.\n"
        "# SR background shape nuisances are reconstructed from shard-level search_bin_variations plus JES/MET unclustered shape shards.\n",
        "# Selected AN17 recoil datacard: high-dM CRs are inclusive 6-bin recoil templates; "
        f"high-dM SR uses {selected_bin_count} bins ({category_count} categories x 6 recoil bins); "
        f"{lowdm_note}. JES/MET shape nuisances are read from the payload when present; low-dM remains nominal.\n",
    )


def write_datacards(
    channels: list[dict[str, Any]],
    mass_keys: list[str],
    template_root: Path,
    root_summary: dict[str, Any],
    output_dir: Path,
    auto_mc_stats: int,
    include_lowdm: bool,
    selected_bin_count: int,
    lumi_name: str,
    lumi_lnn: float,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: dict[str, str] = {}
    for mass_key in mass_keys:
        card = output_dir / f"datacard_{mass_key}.txt"
        card.write_text(
            selected_lowdm_datacard_text(
                template_root,
                channels,
                mass_key,
                root_summary,
                auto_mc_stats,
                include_lowdm,
                selected_bin_count,
                lumi_name=lumi_name,
                lumi_lnn=lumi_lnn,
            )
        )
        cards[mass_key] = str(card)
    return cards


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Combine inputs with selected high-dM AN17 recoil6 SR categories.")
    parser.add_argument("--hists", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-scheme", default=DEFAULT_SELECTED_SCHEME)
    parser.add_argument("--selected-channel", default=DEFAULT_SELECTED_CHANNEL)
    parser.add_argument("--output-tag", default=DEFAULT_OUTPUT_TAG)
    parser.add_argument("--no-lowdm", action="store_true", help="Do not add the one-bin low-dM SR channel.")
    parser.add_argument("--data-mode", choices=["asimov", "observed"], default="asimov")
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--max-mstop", type=int, default=None)
    parser.add_argument("--runner-jobs", type=int, default=8)
    parser.add_argument("--point-timeout", type=int, default=1800)
    parser.add_argument("--lumi-name", default=LUMI_NAME)
    parser.add_argument("--lumi-lnn", type=float, default=LUMI_LNN)
    args = parser.parse_args()

    include_lowdm = not args.no_lowdm
    hists_path = Path(args.hists)
    flat = read_json(hists_path)
    channels = [cr_channel(flat, flat_region, channel_name) for flat_region, channel_name in CONTROL_REGION_MAP.items()]
    selected_channel = selected_recoil6_channel(flat, args.selected_scheme, args.selected_channel)
    selected_bin_count = len(selected_channel["bin_labels"])
    channels.append(selected_channel)
    if include_lowdm:
        channels.append(lowdm_channel(flat))

    high_keys = mass_points_from_selected(flat, args.selected_scheme, args.only, args.max_mstop)
    low_keys = mass_points_from_lowdm(flat, args.only, args.max_mstop) if include_lowdm else []
    mass_keys = sorted(set(high_keys) | set(low_keys), key=lambda key: parse_mass_key(key))
    if args.max_points is not None:
        mass_keys = mass_keys[: args.max_points]
    if not mass_keys:
        raise SystemExit("no signal mass points selected from selected high-dM SR or low-dM SR")

    outdir = Path(args.output_dir)
    template_root = outdir / f"templates_{args.output_tag}{'_lowdm' if include_lowdm else ''}.root"
    datacard_dir = outdir / "datacards"
    limit_dir = outdir / "limits"
    runner = outdir / "run_combine_expected.sh"
    root_summary = build_root_from_flat(channels, flat, mass_keys, template_root, args.data_mode, args.selected_scheme)
    cards = write_datacards(
        channels,
        mass_keys,
        template_root,
        root_summary,
        datacard_dir,
        args.auto_mc_stats,
        include_lowdm,
        selected_bin_count,
        args.lumi_name,
        args.lumi_lnn,
    )
    write_parallel_runner(cards, limit_dir, runner, args.runner_jobs, args.point_timeout)

    manifest = {
        "status": "combine_inputs_ready",
        "schema": "flat_selected_an17_recoil_plus_lowdm_sr_v2" if include_lowdm else "flat_selected_an17_recoil_sr_v2",
        "hists": str(hists_path),
        "template_root": str(template_root),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "data_mode": args.data_mode,
        "mass_point_count": len(mass_keys),
        "mass_points": mass_keys,
        "control_regions": CONTROL_REGION_MAP,
        "selected_recoil_scheme": args.selected_scheme,
        "selected_recoil_channel": args.selected_channel,
        "selected_recoil_bin_count": selected_bin_count,
        "selected_recoil_policy": f"{selected_bin_count // 6} category blocks are expanded into six recoil bins each, preserving payload order.",
        "lowdm_signal_scheme": LOWDM_SIGNAL_SCHEME if include_lowdm else None,
        "lowdm_channel": LOWDM_CHANNEL if include_lowdm else None,
        "lowdm_policy": "one inclusive low-dM SR bin added" if include_lowdm else "not included",
        "systematics_policy": {
            "source": "shape variations already present in the input flat histogram payload",
            "deferred": [
                name
                for name in ("jesTotal", "metUnclustered")
                if name not in set(root_summary.get("background_shape_nuisances") or [])
            ],
            "lumi": {"name": args.lumi_name, "lnN": args.lumi_lnn},
            "autoMCStats": args.auto_mc_stats,
        },
        "channels": [ch["name"] for ch in channels],
        "root_summary": root_summary,
    }
    write_json(outdir / "combine_input_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "channels": len(channels),
        "highdm_selected_bins": selected_bin_count,
        "include_lowdm": include_lowdm,
        "mass_points": len(mass_keys),
        "datacards": len(cards),
        "template_root": str(template_root),
        "runner": str(runner),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
