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
    BACKGROUND_NAME,
    LUMI_LNN,
    LUMI_NAME,
    datacard_text,
    stable_path,
    write_hist,
    write_json,
)
from build_flat_recoil_sr_combine_inputs import (  # noqa: E402
    SIGNAL_PREFIX,
    aggregate_nominal,
    aggregate_variations,
    hist_arrays,
    parse_mass_key,
    recoil_channel,
    sample_to_mass_key,
    signal_process_name,
)
from build_flat_recoil_ntop_split_combine_inputs import (  # noqa: E402
    CONTROL_SPLIT_MAP,
    SIGNAL_SPLIT_MAP,
    SIGNAL_REGIONS,
    mass_points_from_regions,
    signal_array_from_region,
    write_parallel_runner,
)

LOWDM_SIGNAL_SCHEME = "cat7_SR_lowDeltaM"
LOWDM_CHANNEL = "cat8_SR_lowDeltaM_onebin"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def lowdm_channel(flat: dict[str, Any]) -> dict[str, Any]:
    by_sample = ((flat.get("search_bin_histograms") or {}).get(LOWDM_SIGNAL_SCHEME) or {})
    if not by_sample:
        raise ValueError(f"missing low-dM signal histogram scheme {LOWDM_SIGNAL_SCHEME}")
    labels = (((flat.get("search_bin_schemes") or {}).get(LOWDM_SIGNAL_SCHEME) or {}).get("bin_labels") or ["lowdm_inclusive"])
    nbin = max(1, len(labels))
    edges = np.arange(nbin + 1, dtype=float)
    bkg, bkg_s2, data, data_s2, backgrounds = aggregate_nominal(by_sample, nbin)
    return {
        "name": LOWDM_CHANNEL,
        "source_region": LOWDM_SIGNAL_SCHEME,
        "source_kind": "search_bin_histograms",
        "kind": "signal_lowdm_onebin_flat",
        "edges": edges,
        "background": bkg,
        "background_sumw2": bkg_s2,
        "data": data,
        "data_sumw2": data_s2,
        "variations": aggregate_variations(by_sample, backgrounds, nbin, bkg),
        "variable": "lowdm_inclusive",
        "bin_labels": labels,
        "background_samples": backgrounds,
    }


def signal_array_from_search(flat: dict[str, Any], scheme: str, mass_key: str, nbin: int) -> tuple[np.ndarray, np.ndarray]:
    sample = SIGNAL_PREFIX + mass_key
    rec = (((flat.get("search_bin_histograms") or {}).get(scheme) or {}).get(sample) or {}).get("nominal")
    return hist_arrays(rec, nbin)


def mass_points_from_lowdm(flat: dict[str, Any], only: list[str] | None, max_mstop: int | None) -> list[str]:
    by_sample = ((flat.get("search_bin_histograms") or {}).get(LOWDM_SIGNAL_SCHEME) or {})
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


def build_root_from_flat(channels: list[dict[str, Any]], flat: dict[str, Any], mass_keys: list[str], output_root: Path, data_mode: str) -> dict[str, Any]:
    import ROOT

    output_root.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output_root), "RECREATE")
    summary: dict[str, Any] = {
        "channels": {},
        "signals": {},
        "background_shape_nuisances": sorted({name for ch in channels for name in ch.get("variations", {})}),
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
            write_hist(directory, BACKGROUND_NAME, bkg, bkg_s2, edges)
            for syst_name, pair in (channel.get("variations") or {}).items():
                up = np.asarray(pair.get("up", bkg), dtype=float)
                down = np.asarray(pair.get("down", bkg), dtype=float)
                if len(up) == len(bkg):
                    write_hist(directory, f"{BACKGROUND_NAME}_{syst_name}Up", up, bkg_s2, edges)
                if len(down) == len(bkg):
                    write_hist(directory, f"{BACKGROUND_NAME}_{syst_name}Down", down, bkg_s2, edges)
            summary["channels"][name] = {
                "kind": channel.get("kind"),
                "source_region": source_region,
                "source_kind": channel.get("source_kind", "histograms"),
                "bin_count": int(len(bkg)),
                "background_yield": float(np.sum(bkg)),
                "data_yield": float(np.sum(data)),
                "data_mode": data_mode,
                "background_shape_nuisances": sorted((channel.get("variations") or {}).keys()),
                "bin_labels": channel.get("bin_labels") or [],
            }
            for mass_key in mass_keys:
                proc = signal_process_name(mass_key)
                if source_region in SIGNAL_REGIONS:
                    sig, sig_s2 = signal_array_from_region(flat, str(source_region), mass_key, len(bkg))
                elif source_region == LOWDM_SIGNAL_SCHEME:
                    sig, sig_s2 = signal_array_from_search(flat, LOWDM_SIGNAL_SCHEME, mass_key, len(bkg))
                else:
                    sig = np.zeros(len(bkg), dtype=float)
                    sig_s2 = np.zeros(len(bkg), dtype=float)
                write_hist(directory, proc, sig, sig_s2, edges)
                summary["signals"].setdefault(mass_key, {"process": proc, "channels": {}})["channels"][name] = float(np.sum(sig))
    finally:
        root_file.Close()
    return summary


def split_lowdm_datacard_text(template_root: Path, channels: list[dict[str, Any]], mass_key: str, root_summary: dict[str, Any], auto_mc_stats: int) -> str:
    text = datacard_text(template_root, channels, mass_key, root_summary, auto_mc_stats)
    return text.replace(
        "# Boosted AN17 datacard: CR channels use 6-bin recoil/U_T histograms; SR uses 17 boosted top/W tagged search bins.\n"
        "# SR background shape nuisances are reconstructed from shard-level search_bin_variations plus JES/MET unclustered shape shards.\n",
        "# Recoil nTop-split + low-dM datacard: high-dM CR/SR channels are split into nTop>=1 and nTop==0 recoil templates; low-dM contributes one inclusive SR bin.\n"
        "# Low-dM CRs are intentionally not included here because current flat ROOT does not retain exact low-dM CR object-state definitions. JES/MET shape nuisances are deferred.\n",
    )


def write_datacards(channels: list[dict[str, Any]], mass_keys: list[str], template_root: Path, root_summary: dict[str, Any], output_dir: Path, auto_mc_stats: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: dict[str, str] = {}
    for mass_key in mass_keys:
        card = output_dir / f"datacard_{mass_key}.txt"
        card.write_text(split_lowdm_datacard_text(template_root, channels, mass_key, root_summary, auto_mc_stats))
        cards[mass_key] = str(card)
    return cards


def validate_regions(flat: dict[str, Any], regions: list[str]) -> None:
    missing = [region for region in regions if region not in (flat.get("histograms") or {})]
    if missing:
        raise SystemExit("missing split histogram regions: " + ", ".join(missing))
    if LOWDM_SIGNAL_SCHEME not in (flat.get("search_bin_histograms") or {}):
        raise SystemExit(f"missing low-dM SR histogram scheme: {LOWDM_SIGNAL_SCHEME}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Combine inputs with high-dM nTop split CR/SR plus one-bin low-dM SR.")
    parser.add_argument("--hists", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-mode", choices=["asimov", "observed"], default="asimov")
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--max-mstop", type=int, default=None)
    parser.add_argument("--runner-jobs", type=int, default=8)
    parser.add_argument("--point-timeout", type=int, default=1800)
    args = parser.parse_args()

    hists_path = Path(args.hists)
    flat = read_json(hists_path)
    control_regions = [region for region, _ in CONTROL_SPLIT_MAP]
    signal_regions = [region for region, _ in SIGNAL_SPLIT_MAP]
    validate_regions(flat, control_regions + signal_regions)

    channels = []
    for flat_region, channel_name in CONTROL_SPLIT_MAP:
        channels.append(recoil_channel(flat, flat_region, channel_name, "control_recoil_6bin_flat_nTop_split"))
    for flat_region, channel_name in SIGNAL_SPLIT_MAP:
        kind = "signal_recoil_6bin_flat_nTop_ge_1" if flat_region.endswith("Nt1") else "signal_recoil_6bin_flat_nTop_eq_0"
        channels.append(recoil_channel(flat, flat_region, channel_name, kind))
    channels.append(lowdm_channel(flat))

    high_keys = mass_points_from_regions(flat, signal_regions, args.only, None, args.max_mstop)
    low_keys = mass_points_from_lowdm(flat, args.only, args.max_mstop)
    mass_keys = sorted(set(high_keys) | set(low_keys), key=lambda key: parse_mass_key(key))
    if args.max_points is not None:
        mass_keys = mass_keys[: args.max_points]
    if not mass_keys:
        raise SystemExit("no signal mass points selected from high-dM split SR or low-dM SR")

    outdir = Path(args.output_dir)
    template_root = outdir / "templates_recoil_ntop_split_lowdm.root"
    datacard_dir = outdir / "datacards"
    limit_dir = outdir / "limits"
    runner = outdir / "run_combine_expected.sh"
    root_summary = build_root_from_flat(channels, flat, mass_keys, template_root, args.data_mode)
    cards = write_datacards(channels, mass_keys, template_root, root_summary, datacard_dir, args.auto_mc_stats)
    write_parallel_runner(cards, limit_dir, runner, args.runner_jobs, args.point_timeout)

    manifest = {
        "status": "combine_inputs_ready",
        "schema": "flat_recoil_ntop_split_plus_lowdm_sr_v1",
        "hists": str(hists_path),
        "template_root": str(template_root),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "data_mode": args.data_mode,
        "mass_point_count": len(mass_keys),
        "mass_points": mass_keys,
        "control_split_regions": dict(CONTROL_SPLIT_MAP),
        "signal_split_regions": dict(SIGNAL_SPLIT_MAP),
        "lowdm_signal_scheme": LOWDM_SIGNAL_SCHEME,
        "lowdm_policy": "one inclusive low-dM SR bin added; low-dM CRs excluded from combine until exact CR object-state branches are retained",
        "systematics_policy": {
            "source": "shape variations already present in the input flat histogram payload",
            "deferred": ["jesTotal", "metUnclustered"],
            "lumi": {"name": LUMI_NAME, "lnN": LUMI_LNN},
            "autoMCStats": args.auto_mc_stats,
        },
        "channels": [ch["name"] for ch in channels],
        "root_summary": root_summary,
    }
    write_json(outdir / "combine_input_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "channels": len(channels),
        "mass_points": len(mass_keys),
        "datacards": len(cards),
        "template_root": str(template_root),
        "runner": str(runner),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
