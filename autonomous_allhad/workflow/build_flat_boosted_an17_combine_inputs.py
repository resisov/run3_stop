#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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
    SR_CHANNEL,
    build_root,
    datacard_text,
    pair_variations,
    parse_mass_key,
    stable_path,
    unit_edges,
    write_datacards,
    write_json,
    write_limit_runner,
)

CONTROL_REGION_MAP = {
    "LLCR": "cat2_LLCR_highDeltaM",
    "QCDCR": "cat3_QCDCR_highDeltaM",
    "GCR": "cat4_GCR_highDeltaM",
    "DY2E": "cat5_DY2E_highDeltaM",
    "DY2M": "cat6_DY2M_highDeltaM",
}
SEARCH_SCHEME = "boosted_an_17_SR_Nt1"
SIGNAL_PREFIX = "T2tt_"
MIN_BIN = 1.0e-9


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sample_is_signal(name: str) -> bool:
    return name.startswith(SIGNAL_PREFIX)


def sample_to_mass_key(name: str) -> str | None:
    match = re.match(r"T2tt_(mStop\d+_mLSP\d+)$", name)
    return match.group(1) if match else None


def hist_arrays(rec: dict[str, Any] | None, nbin: int) -> tuple[np.ndarray, np.ndarray]:
    if not rec:
        return np.zeros(nbin, dtype=float), np.zeros(nbin, dtype=float)
    vals = np.asarray(rec.get("sumw") or [], dtype=float)
    s2 = np.asarray(rec.get("sumw2") or [], dtype=float)
    out = np.zeros(nbin, dtype=float)
    out2 = np.zeros(nbin, dtype=float)
    n = min(nbin, len(vals))
    if n:
        out[:n] = vals[:n]
    n2 = min(nbin, len(s2))
    if n2:
        out2[:n2] = s2[:n2]
    return out, out2


def aggregate_nominal(by_sample: dict[str, Any], nbin: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    bkg = np.zeros(nbin, dtype=float)
    bkg_s2 = np.zeros(nbin, dtype=float)
    data = np.zeros(nbin, dtype=float)
    data_s2 = np.zeros(nbin, dtype=float)
    backgrounds: list[str] = []
    for sample, variations in sorted(by_sample.items()):
        vals, s2 = hist_arrays((variations or {}).get("nominal"), nbin)
        if sample == "data_obs":
            data += vals
            data_s2 += s2
        elif sample_is_signal(sample):
            continue
        else:
            backgrounds.append(sample)
            bkg += vals
            bkg_s2 += s2
    return bkg, bkg_s2, data, data_s2, backgrounds


def aggregate_variations(by_sample: dict[str, Any], backgrounds: list[str], nbin: int, nominal: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    var_names = sorted({name for sample in backgrounds for name in ((by_sample.get(sample) or {}).keys()) if name != "nominal"})
    varied: dict[str, np.ndarray] = {}
    for var_name in var_names:
        arr = np.zeros(nbin, dtype=float)
        for sample in backgrounds:
            variations = by_sample.get(sample) or {}
            vals, _ = hist_arrays(variations.get(var_name) or variations.get("nominal"), nbin)
            arr += vals
        varied[var_name] = arr
    return pair_variations(nominal, varied)


def cr_channel(flat: dict[str, Any], flat_region: str, channel_name: str) -> dict[str, Any]:
    by_sample = ((flat.get("histograms") or {}).get(flat_region) or {})
    edges = np.asarray(flat.get("recoil_pt_bins") or [], dtype=float)
    nbin = max(len(edges) - 1, 0)
    if nbin <= 0:
        raise ValueError("missing recoil_pt_bins")
    bkg, bkg_s2, data, data_s2, backgrounds = aggregate_nominal(by_sample, nbin)
    return {
        "name": channel_name,
        "source_region": flat_region,
        "kind": "control_recoil_6bin_flat",
        "edges": edges,
        "background": bkg,
        "background_sumw2": bkg_s2,
        "data": data,
        "data_sumw2": data_s2,
        "variations": aggregate_variations(by_sample, backgrounds, nbin, bkg),
        "variable": "recoil_or_met",
        "background_samples": backgrounds,
    }


def sr_channel(flat: dict[str, Any]) -> dict[str, Any]:
    scheme = ((flat.get("search_bin_schemes") or {}).get(SEARCH_SCHEME) or {})
    labels = list(scheme.get("bin_labels") or [])
    if not labels:
        raise ValueError(f"missing search bin labels for {SEARCH_SCHEME}")
    by_sample = ((flat.get("search_bin_histograms") or {}).get(SEARCH_SCHEME) or {})
    nbin = len(labels)
    bkg, bkg_s2, data, data_s2, backgrounds = aggregate_nominal(by_sample, nbin)
    return {
        "name": SR_CHANNEL,
        "source_region": SEARCH_SCHEME,
        "kind": "signal_region_boosted_an17_17bin_flat_SR_Nt1",
        "edges": unit_edges(nbin),
        "background": bkg,
        "background_sumw2": bkg_s2,
        "data": data,
        "data_sumw2": data_s2,
        "variations": aggregate_variations(by_sample, backgrounds, nbin, bkg),
        "bin_labels": labels,
        "background_samples": backgrounds,
    }


def mass_points_from_flat(flat: dict[str, Any], only: list[str] | None, max_points: int | None, max_mstop: int) -> list[str]:
    by_sample = ((flat.get("search_bin_histograms") or {}).get(SEARCH_SCHEME) or {})
    out: list[tuple[int, int, str]] = []
    for sample in by_sample:
        mass_key = sample_to_mass_key(sample)
        if not mass_key:
            continue
        if only and mass_key not in only:
            continue
        try:
            mstop, mlsp = parse_mass_key(mass_key)
        except ValueError:
            continue
        if mlsp >= mstop or mstop >= max_mstop:
            continue
        out.append((mstop, mlsp, mass_key))
    out.sort()
    mass_keys = [key for _, _, key in out]
    if max_points is not None:
        mass_keys = mass_keys[:max_points]
    return mass_keys


def signal_array_from_flat(flat: dict[str, Any], mass_key: str, nbin: int) -> tuple[np.ndarray, np.ndarray]:
    sample = SIGNAL_PREFIX + mass_key
    rec = (((flat.get("search_bin_histograms") or {}).get(SEARCH_SCHEME) or {}).get(sample) or {}).get("nominal")
    return hist_arrays(rec, nbin)


def signal_variations_from_flat(flat: dict[str, Any], mass_key: str, nbin: int, nominal: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    sample = SIGNAL_PREFIX + mass_key
    variations = (((flat.get("search_bin_histograms") or {}).get(SEARCH_SCHEME) or {}).get(sample) or {})
    varied: dict[str, np.ndarray] = {}
    for var_name, rec in variations.items():
        if var_name == "nominal":
            continue
        vals, _ = hist_arrays(rec, nbin)
        varied[var_name] = vals
    return pair_variations(nominal, varied)


def build_root_from_flat(channels: list[dict[str, Any]], flat: dict[str, Any], mass_keys: list[str], output_root: Path, data_mode: str) -> dict[str, Any]:
    import ROOT

    output_root.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output_root), "RECREATE")
    summary: dict[str, Any] = {"channels": {}, "signals": {}, "background_shape_nuisances": sorted({name for ch in channels for name in ch.get("variations", {})})}
    try:
        for channel in channels:
            name = channel["name"]
            directory = root_file.mkdir(name)
            edges = np.asarray(channel["edges"], dtype=float)
            bkg = np.asarray(channel["background"], dtype=float)
            bkg_s2 = np.asarray(channel["background_sumw2"], dtype=float)
            data = bkg if data_mode == "asimov" else np.asarray(channel["data"], dtype=float)
            from build_boosted_an17_combine_inputs import write_hist

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
                "bin_count": int(len(bkg)),
                "background_yield": float(np.sum(bkg)),
                "data_yield": float(np.sum(data)),
                "data_mode": data_mode,
                "background_shape_nuisances": sorted((channel.get("variations") or {}).keys()),
            }
            if channel.get("bin_labels"):
                summary["channels"][name]["bin_labels"] = channel.get("bin_labels")
            for mass_key in mass_keys:
                proc = "sig_" + re.sub(r"[^A-Za-z0-9_]+", "_", mass_key).strip("_")
                if name == SR_CHANNEL:
                    sig, sig_s2 = signal_array_from_flat(flat, mass_key, len(bkg))
                else:
                    sig = np.zeros(len(bkg), dtype=float)
                    sig_s2 = np.zeros(len(bkg), dtype=float)
                write_hist(directory, proc, sig, sig_s2, edges)
                summary["signals"].setdefault(mass_key, {"process": proc, "channels": {}})["channels"][name] = float(np.sum(sig))
    finally:
        root_file.Close()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Combine ROOT templates/datacards from flat boosted recoil + AN17 histograms.")
    parser.add_argument("--hists", required=True)
    parser.add_argument("--output-dir", default="analysis/combine/flat_boosted_an17_20260630")
    parser.add_argument("--data-mode", choices=["asimov", "observed"], default="asimov")
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--max-mstop", type=int, default=1800)
    args = parser.parse_args()

    hists_path = Path(args.hists)
    flat = read_json(hists_path)
    channels = [cr_channel(flat, flat_region, channel_name) for flat_region, channel_name in CONTROL_REGION_MAP.items()]
    channels.append(sr_channel(flat))
    mass_keys = mass_points_from_flat(flat, args.only, args.max_points, args.max_mstop)
    if not mass_keys:
        raise SystemExit("no signal mass points selected from flat histograms")

    outdir = Path(args.output_dir)
    template_root = outdir / "templates_flat_boosted_an17.root"
    datacard_dir = outdir / "datacards"
    limit_dir = outdir / "limits"
    runner = outdir / "run_combine_expected.sh"
    root_summary = build_root_from_flat(channels, flat, mass_keys, template_root, args.data_mode)
    cards = write_datacards(channels, mass_keys, template_root, root_summary, datacard_dir, args.auto_mc_stats)
    write_limit_runner(cards, limit_dir, runner)
    manifest = {
        "status": "combine_inputs_ready",
        "schema": "flat_boosted_an17_searchbin_v1",
        "hists": str(hists_path),
        "template_root": str(template_root),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "data_mode": args.data_mode,
        "mass_point_count": len(mass_keys),
        "mass_points": mass_keys,
        "max_mstop_policy": f"mStop < {args.max_mstop}",
        "search_bin_scheme": SEARCH_SCHEME,
        "search_bin_count": len(channels[-1].get("bin_labels") or []),
        "search_bin_order": channels[-1].get("bin_labels") or [],
        "channels": [ch["name"] for ch in channels],
        "systematics_policy": {
            "source": "post-skim flat ntuple scale-factor variations from compute_weight_bundle; JEC/FJEC already applied before skim",
            "lumi": {"name": LUMI_NAME, "lnN": LUMI_LNN},
            "autoMCStats": args.auto_mc_stats,
        },
        "root_summary": root_summary,
    }
    write_json(outdir / "combine_input_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "channels": len(channels),
        "search_bins": manifest["search_bin_count"],
        "mass_points": len(mass_keys),
        "datacards": len(cards),
        "template_root": str(template_root),
        "runner": str(runner),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
