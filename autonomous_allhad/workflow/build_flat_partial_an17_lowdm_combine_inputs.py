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
    datacard_text,
    parse_mass_key,
    stable_path,
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


INCLUSIVE_SCHEME = "boosted_an_17_SR"
NT1_SCHEME = "boosted_an_17_SR_Nt1"
PARTIAL_CHANNEL = "cat7_SR_boosted_an17_partial_nTop_split"
DEFAULT_SPLIT_BINS = [4, 5, 8, 9, 14, 15, 16]
MIN_BIN = 1.0e-9


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sample_is_signal(name: str) -> bool:
    return name.startswith(SIGNAL_PREFIX)


def subtract_hist(a: dict[str, Any] | None, b: dict[str, Any] | None, nbin: int) -> tuple[np.ndarray, np.ndarray]:
    avals, as2 = hist_arrays(a, nbin)
    bvals, bs2 = hist_arrays(b, nbin)
    vals = avals - bvals
    s2 = as2 - bs2
    vals[np.abs(vals) < 1.0e-10] = 0.0
    s2[np.abs(s2) < 1.0e-10] = 0.0
    return np.maximum(vals, 0.0), np.maximum(s2, 0.0)


def expanded_labels(labels: list[str], split_bins: list[int]) -> list[str]:
    split = set(split_bins)
    out: list[str] = []
    for idx, label in enumerate(labels, start=1):
        if idx in split:
            out.append(f"{idx}: {label}, N_t=0")
            out.append(f"{idx}: {label}, N_t>=1")
        else:
            out.append(f"{idx}: {label}")
    return out


def expand_sample_variation(
    inc_rec: dict[str, Any] | None,
    nt1_rec: dict[str, Any] | None,
    nbin_in: int,
    split_bins: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    inc_vals, inc_s2 = hist_arrays(inc_rec, nbin_in)
    nt1_vals, nt1_s2 = hist_arrays(nt1_rec, nbin_in)
    nt0_vals, nt0_s2 = subtract_hist(inc_rec, nt1_rec, nbin_in)
    split = set(split_bins)
    vals: list[float] = []
    s2: list[float] = []
    for idx in range(nbin_in):
        if idx + 1 in split:
            vals.extend([float(nt0_vals[idx]), float(nt1_vals[idx])])
            s2.extend([float(nt0_s2[idx]), float(nt1_s2[idx])])
        else:
            vals.append(float(inc_vals[idx]))
            s2.append(float(inc_s2[idx]))
    return np.asarray(vals, dtype=float), np.asarray(s2, dtype=float)


def partial_sr_channel(flat: dict[str, Any], split_bins: list[int]) -> dict[str, Any]:
    labels = (((flat.get("search_bin_schemes") or {}).get(INCLUSIVE_SCHEME) or {}).get("bin_labels") or [])
    if not labels:
        raise ValueError(f"missing search bin labels for {INCLUSIVE_SCHEME}")
    nbin_in = len(labels)
    by_inc = ((flat.get("search_bin_histograms") or {}).get(INCLUSIVE_SCHEME) or {})
    by_nt1 = ((flat.get("search_bin_histograms") or {}).get(NT1_SCHEME) or {})
    if not by_inc:
        raise ValueError(f"missing search histograms for {INCLUSIVE_SCHEME}")
    if not by_nt1:
        raise ValueError(f"missing search histograms for {NT1_SCHEME}")

    out_labels = expanded_labels(labels, split_bins)
    nbin_out = len(out_labels)
    bkg = np.zeros(nbin_out, dtype=float)
    bkg_s2 = np.zeros(nbin_out, dtype=float)
    data = np.zeros(nbin_out, dtype=float)
    data_s2 = np.zeros(nbin_out, dtype=float)
    backgrounds: list[str] = []

    for sample in sorted(by_inc):
        vals, s2 = expand_sample_variation(
            (by_inc.get(sample) or {}).get("nominal"),
            (by_nt1.get(sample) or {}).get("nominal"),
            nbin_in,
            split_bins,
        )
        if sample == "data_obs":
            data += vals
            data_s2 += s2
        elif sample_is_signal(sample):
            continue
        else:
            backgrounds.append(sample)
            bkg += vals
            bkg_s2 += s2

    var_names = sorted({name for sample in backgrounds for name in ((by_inc.get(sample) or {}).keys()) if name != "nominal"})
    varied: dict[str, np.ndarray] = {}
    for var_name in var_names:
        arr = np.zeros(nbin_out, dtype=float)
        for sample in backgrounds:
            inc_vars = by_inc.get(sample) or {}
            nt1_vars = by_nt1.get(sample) or {}
            vals, _ = expand_sample_variation(
                inc_vars.get(var_name) or inc_vars.get("nominal"),
                nt1_vars.get(var_name) or nt1_vars.get("nominal"),
                nbin_in,
                split_bins,
            )
            arr += vals
        varied[var_name] = arr

    return {
        "name": PARTIAL_CHANNEL,
        "source_region": "partial_an17_nTop_split",
        "source_kind": "search_bin_histograms",
        "kind": "signal_region_an17_partial_nTop_split",
        "edges": unit_edges(nbin_out),
        "background": bkg,
        "background_sumw2": bkg_s2,
        "data": data,
        "data_sumw2": data_s2,
        "variations": aggregate_variations({sample: by_inc[sample] for sample in backgrounds}, backgrounds, nbin_in, np.zeros(nbin_in)) if False else {},
        "variation_arrays": varied,
        "bin_labels": out_labels,
        "source_bin_labels": labels,
        "split_bins_1based": split_bins,
        "background_samples": backgrounds,
    }


def mass_points_from_partial(flat: dict[str, Any], only: list[str] | None, max_mstop: int | None) -> list[str]:
    by_sample = ((flat.get("search_bin_histograms") or {}).get(INCLUSIVE_SCHEME) or {})
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


def signal_array_from_partial(flat: dict[str, Any], mass_key: str, nbin: int, split_bins: list[int]) -> tuple[np.ndarray, np.ndarray]:
    sample = SIGNAL_PREFIX + mass_key
    labels = (((flat.get("search_bin_schemes") or {}).get(INCLUSIVE_SCHEME) or {}).get("bin_labels") or [])
    vals, s2 = expand_sample_variation(
        ((((flat.get("search_bin_histograms") or {}).get(INCLUSIVE_SCHEME) or {}).get(sample) or {}).get("nominal")),
        ((((flat.get("search_bin_histograms") or {}).get(NT1_SCHEME) or {}).get(sample) or {}).get("nominal")),
        len(labels),
        split_bins,
    )
    if len(vals) != nbin:
        out = np.zeros(nbin, dtype=float)
        out2 = np.zeros(nbin, dtype=float)
        n = min(nbin, len(vals))
        out[:n] = vals[:n]
        out2[:n] = s2[:n]
        return out, out2
    return vals, s2


def paired_variations_from_arrays(nominal: np.ndarray, varied: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    pairs: dict[str, dict[str, np.ndarray]] = {}
    for name, arr in varied.items():
        if name.endswith("Up"):
            base = name[:-2]
            pairs.setdefault(base, {})["up"] = arr
        elif name.endswith("Down"):
            base = name[:-4]
            pairs.setdefault(base, {})["down"] = arr
    out: dict[str, dict[str, np.ndarray]] = {}
    for base, pair in pairs.items():
        up = pair.get("up", nominal)
        down = pair.get("down", nominal)
        if len(up) == len(nominal) and len(down) == len(nominal):
            out[base] = {"up": up, "down": down}
    return out


def build_root_from_flat(channels: list[dict[str, Any]], flat: dict[str, Any], mass_keys: list[str], output_root: Path, data_mode: str, split_bins: list[int]) -> dict[str, Any]:
    import ROOT

    output_root.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output_root), "RECREATE")
    summary: dict[str, Any] = {"channels": {}, "signals": {}, "background_shape_nuisances": []}
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
            write_hist(directory, BACKGROUND_NAME, np.maximum(bkg, MIN_BIN), bkg_s2, edges)

            variations = channel.get("variations") or {}
            if channel.get("variation_arrays"):
                variations = paired_variations_from_arrays(bkg, channel.get("variation_arrays") or {})
            for syst_name, pair in variations.items():
                up = np.asarray(pair.get("up", bkg), dtype=float)
                down = np.asarray(pair.get("down", bkg), dtype=float)
                if len(up) == len(bkg):
                    write_hist(directory, f"{BACKGROUND_NAME}_{syst_name}Up", np.maximum(up, MIN_BIN), bkg_s2, edges)
                if len(down) == len(bkg):
                    write_hist(directory, f"{BACKGROUND_NAME}_{syst_name}Down", np.maximum(down, MIN_BIN), bkg_s2, edges)

            summary["background_shape_nuisances"] = sorted(set(summary["background_shape_nuisances"]) | set(variations))
            summary["channels"][name] = {
                "kind": channel.get("kind"),
                "source_region": source_region,
                "source_kind": channel.get("source_kind", "histograms"),
                "bin_count": int(len(bkg)),
                "background_yield": float(np.sum(bkg)),
                "data_yield": float(np.sum(data)),
                "data_mode": data_mode,
                "background_shape_nuisances": sorted(variations),
                "bin_labels": channel.get("bin_labels") or [],
            }
            for mass_key in mass_keys:
                proc = signal_process_name(mass_key)
                if name == PARTIAL_CHANNEL:
                    sig, sig_s2 = signal_array_from_partial(flat, mass_key, len(bkg), split_bins)
                elif source_region == LOWDM_SIGNAL_SCHEME:
                    sig, sig_s2 = signal_array_from_search(flat, LOWDM_SIGNAL_SCHEME, mass_key, len(bkg))
                else:
                    sig = np.zeros(len(bkg), dtype=float)
                    sig_s2 = np.zeros(len(bkg), dtype=float)
                write_hist(directory, proc, np.maximum(sig, MIN_BIN), sig_s2, edges)
                summary["signals"].setdefault(mass_key, {"process": proc, "channels": {}})["channels"][name] = float(np.sum(sig))
    finally:
        root_file.Close()
    return summary


def partial_lowdm_datacard_text(template_root: Path, channels: list[dict[str, Any]], mass_key: str, root_summary: dict[str, Any], auto_mc_stats: int) -> str:
    text = datacard_text(template_root, channels, mass_key, root_summary, auto_mc_stats)
    return text.replace(
        "# Boosted AN17 datacard: CR channels use 6-bin recoil/U_T histograms; SR uses 17 boosted top/W tagged search bins.\n"
        "# SR background shape nuisances are reconstructed from shard-level search_bin_variations plus JES/MET unclustered shape shards.\n",
        "# Partial AN17 nTop split + low-dM datacard: high-dM CRs are inclusive; selected high-dM AN17 SR bins are split into N_t=0 and N_t>=1; low-dM contributes one inclusive SR bin.\n"
        "# The selected split bins are 1-based AN17 search-bin numbers. JES/MET shape nuisances remain deferred.\n",
    )


def write_datacards(channels: list[dict[str, Any]], mass_keys: list[str], template_root: Path, root_summary: dict[str, Any], output_dir: Path, auto_mc_stats: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: dict[str, str] = {}
    for mass_key in mass_keys:
        card = output_dir / f"datacard_{mass_key}.txt"
        card.write_text(partial_lowdm_datacard_text(template_root, channels, mass_key, root_summary, auto_mc_stats))
        cards[mass_key] = str(card)
    return cards


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Combine inputs with selected AN17 SR bins split by nTop plus one-bin low-dM SR.")
    parser.add_argument("--hists", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-bins", nargs="*", type=int, default=DEFAULT_SPLIT_BINS)
    parser.add_argument("--data-mode", choices=["asimov", "observed"], default="asimov")
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--max-mstop", type=int, default=None)
    parser.add_argument("--runner-jobs", type=int, default=8)
    parser.add_argument("--point-timeout", type=int, default=1800)
    args = parser.parse_args()

    split_bins = sorted(set(args.split_bins))
    flat = read_json(Path(args.hists))
    channels = [cr_channel(flat, flat_region, channel_name) for flat_region, channel_name in CONTROL_REGION_MAP.items()]
    channels.append(partial_sr_channel(flat, split_bins))
    channels.append(lowdm_channel(flat))

    high_keys = mass_points_from_partial(flat, args.only, args.max_mstop)
    low_keys = mass_points_from_lowdm(flat, args.only, args.max_mstop)
    mass_keys = sorted(set(high_keys) | set(low_keys), key=lambda key: parse_mass_key(key))
    if args.max_points is not None:
        mass_keys = mass_keys[: args.max_points]
    if not mass_keys:
        raise SystemExit("no signal mass points selected from partial AN17 SR or low-dM SR")

    outdir = Path(args.output_dir)
    template_root = outdir / "templates_partial_an17_nTop_lowdm.root"
    datacard_dir = outdir / "datacards"
    limit_dir = outdir / "limits"
    runner = outdir / "run_combine_expected.sh"
    root_summary = build_root_from_flat(channels, flat, mass_keys, template_root, args.data_mode, split_bins)
    cards = write_datacards(channels, mass_keys, template_root, root_summary, datacard_dir, args.auto_mc_stats)
    write_parallel_runner(cards, limit_dir, runner, args.runner_jobs, args.point_timeout)

    manifest = {
        "status": "combine_inputs_ready",
        "schema": "flat_partial_an17_nTop_split_plus_lowdm_sr_v1",
        "hists": str(Path(args.hists)),
        "template_root": str(template_root),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "data_mode": args.data_mode,
        "mass_point_count": len(mass_keys),
        "mass_points": mass_keys,
        "control_regions": CONTROL_REGION_MAP,
        "highdm_inclusive_scheme": INCLUSIVE_SCHEME,
        "highdm_nt1_scheme": NT1_SCHEME,
        "partial_split_channel": PARTIAL_CHANNEL,
        "partial_split_bins_1based": split_bins,
        "partial_split_policy": "Only requested 1-based AN17 SR bins are split into N_t=0 then N_t>=1; all other AN17 SR bins stay inclusive.",
        "lowdm_signal_scheme": LOWDM_SIGNAL_SCHEME,
        "lowdm_policy": "one inclusive low-dM SR bin added",
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
        "partial_split_bins_1based": split_bins,
        "template_root": str(template_root),
        "runner": str(runner),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
