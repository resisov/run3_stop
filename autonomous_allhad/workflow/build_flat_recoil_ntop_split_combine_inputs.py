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

CONTROL_SPLIT_MAP = [
    ("LLCR_Nt1", "cat2_LLCR_Nt1_highDeltaM"),
    ("LLCR_Nt0", "cat2_LLCR_Nt0_highDeltaM"),
    ("QCDCR_Nt1", "cat3_QCDCR_Nt1_highDeltaM"),
    ("QCDCR_Nt0", "cat3_QCDCR_Nt0_highDeltaM"),
    ("GCR_Nt1", "cat4_GCR_Nt1_highDeltaM"),
    ("GCR_Nt0", "cat4_GCR_Nt0_highDeltaM"),
    ("DY2E_Nt1", "cat5_DY2E_Nt1_highDeltaM"),
    ("DY2E_Nt0", "cat5_DY2E_Nt0_highDeltaM"),
    ("DY2M_Nt1", "cat6_DY2M_Nt1_highDeltaM"),
    ("DY2M_Nt0", "cat6_DY2M_Nt0_highDeltaM"),
]
SIGNAL_SPLIT_MAP = [
    ("SR_Nt1", "cat7_SR_Nt1_recoil"),
    ("SR_Nt0", "cat7_SR_Nt0_recoil"),
]
SIGNAL_REGIONS = {region for region, _ in SIGNAL_SPLIT_MAP}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def mass_points_from_regions(flat: dict[str, Any], regions: list[str], only: list[str] | None, max_points: int | None, max_mstop: int | None) -> list[str]:
    out: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for region in regions:
        by_sample = ((flat.get("histograms") or {}).get(region) or {})
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
    keys = [key for _, _, key in out]
    return keys[:max_points] if max_points is not None else keys


def signal_array_from_region(flat: dict[str, Any], region: str, mass_key: str, nbin: int) -> tuple[np.ndarray, np.ndarray]:
    sample = SIGNAL_PREFIX + mass_key
    rec = (((flat.get("histograms") or {}).get(region) or {}).get(sample) or {}).get("nominal")
    return hist_arrays(rec, nbin)


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
                else:
                    sig = np.zeros(len(bkg), dtype=float)
                    sig_s2 = np.zeros(len(bkg), dtype=float)
                write_hist(directory, proc, sig, sig_s2, edges)
                summary["signals"].setdefault(mass_key, {"process": proc, "channels": {}})["channels"][name] = float(np.sum(sig))
    finally:
        root_file.Close()
    return summary


def split_datacard_text(template_root: Path, channels: list[dict[str, Any]], mass_key: str, root_summary: dict[str, Any], auto_mc_stats: int) -> str:
    text = datacard_text(template_root, channels, mass_key, root_summary, auto_mc_stats)
    return text.replace(
        "# Boosted AN17 datacard: CR channels use 6-bin recoil/U_T histograms; SR uses 17 boosted top/W tagged search bins.\n"
        "# SR background shape nuisances are reconstructed from shard-level search_bin_variations plus JES/MET unclustered shape shards.\n",
        "# Recoil nTop-split datacard: CR and SR channels are split into nTop>=1 and nTop==0 categories with 6-bin recoil_pt templates.\n"
        "# Background shape nuisances are propagated from variations present in the flat histogram payload; JES/MET shape nuisances are intentionally deferred.\n",
    )


def write_datacards(channels: list[dict[str, Any]], mass_keys: list[str], template_root: Path, root_summary: dict[str, Any], output_dir: Path, auto_mc_stats: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: dict[str, str] = {}
    for mass_key in mass_keys:
        card = output_dir / f"datacard_{mass_key}.txt"
        card.write_text(split_datacard_text(template_root, channels, mass_key, root_summary, auto_mc_stats))
        cards[mass_key] = str(card)
    return cards


def write_parallel_runner(cards: dict[str, str], output_dir: Path, runner: Path, jobs: int, point_timeout: int) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "COMBINE=${COMBINE:-combine}",
        f"OUTDIR={stable_path(output_dir)}",
        f"MAX_JOBS={int(jobs)}",
        f"POINT_TIMEOUT={int(point_timeout)}",
        "mkdir -p \"$OUTDIR\"",
        "run_one() {",
        "  local mass=\"$1\"",
        "  local card=\"$2\"",
        "  if compgen -G \"${OUTDIR}/higgsCombine_${mass}.AsymptoticLimits*.root\" >/dev/null; then echo \"[combine-skip] ${mass}\"; return 0; fi",
        "  echo \"[combine-start] ${mass}\"",
        "  (cd \"$OUTDIR\" && timeout \"$POINT_TIMEOUT\" \"$COMBINE\" -M AsymptoticLimits --run blind -n \"_${mass}\" \"$card\") > \"$OUTDIR/log_${mass}.txt\" 2>&1",
        "  local rc=$?",
        "  echo \"[combine-rc] ${mass} rc=${rc}\"",
        "  return ${rc}",
        "}",
        "fail=0",
        "running=0",
    ]
    for mass_key, card in sorted(cards.items()):
        lines.append(f"run_one {mass_key} {stable_path(Path(card))} || fail=1 &")
        lines.append("running=$((running + 1))")
        lines.append("if [ \"$running\" -ge \"$MAX_JOBS\" ]; then wait -n || fail=1; running=$((running - 1)); fi")
    lines.extend(["wait || fail=1", "exit $fail"])
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("\n".join(lines) + "\n")
    runner.chmod(0o755)


def validate_regions(flat: dict[str, Any], regions: list[str]) -> None:
    missing = [region for region in regions if region not in (flat.get("histograms") or {})]
    if missing:
        raise SystemExit("missing split histogram regions: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Combine inputs with CR and SR split into nTop>=1 and nTop==0 recoil channels.")
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

    mass_keys = mass_points_from_regions(flat, signal_regions, args.only, args.max_points, args.max_mstop)
    if not mass_keys:
        raise SystemExit("no signal mass points selected from split SR regions")

    outdir = Path(args.output_dir)
    template_root = outdir / "templates_recoil_ntop_split.root"
    datacard_dir = outdir / "datacards"
    limit_dir = outdir / "limits"
    runner = outdir / "run_combine_expected.sh"
    root_summary = build_root_from_flat(channels, flat, mass_keys, template_root, args.data_mode)
    cards = write_datacards(channels, mass_keys, template_root, root_summary, datacard_dir, args.auto_mc_stats)
    write_parallel_runner(cards, limit_dir, runner, args.runner_jobs, args.point_timeout)

    manifest = {
        "status": "combine_inputs_ready",
        "schema": "flat_recoil_ntop_split_cr_sr_6bin_v1",
        "hists": str(hists_path),
        "template_root": str(template_root),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "data_mode": args.data_mode,
        "mass_point_count": len(mass_keys),
        "mass_points": mass_keys,
        "max_mstop_policy": f"mStop < {args.max_mstop}" if args.max_mstop is not None else "none",
        "control_split_regions": dict(CONTROL_SPLIT_MAP),
        "signal_split_regions": dict(SIGNAL_SPLIT_MAP),
        "sr_bin_count_per_channel": len(flat.get("recoil_pt_bins") or []) - 1,
        "sr_bin_edges": flat.get("recoil_pt_bins"),
        "channels": [ch["name"] for ch in channels],
        "systematics_policy": {
            "source": "shape variations already present in the input flat histogram payload",
            "deferred": ["jesTotal", "metUnclustered"],
            "lumi": {"name": LUMI_NAME, "lnN": LUMI_LNN},
            "autoMCStats": args.auto_mc_stats,
        },
        "root_summary": root_summary,
    }
    write_json(outdir / "combine_input_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "channels": len(channels),
        "control_split_channels": len(CONTROL_SPLIT_MAP),
        "signal_split_channels": len(SIGNAL_SPLIT_MAP),
        "sr_bins_per_channel": manifest["sr_bin_count_per_channel"],
        "mass_points": len(mass_keys),
        "datacards": len(cards),
        "template_root": str(template_root),
        "runner": str(runner),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
