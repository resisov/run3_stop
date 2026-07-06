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
    build_root,
    datacard_text,
    pair_variations,
    stable_path,
    unit_edges,
    write_datacards,
    write_hist,
    write_json,
)

CONTROL_REGION_MAP = {
    "LLCR": "cat2_LLCR_highDeltaM",
    "QCDCR": "cat3_QCDCR_highDeltaM",
    "GCR": "cat4_GCR_highDeltaM",
    "DY2E": "cat5_DY2E_highDeltaM",
    "DY2M": "cat6_DY2M_highDeltaM",
}
SIGNAL_PREFIX = "T2tt_"
SR_CHANNEL = "cat7_SR_recoil"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sample_is_signal(name: str) -> bool:
    return name.startswith(SIGNAL_PREFIX)


def sample_to_mass_key(name: str) -> str | None:
    match = re.match(r"T2tt_(mStop\d+_mLSP\d+)$", name)
    return match.group(1) if match else None


def parse_mass_key(key: str) -> tuple[int, int]:
    match = re.match(r"mStop(\d+)_mLSP(\d+)$", key)
    if not match:
        raise ValueError(f"invalid mass key: {key}")
    return int(match.group(1)), int(match.group(2))


def signal_process_name(mass_key: str) -> str:
    return "sig_" + re.sub(r"[^A-Za-z0-9_]+", "_", mass_key).strip("_")


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


def recoil_channel(flat: dict[str, Any], flat_region: str, channel_name: str, kind: str) -> dict[str, Any]:
    by_sample = ((flat.get("histograms") or {}).get(flat_region) or {})
    edges = np.asarray(flat.get("recoil_pt_bins") or [], dtype=float)
    nbin = max(len(edges) - 1, 0)
    if nbin <= 0:
        raise ValueError("missing recoil_pt_bins")
    if not by_sample:
        raise ValueError(f"missing histogram region {flat_region}")
    bkg, bkg_s2, data, data_s2, backgrounds = aggregate_nominal(by_sample, nbin)
    return {
        "name": channel_name,
        "source_region": flat_region,
        "kind": kind,
        "edges": edges,
        "background": bkg,
        "background_sumw2": bkg_s2,
        "data": data,
        "data_sumw2": data_s2,
        "variations": aggregate_variations(by_sample, backgrounds, nbin, bkg),
        "variable": "recoil_pt",
        "bin_labels": [f"{edges[i]:.0f}-{edges[i + 1]:.0f}" for i in range(nbin)],
        "background_samples": backgrounds,
    }


def mass_points_from_region(flat: dict[str, Any], region: str, only: list[str] | None, max_points: int | None, max_mstop: int | None) -> list[str]:
    by_sample = ((flat.get("histograms") or {}).get(region) or {})
    out: list[tuple[int, int, str]] = []
    for sample in by_sample:
        mass_key = sample_to_mass_key(sample)
        if not mass_key:
            continue
        if only and mass_key not in only:
            continue
        mstop, mlsp = parse_mass_key(mass_key)
        if mlsp >= mstop:
            continue
        if max_mstop is not None and mstop >= max_mstop:
            continue
        out.append((mstop, mlsp, mass_key))
    out.sort()
    keys = [key for _, _, key in out]
    return keys[:max_points] if max_points is not None else keys


def signal_array_from_region(flat: dict[str, Any], region: str, mass_key: str, nbin: int) -> tuple[np.ndarray, np.ndarray]:
    sample = SIGNAL_PREFIX + mass_key
    rec = (((flat.get("histograms") or {}).get(region) or {}).get(sample) or {}).get("nominal")
    return hist_arrays(rec, nbin)


def build_root_from_flat(channels: list[dict[str, Any]], flat: dict[str, Any], sr_region: str, mass_keys: list[str], output_root: Path, data_mode: str) -> dict[str, Any]:
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
                "bin_count": int(len(bkg)),
                "background_yield": float(np.sum(bkg)),
                "data_yield": float(np.sum(data)),
                "data_mode": data_mode,
                "background_shape_nuisances": sorted((channel.get("variations") or {}).keys()),
                "bin_labels": channel.get("bin_labels") or [],
            }
            for mass_key in mass_keys:
                proc = signal_process_name(mass_key)
                if name == SR_CHANNEL:
                    sig, sig_s2 = signal_array_from_region(flat, sr_region, mass_key, len(bkg))
                else:
                    sig = np.zeros(len(bkg), dtype=float)
                    sig_s2 = np.zeros(len(bkg), dtype=float)
                write_hist(directory, proc, sig, sig_s2, edges)
                summary["signals"].setdefault(mass_key, {"process": proc, "channels": {}})["channels"][name] = float(np.sum(sig))
    finally:
        root_file.Close()
    return summary


def recoil_datacard_text(template_root: Path, channels: list[dict[str, Any]], mass_key: str, root_summary: dict[str, Any], auto_mc_stats: int) -> str:
    text = datacard_text(template_root, channels, mass_key, root_summary, auto_mc_stats)
    return text.replace(
        "# Boosted AN17 datacard: CR channels use 6-bin recoil/U_T histograms; SR uses 17 boosted top/W tagged search bins.\n"
        "# SR background shape nuisances are reconstructed from shard-level search_bin_variations plus JES/MET unclustered shape shards.\n",
        "# Recoil SR datacard: CR channels and SR all use 6-bin recoil_pt templates from flat histograms.\n"
        "# Background shape nuisances are propagated from variations already present in the flat histogram payload.\n",
    )


def write_recoil_datacards(channels: list[dict[str, Any]], mass_keys: list[str], template_root: Path, root_summary: dict[str, Any], output_dir: Path, auto_mc_stats: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards: dict[str, str] = {}
    for mass_key in mass_keys:
        card = output_dir / f"datacard_{mass_key}.txt"
        card.write_text(recoil_datacard_text(template_root, channels, mass_key, root_summary, auto_mc_stats))
        cards[mass_key] = str(card)
    return cards


def write_tolerant_runner(cards: dict[str, str], output_dir: Path, runner: Path, jobs: int = 8, point_timeout: int = 1800) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Combine inputs with CR-like 6-bin recoil SR templates.")
    parser.add_argument("--hists", required=True)
    parser.add_argument("--sr-region", choices=["SR", "SR_Nt1", "SR_Nt0"], required=True)
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
    channels = [recoil_channel(flat, flat_region, channel_name, "control_recoil_6bin_flat") for flat_region, channel_name in CONTROL_REGION_MAP.items()]
    if args.sr_region == "SR":
        sr_kind = "signal_recoil_6bin_flat"
    elif args.sr_region == "SR_Nt1":
        sr_kind = "signal_recoil_6bin_flat_nT_ge_1"
    else:
        sr_kind = "signal_recoil_6bin_flat_nT_eq_0"
    channels.append(recoil_channel(flat, args.sr_region, SR_CHANNEL, sr_kind))
    mass_keys = mass_points_from_region(flat, args.sr_region, args.only, args.max_points, args.max_mstop)
    if not mass_keys:
        raise SystemExit(f"no signal mass points selected from {args.sr_region}")

    outdir = Path(args.output_dir)
    template_root = outdir / f"templates_recoil_{args.sr_region}.root"
    datacard_dir = outdir / "datacards"
    limit_dir = outdir / "limits"
    runner = outdir / "run_combine_expected.sh"
    root_summary = build_root_from_flat(channels, flat, args.sr_region, mass_keys, template_root, args.data_mode)
    cards = write_recoil_datacards(channels, mass_keys, template_root, root_summary, datacard_dir, args.auto_mc_stats)
    write_tolerant_runner(cards, limit_dir, runner, args.runner_jobs, args.point_timeout)
    manifest = {
        "status": "combine_inputs_ready",
        "schema": "flat_recoil_sr_6bin_v1",
        "hists": str(hists_path),
        "sr_region": args.sr_region,
        "template_root": str(template_root),
        "datacard_dir": str(datacard_dir),
        "limit_dir": str(limit_dir),
        "runner": str(runner),
        "data_mode": args.data_mode,
        "mass_point_count": len(mass_keys),
        "mass_points": mass_keys,
        "max_mstop_policy": f"mStop < {args.max_mstop}" if args.max_mstop is not None else "none",
        "sr_channel": SR_CHANNEL,
        "sr_bin_count": 6,
        "sr_bin_edges": flat.get("recoil_pt_bins"),
        "sr_bin_selection": "inclusive_SR" if args.sr_region == "SR" else ("SR_nT_ge_1" if args.sr_region == "SR_Nt1" else "SR_nT_eq_0"),
        "channels": [ch["name"] for ch in channels],
        "systematics_policy": {
            "source": "shape variations already present in the input flat histogram payload",
            "upstream_note": "btag SF weight variations and any external JEC/MET shape variations must be restored before this builder is run; this builder propagates them into ROOT templates/datacards.",
            "lumi": {"name": LUMI_NAME, "lnN": LUMI_LNN},
            "autoMCStats": args.auto_mc_stats,
        },
        "root_summary": root_summary,
    }
    write_json(outdir / "combine_input_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "sr_region": args.sr_region,
        "channels": len(channels),
        "sr_bins": manifest["sr_bin_count"],
        "mass_points": len(mass_keys),
        "datacards": len(cards),
        "template_root": str(template_root),
        "runner": str(runner),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
