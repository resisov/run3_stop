#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import uproot

REQUIRED_BRANCHES = {
    "run",
    "luminosityBlock",
    "event",
    "entry",
    "dataset_id",
    "physical_dataset_id",
    "process_id",
    "file_id",
    "year",
    "mStop",
    "mLSP",
    "met",
    "ht",
    "recoil_gcr",
    "recoil_dy2e",
    "recoil_dy2m",
    "gen_weight",
    "pu_ntrueint",
    "feature_flat_preselection",
    "feature_SR_Nt1",
    "feature_lowdm_preselection",
    "feature_lowdm_sr_base",
    "pass_lowdm_topology_veto",
    "pass_lowdm_isr",
    "pass_lowdm_isr_bveto",
    "pass_lowdm_met_sqrt_ht",
    "pass_lowdm_mtb",
    "lowdm_search_bin",
    "lowdm_isr_pt",
    "lowdm_isr_eta",
    "lowdm_isr_phi",
    "lowdm_isr_dphi",
    "lowdm_met_sqrt_ht",
    "lowdm_ptb",
    "lowdm_mtb",
    "n_lowdm_isr",
    "n_sv_softb",
    "nb_loose",
    "nb_medium_lowdm",
    "good_jet_pt",
    "good_jet_eta",
    "good_jet_phi",
    "good_jet_btag_upart",
    "good_jet_hadron_flavour",
    "good_jet_b_loose",
    "good_jet_b_medium",
    "lowdm_fatjet_pt",
    "lowdm_fatjet_eta",
    "lowdm_fatjet_phi",
    "electron_veto_pt",
    "electron_veto_eta_sc",
    "electron_medium_pt",
    "electron_medium_eta_sc",
    "muon_loose_pt",
    "muon_loose_eta",
    "muon_medium_pt",
    "muon_medium_eta",
    "photon_medium_pt",
    "photon_medium_eta",
    "gen_top_pt",
}

OK_STATUSES = {"complete", "complete_with_bad_files"}


def load_expected_names(args_file: Path | None) -> list[str]:
    if args_file is None:
        return []
    names: list[str] = []
    for raw in args_file.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        names.append(raw.split()[0])
    return names


def validate_one(meta_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"name": meta_path.stem, "metadata": str(meta_path), "ok": False, "errors": []}
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        result["errors"].append(f"metadata_read_failed: {type(exc).__name__}: {exc}")
        return result
    result.update({
        "status": meta.get("status"),
        "files_processed": meta.get("files_processed"),
        "files_attempted": meta.get("files_attempted"),
        "events_read": meta.get("events_read"),
        "events_written": meta.get("events_written"),
        "bad_files": len(meta.get("bad_files") or []),
    })
    if meta.get("status") not in OK_STATUSES:
        result["errors"].append(f"bad_status: {meta.get('status')}")
    if meta.get("require_object_corrections") is not True:
        result["errors"].append("object_corrections_not_required")
    policy = meta.get("normalization_policy") or {}
    if "raw_gen_weight_only" not in str(policy.get("root_event_weight_status", "")):
        result["errors"].append("raw_weight_policy_missing")
    root_path = Path(str(meta.get("root_file") or meta_path.with_suffix(".root")))
    if not root_path.exists():
        result["errors"].append(f"root_missing: {root_path}")
        return result
    result["root"] = str(root_path)
    try:
        with uproot.open(root_path) as root_file:
            if "Events" not in {str(k).split(";", 1)[0] for k in root_file.keys()}:
                result["errors"].append("Events_tree_missing")
                return result
            tree = root_file["Events"]
            keys = {str(k).split(";", 1)[0] for k in tree.keys()}
            result["entries"] = int(tree.num_entries)
            result["branches"] = len(keys)
            missing = sorted(REQUIRED_BRANCHES - keys)
            if missing:
                result["errors"].append("missing_required_branches: " + ",".join(missing))
            weight_like = sorted(k for k in keys if k.startswith("weight_"))
            if weight_like:
                result["errors"].append("unexpected_weight_branches: " + ",".join(weight_like[:20]))
            if int(meta.get("events_written") or 0) != int(tree.num_entries):
                result["errors"].append(f"entry_mismatch: meta={meta.get('events_written')} root={tree.num_entries}")
    except Exception as exc:
        result["errors"].append(f"root_open_failed: {type(exc).__name__}: {exc}")
    result["ok"] = not result["errors"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate flat ntuple ROOT/JSON outputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--args-file", default=None)
    parser.add_argument("--require-all-expected", action="store_true")
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--summary-output", default=None)
    opts = parser.parse_args()

    outdir = Path(opts.output_dir)
    expected = load_expected_names(Path(opts.args_file)) if opts.args_file else []
    if opts.names:
        expected = list(opts.names)
    if expected:
        meta_paths = [outdir / f"{name}.json" for name in expected if (outdir / f"{name}.json").exists()]
        missing = [name for name in expected if not (outdir / f"{name}.json").exists()]
    else:
        meta_paths = sorted(outdir.glob("*.json"))
        missing = []

    results = [validate_one(path) for path in sorted(meta_paths)]
    payload = {
        "output_dir": str(outdir),
        "expected": len(expected) if expected else None,
        "metadata_found": len(meta_paths),
        "missing_expected": missing,
        "ok": sum(1 for r in results if r.get("ok")),
        "bad": sum(1 for r in results if not r.get("ok")),
        "results": results,
    }
    payload["status"] = "complete" if payload["bad"] == 0 and (not opts.require_all_expected or not missing) else "failed"
    text = json.dumps(payload, indent=2, sort_keys=True)
    if opts.summary_output:
        Path(opts.summary_output).write_text(text + "\n")
    print(text)
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
