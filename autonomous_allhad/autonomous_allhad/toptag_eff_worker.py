from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

from .real_subset_worker import (
    BOOSTED_TOP_SCORE_BRANCH,
    BOOSTED_TOP_SCORE_WP,
    FATJET_ID_INPUTS,
    ak8_tight_lepton_veto_mask,
    apply_jec,
    cleanup_xrd_cache,
    open_root_with_xrd_fallback,
)


SCHEMA_VERSION = "toptag_eff_shard_v1"
PT_EDGES = np.asarray([400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0])
ABSETA_EDGES = np.asarray([0.0, 0.8, 1.5, 2.0, 2.5])
TOP_PT_MIN = 400.0
TOP_ABSETA_MAX = 2.5
TOP_MSD_MIN = 105.0

REQUIRED_BRANCHES = [
    "run",
    "Rho_fixedGridRhoFastjetAll",
    "FatJet_pt",
    "FatJet_eta",
    "FatJet_phi",
    "FatJet_mass",
    "FatJet_area",
    "FatJet_msoftdrop",
    BOOSTED_TOP_SCORE_BRANCH,
    *FATJET_ID_INPUTS,
]
OPTIONAL_BRANCHES = ["genWeight"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def empty_histogram() -> np.ndarray:
    return np.zeros((len(ABSETA_EDGES) - 1, len(PT_EDGES) - 1), dtype=np.float64)


def efficiency_key(record: dict[str, Any]) -> str:
    key = str(
        record.get("dataset")
        or record.get("sample_name")
        or record.get("dataset_key")
        or record.get("process_group")
        or record.get("process")
        or "unknown"
    )
    return re.sub(r"____\d+_$", "", key)


def fill_histogram(
    eta: np.ndarray,
    pt: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if len(pt) == 0:
        return empty_histogram()
    # The last pT bin includes overflow; eta is selected strictly below 2.5.
    pt = np.minimum(pt, np.nextafter(PT_EDGES[-1], PT_EDGES[0]))
    hist, _, _ = np.histogram2d(
        eta,
        pt,
        bins=(ABSETA_EDGES, PT_EDGES),
        weights=weights,
    )
    return hist.astype(np.float64, copy=False)


def chunk_histograms(arrays: Any, repo: Path, year: str, process: str) -> dict[str, Any]:
    fj_pt_raw = arrays["FatJet_pt"]
    fj_eta = arrays["FatJet_eta"]
    fj_phi = arrays["FatJet_phi"]
    fj_mass_raw = arrays["FatJet_mass"]
    fj_msd = arrays["FatJet_msoftdrop"]
    fj_score = arrays[BOOSTED_TOP_SCORE_BRANCH]

    fj_pt, _, jec_status = apply_jec(
        arrays,
        repo,
        year,
        process,
        "FatJet",
        fj_pt_raw,
        fj_eta,
        fj_phi,
        fj_mass_raw,
        "nominal",
    )
    if not jec_status.get("applied"):
        raise RuntimeError(f"AK8 JEC was not applied: {jec_status}")

    jet_id, jet_id_source = ak8_tight_lepton_veto_mask(arrays, fj_pt, fj_eta, repo)
    eligible = (
        jet_id
        & (fj_pt > TOP_PT_MIN)
        & (abs(fj_eta) < TOP_ABSETA_MAX)
        & (fj_msd > TOP_MSD_MIN)
    )
    passed = eligible & (fj_score > BOOSTED_TOP_SCORE_WP)

    pt_all = np.asarray(ak.to_numpy(ak.flatten(fj_pt[eligible])), dtype=np.float64)
    eta_all = np.asarray(ak.to_numpy(ak.flatten(abs(fj_eta[eligible]))), dtype=np.float64)
    pt_pass = np.asarray(ak.to_numpy(ak.flatten(fj_pt[passed])), dtype=np.float64)
    eta_pass = np.asarray(ak.to_numpy(ak.flatten(abs(fj_eta[passed]))), dtype=np.float64)

    finite_all = np.isfinite(pt_all) & np.isfinite(eta_all)
    finite_pass = np.isfinite(pt_pass) & np.isfinite(eta_pass)
    pt_all, eta_all = pt_all[finite_all], eta_all[finite_all]
    pt_pass, eta_pass = pt_pass[finite_pass], eta_pass[finite_pass]

    total_signed = empty_histogram()
    passed_signed = empty_histogram()
    if "genWeight" in ak.fields(arrays):
        jet_weight, _ = ak.broadcast_arrays(arrays["genWeight"], fj_pt)
        weight_all = np.asarray(
            ak.to_numpy(ak.flatten(jet_weight[eligible])), dtype=np.float64
        )[finite_all]
        weight_pass = np.asarray(
            ak.to_numpy(ak.flatten(jet_weight[passed])), dtype=np.float64
        )[finite_pass]
        total_signed = fill_histogram(eta_all, pt_all, weight_all)
        passed_signed = fill_histogram(eta_pass, pt_pass, weight_pass)

    return {
        "total": fill_histogram(eta_all, pt_all),
        "passed": fill_histogram(eta_pass, pt_pass),
        "total_signed": total_signed,
        "passed_signed": passed_signed,
        "events": len(arrays),
        "eligible_jets": len(pt_all),
        "passed_jets": len(pt_pass),
        "jec_status": jec_status,
        "jet_id_source": jet_id_source,
    }


def process_file_once(
    record: dict[str, Any],
    repo: Path,
    chunk_size: int,
    max_events: int | None,
    prefer_cache: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    file_path = str(record["file_path"])
    process = efficiency_key(record)
    year = str(record.get("year") or "2024")
    old_prefer_cache = os.environ.get("AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE")
    os.environ["AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE"] = "1" if prefer_cache else "0"
    root = None
    access: dict[str, Any] = {}
    try:
        root, access = open_root_with_xrd_fallback(file_path)
        if "Events" not in root:
            raise RuntimeError("Events tree is missing")
        tree = root["Events"]
        available = set(tree.keys())
        missing = [name for name in REQUIRED_BRANCHES if name not in available]
        if missing:
            raise RuntimeError(f"required branches are missing: {missing}")
        branches = REQUIRED_BRANCHES + [name for name in OPTIONAL_BRANCHES if name in available]

        accum = {
            "total": empty_histogram(),
            "passed": empty_histogram(),
            "total_signed": empty_histogram(),
            "passed_signed": empty_histogram(),
            "events": 0,
            "eligible_jets": 0,
            "passed_jets": 0,
        }
        jec_status: dict[str, Any] | None = None
        jet_id_source: str | None = None
        for arrays in tree.iterate(
            branches,
            step_size=chunk_size,
            entry_stop=max_events,
            library="ak",
        ):
            out = chunk_histograms(arrays, repo, year, process)
            for name in ("total", "passed", "total_signed", "passed_signed"):
                accum[name] += out[name]
            for name in ("events", "eligible_jets", "passed_jets"):
                accum[name] += int(out[name])
            jec_status = out["jec_status"]
            jet_id_source = out["jet_id_source"]

        info = {
            "file_path": file_path,
            "dataset": record.get("dataset") or record.get("sample_name"),
            "process": process,
            "process_group": record.get("process_group") or record.get("process") or "unknown",
            "year": year,
            "status": "complete",
            "events": accum["events"],
            "eligible_jets": accum["eligible_jets"],
            "passed_jets": accum["passed_jets"],
            "access": access,
            "jec_status": jec_status,
            "jet_id_source": jet_id_source,
        }
        return accum, info
    finally:
        try:
            if root is not None:
                root.close()
        finally:
            cleanup_xrd_cache(access)
            if old_prefer_cache is None:
                os.environ.pop("AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE", None)
            else:
                os.environ["AUTONOMOUS_ALLHAD_XRD_PREFER_CACHE"] = old_prefer_cache


def process_record(
    record: dict[str, Any],
    repo: Path,
    chunk_size: int,
    max_events: int | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    last_exc: Exception | None = None
    # Sparse remote reading is much faster for this small branch set. If it fails,
    # retry by copying the full file through the established XRootD cache path.
    for prefer_cache in (False, True):
        try:
            out, info = process_file_once(
                record,
                repo,
                chunk_size,
                max_events,
                prefer_cache=prefer_cache,
            )
            info["attempts"] = attempts + [
                {"mode": "xrdcp_cache" if prefer_cache else "sparse_remote", "status": "success"}
            ]
            return out, info
        except Exception as exc:
            last_exc = exc
            attempts.append(
                {
                    "mode": "xrdcp_cache" if prefer_cache else "sparse_remote",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                }
            )
    return None, {
        "file_path": str(record.get("file_path")),
        "dataset": record.get("dataset") or record.get("sample_name"),
        "process": efficiency_key(record),
        "process_group": record.get("process_group") or record.get("process") or "unknown",
        "year": str(record.get("year") or "2024"),
        "status": "failed",
        "error": f"{type(last_exc).__name__}: {last_exc}"[:1000] if last_exc else "unknown",
        "attempts": attempts,
    }


def save_npz(path: Path, processes: list[str], accumulators: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(processes), len(ABSETA_EDGES) - 1, len(PT_EDGES) - 1)

    def stack(name: str) -> np.ndarray:
        if not processes:
            return np.zeros(shape, dtype=np.float64)
        return np.stack([accumulators[p][name] for p in processes], axis=0)

    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as fout:
            np.savez_compressed(
                fout,
                schema_version=np.asarray([SCHEMA_VERSION]),
                process=np.asarray(processes),
                pt_edges=PT_EDGES,
                abseta_edges=ABSETA_EDGES,
                total=stack("total"),
                passed=stack("passed"),
                total_signed=stack("total_signed"),
                passed_signed=stack("passed_signed"),
            )
            fout.flush()
            os.fsync(fout.fileno())
        if tmp.stat().st_size == 0:
            raise RuntimeError(f"NPZ writer produced an empty temporary file: {tmp}")
        os.replace(tmp, path)

        required = {
            "schema_version",
            "process",
            "pt_edges",
            "abseta_edges",
            "total",
            "passed",
            "total_signed",
            "passed_signed",
        }
        with np.load(path, allow_pickle=False) as payload:
            missing = required.difference(payload.files)
            if missing:
                raise RuntimeError(f"NPZ output is missing arrays: {sorted(missing)}")
            for name in required:
                np.asarray(payload[name])
        if path.stat().st_size == 0:
            raise RuntimeError(f"NPZ output is empty after atomic rename: {path}")
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure per-process 2024 Top-tag MC efficiencies")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-events-per-file", type=int)
    args = parser.parse_args()

    started = time.time()
    repo = args.repo.resolve()
    shard = json.loads(args.shard.read_text())
    records = list(shard.get("records", []))
    if args.max_records is not None:
        records = records[: args.max_records]
    records = [r for r in records if not r.get("is_data")]

    accumulators: dict[str, dict[str, np.ndarray]] = {}
    file_summaries: list[dict[str, Any]] = []
    for record in records:
        process = efficiency_key(record)
        accumulators.setdefault(
            process,
            {
                "total": empty_histogram(),
                "passed": empty_histogram(),
                "total_signed": empty_histogram(),
                "passed_signed": empty_histogram(),
            },
        )
        out, info = process_record(record, repo, args.chunk_size, args.max_events_per_file)
        file_summaries.append(info)
        if out is None:
            continue
        for name in ("total", "passed", "total_signed", "passed_signed"):
            accumulators[process][name] += out[name]

    processes = sorted(accumulators)
    save_npz(args.output, processes, accumulators)
    failed = [x for x in file_summaries if x.get("status") != "complete"]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not failed else "complete_with_bad_files",
        "shard": str(args.shard),
        "output": str(args.output),
        "processes": processes,
        "records_requested": len(records),
        "files_processed": len(file_summaries) - len(failed),
        "files_failed": len(failed),
        "events": sum(int(x.get("events", 0)) for x in file_summaries),
        "eligible_jets": sum(int(x.get("eligible_jets", 0)) for x in file_summaries),
        "passed_jets": sum(int(x.get("passed_jets", 0)) for x in file_summaries),
        "definition": {
            "score_branch": BOOSTED_TOP_SCORE_BRANCH,
            "score_wp": BOOSTED_TOP_SCORE_WP,
            "pt_min_gev": TOP_PT_MIN,
            "abseta_max": TOP_ABSETA_MAX,
            "msoftdrop_min_gev": TOP_MSD_MIN,
            "jet_id": "AK8PUPPI_TightLeptonVeto",
            "pt_edges_gev": PT_EDGES.tolist(),
            "abseta_edges": ABSETA_EDGES.tolist(),
            "counting": "unweighted jets; signed genWeight sums stored as diagnostics",
        },
        "files": file_summaries,
        "bad_files": failed,
        "wall_time_s": time.time() - started,
    }
    write_json(args.metadata_output, metadata)
    # A bad input file is recorded and excluded, but it must not kill the shard.
    # Campaign-level validation can retry those files from the emitted manifest.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
