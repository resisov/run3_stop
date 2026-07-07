#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_flat_boosted_recoil_hists import (  # noqa: E402
    READ_BRANCHES,
    RECOIL_BIN_LABELS,
    RECOIL_PT_BINS,
    add_index_hist,
    as_bool,
    data_process_allowed,
    dataset_label,
    empty_index_hist,
    finite_array,
    flat_arrays_for_weights,
    norm_vector,
    note_data_exclusion,
    read_json,
    sample_label,
)
from autonomous_allhad.real_subset_worker import compute_weight_bundle  # noqa: E402

OLD_SELECTED_SCHEME = "boosted_an17_selected_recoil6_SR"
NEW_SELECTED_SCHEME = "boosted_an17_selected_recoil6_with_nt0_SR"
NEW_CATEGORY_KEY = "Nb1plus_T0_W0"
NEW_CATEGORY_LABEL = "$N_{b}\\geq1$, $N_{t}=0$\n$N_{W}=0$"
NEW_CATEGORY_PREFIX = "NT0_Nb1plus_T0_W0"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def chunked(items: list[Path], n: int) -> list[list[Path]]:
    n = max(int(n), 1)
    return [items[i::n] for i in range(n) if items[i::n]]


def selected_nt0_recoil_indices(chunk: dict[str, Any], n: int) -> np.ndarray:
    nb = np.asarray(chunk["nb_medium"], dtype=int)
    nt = np.asarray(chunk["nboosted_top"], dtype=int)
    nw = np.asarray(chunk["nboosted_w"], dtype=int)
    sr = as_bool(chunk["feature_SR"], n)
    recoil = finite_array(chunk["met"], n, 0.0)
    recoil_idx = np.searchsorted(np.asarray(RECOIL_PT_BINS, dtype=float), recoil, side="right") - 1
    mask = sr & (nb >= 1) & (nt == 0) & (nw == 0) & (recoil_idx >= 0) & (recoil_idx < len(RECOIL_PT_BINS) - 1)
    out = np.full(n, -1, dtype=int)
    out[mask] = recoil_idx[mask]
    return out


def merge_hist(target: dict[str, Any], source: dict[str, Any]) -> None:
    for sample, variations in source.items():
        dst_vars = target.setdefault(sample, {})
        for vname, hist in variations.items():
            dst = dst_vars.setdefault(vname, empty_index_hist(len(RECOIL_PT_BINS) - 1))
            dst["sumw"] = (np.asarray(dst["sumw"], dtype=float) + np.asarray(hist.get("sumw") or [], dtype=float)).tolist()
            dst["sumw2"] = (np.asarray(dst["sumw2"], dtype=float) + np.asarray(hist.get("sumw2") or [], dtype=float)).tolist()
            dst["entries"] = (np.asarray(dst["entries"], dtype=int) + np.asarray(hist.get("entries") or [], dtype=int)).astype(int).tolist()


def process_roots(args: tuple[int, list[str], str, str, int]) -> dict[str, Any]:
    worker_id, roots_raw, repo_raw, norm_raw, step_size = args
    repo = Path(repo_raw).resolve()
    norm = read_json(Path(norm_raw))
    hists: dict[str, Any] = {}
    summary: dict[str, Any] = {"worker": worker_id, "roots": 0, "events": 0, "selected_entries": 0, "weight_failures": [], "data_stream_exclusions": {}}
    for pos, raw in enumerate(roots_raw, start=1):
        root_path = Path(raw)
        meta_path = root_path.with_suffix(".json")
        if not root_path.exists() or not meta_path.exists():
            summary.setdefault("missing", []).append(str(root_path))
            continue
        meta = read_json(meta_path)
        try:
            with uproot.open(root_path) as root_file:
                tree = root_file["Events"]
                present = set(tree.keys())
                branches = [b for b in READ_BRANCHES if b in present]
                for chunk in tree.iterate(branches, step_size=step_size, library="ak"):
                    n_chunk = len(chunk["dataset_id"])
                    if n_chunk == 0:
                        continue
                    dsids = np.asarray(chunk["dataset_id"], dtype=np.int64)
                    for dsid in sorted(set(int(x) for x in dsids)):
                        mask_ds = dsids == dsid
                        sub = {name: chunk[name][mask_ds] for name in ak.fields(chunk)}
                        dataset, process, is_data, is_signal = dataset_label(meta, dsid)
                        if is_signal:
                            mstops = np.asarray(sub["mStop"], dtype=int)
                            mlsps = np.asarray(sub["mLSP"], dtype=int)
                            subgroups = [(mstops == ms) & (mlsps == ml) for ms, ml in sorted(set(zip(mstops.tolist(), mlsps.tolist())))]
                        else:
                            subgroups = [np.ones(int(np.count_nonzero(mask_ds)), dtype=bool)]
                        for mask_group in subgroups:
                            if not np.any(mask_group):
                                continue
                            sub_group = {name: arr[mask_group] for name, arr in sub.items()}
                            arrays, inputs = flat_arrays_for_weights(sub_group)
                            indices = selected_nt0_recoil_indices(sub_group, inputs["n"])
                            selected = int(np.count_nonzero(indices >= 0))
                            if selected <= 0:
                                summary["events"] += int(inputs["n"])
                                continue
                            label = sample_label(process, is_data, is_signal, sub_group)
                            if is_data and not data_process_allowed(process, "SR"):
                                note_data_exclusion(summary, NEW_SELECTED_SCHEME, process, selected)
                                summary["events"] += int(inputs["n"])
                                continue
                            try:
                                year_vals = np.asarray(sub_group["year"], dtype=int)
                                year = str(int(year_vals[0])) if len(year_vals) else "2024"
                                _gen, variations, status = compute_weight_bundle(
                                    arrays, repo, dataset, process, year, inputs["n"],
                                    inputs["jet_pt"], inputs["jet_eta"], inputs["jet_hadflav"], inputs["b_med"],
                                    inputs["e_eta"], inputs["e_delta_eta_sc"], inputs["e_pt"], inputs["e_phi"], inputs["e_veto"], inputs["e_med"], inputs["n_e_veto"], inputs["n_e_med"],
                                    inputs["m_eta"], inputs["m_pt"], inputs["m_phi"], inputs["m_loose"], inputs["m_med"], inputs["n_m_loose"], inputs["n_m_med"],
                                    inputs["p_eta"], inputs["p_pt"], inputs["p_phi"], inputs["p_med"], inputs["gcr_mask"],
                                )
                            except Exception as exc:
                                summary["weight_failures"].append({"root": str(root_path), "dataset_id": int(dsid), "dataset": dataset, "process": process, "label": label, "error": f"{type(exc).__name__}: {exc}"[:500]})
                                variations = {"nominal": np.asarray(sub_group["gen_weight"], dtype=float)} if not is_data else {"nominal": np.ones(inputs["n"], dtype=float)}
                            normv = norm_vector(norm, sub_group, dsid, is_data, is_signal)
                            for vname, wraw in variations.items():
                                weights = finite_array(wraw, inputs["n"], 0.0) * normv
                                target = hists.setdefault(label, {}).setdefault(vname, empty_index_hist(len(RECOIL_PT_BINS) - 1))
                                add_index_hist(target, indices, weights)
                            summary["selected_entries"] += selected
                            summary["events"] += int(inputs["n"])
        except Exception as exc:
            summary.setdefault("root_failures", []).append({"root": str(root_path), "error": f"{type(exc).__name__}: {exc}"[:500]})
            continue
        summary["roots"] += 1
        if pos % 25 == 0:
            print(json.dumps({"worker": worker_id, "processed": pos, "roots": len(roots_raw), "selected_entries": summary["selected_entries"]}), flush=True)
    return {"histograms": hists, "summary": summary}


def padded_record(variations: dict[str, Any], vname: str, nbin: int) -> dict[str, Any]:
    rec = variations.get(vname) or variations.get("nominal")
    if not rec:
        return empty_index_hist(nbin)
    vals = list(rec.get("sumw") or [])[:nbin]
    s2 = list(rec.get("sumw2") or [])[:nbin]
    ent = list(rec.get("entries") or [])[:nbin]
    vals += [0.0] * (nbin - len(vals))
    s2 += [0.0] * (nbin - len(s2))
    ent += [0] * (nbin - len(ent))
    return {"sumw": vals, "sumw2": s2, "entries": ent}


def concat_record(new_rec: dict[str, Any], old_rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "sumw": list(new_rec.get("sumw") or []) + list(old_rec.get("sumw") or []),
        "sumw2": list(new_rec.get("sumw2") or []) + list(old_rec.get("sumw2") or []),
        "entries": list(new_rec.get("entries") or []) + list(old_rec.get("entries") or []),
    }


def build_combined_scheme(base: dict[str, Any], new6: dict[str, Any]) -> dict[str, Any]:
    old_by_sample = ((base.get("search_bin_histograms") or {}).get(OLD_SELECTED_SCHEME) or {})
    if not old_by_sample:
        raise ValueError(f"missing old selected scheme: {OLD_SELECTED_SCHEME}")
    combined: dict[str, Any] = {}
    for sample in sorted(set(old_by_sample) | set(new6)):
        old_vars = old_by_sample.get(sample) or {}
        new_vars = new6.get(sample) or {}
        vnames = sorted(set(old_vars) | set(new_vars) | {"nominal"})
        for vname in vnames:
            old_rec = padded_record(old_vars, vname, 42)
            new_rec = padded_record(new_vars, vname, 6)
            combined.setdefault(sample, {})[vname] = concat_record(new_rec, old_rec)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Append the requested Nt=0,NW=0 recoil category in front of selected AN17 recoil6 histograms.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--normalization", default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--step-size", type=int, default=200000)
    parser.add_argument("--max-roots", type=int, default=None)
    args = parser.parse_args()

    start = time.time()
    base_path = Path(args.input)
    base = read_json(base_path)
    roots = [Path(x) for x in ((base.get("summary") or {}).get("input_roots") or [])]
    if args.max_roots is not None:
        roots = roots[: args.max_roots]
    if not roots:
        raise SystemExit("no input roots listed in flat histogram summary")
    norm = args.normalization or base.get("normalization")
    if not norm:
        raise SystemExit("normalization path is missing")

    jobs = max(1, min(int(args.jobs), len(roots)))
    tasks = [(idx, [str(p) for p in subset], args.repo, str(norm), int(args.step_size)) for idx, subset in enumerate(chunked(roots, jobs))]
    merged_new6: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    if jobs == 1:
        results = [process_roots(tasks[0])]
    else:
        with Pool(processes=jobs) as pool:
            results = list(pool.imap_unordered(process_roots, tasks))
    for result in results:
        merge_hist(merged_new6, result.get("histograms") or {})
        summaries.append(result.get("summary") or {})

    new_labels = [f"{NEW_CATEGORY_PREFIX}_recoil_{label}" for label in RECOIL_BIN_LABELS]
    old_labels = list((((base.get("search_bin_schemes") or {}).get(OLD_SELECTED_SCHEME) or {}).get("bin_labels") or []))
    if len(old_labels) != 42:
        raise ValueError(f"expected 42 old selected labels, got {len(old_labels)}")
    combined = build_combined_scheme(base, merged_new6)
    base.setdefault("search_bin_histograms", {})[NEW_SELECTED_SCHEME] = combined
    base.setdefault("search_bin_schemes", {})[NEW_SELECTED_SCHEME] = {
        "bin_labels": new_labels + old_labels,
        "selection": "feature_SR, first category Nb>=1 and Nt=0 and NW=0, followed by selected AN17 bins 4,5,8,9,14,15,16, all split into six recoil/MET bins",
        "prepended_category": {"key": NEW_CATEGORY_KEY, "label": NEW_CATEGORY_LABEL, "recoil_pt_bins": RECOIL_PT_BINS},
        "source_scheme_for_existing_42_bins": OLD_SELECTED_SCHEME,
        "recoil_pt_bins": RECOIL_PT_BINS,
    }
    base.setdefault("summary", {}).setdefault("derived_search_bin_schemes", {})[NEW_SELECTED_SCHEME] = {
        "created_from": str(base_path),
        "new_category": NEW_CATEGORY_KEY,
        "old_scheme": OLD_SELECTED_SCHEME,
        "new_category_selected_entries": int(sum(s.get("selected_entries", 0) for s in summaries)),
        "workers": summaries,
        "elapsed_seconds": time.time() - start,
    }
    write_json(Path(args.output), base)
    print(json.dumps({"status": "complete", "scheme": NEW_SELECTED_SCHEME, "bins": 48, "new_category_entries": int(sum(s.get("selected_entries", 0) for s in summaries)), "output": args.output, "elapsed_seconds": time.time() - start}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
