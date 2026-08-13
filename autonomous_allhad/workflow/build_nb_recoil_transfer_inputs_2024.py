#!/usr/bin/env python3
"""Build exact Nb x recoil transfer-factor inputs from nominal feature ROOTs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import build_flat_boosted_recoil_hists as bh  # noqa: E402
from autonomous_allhad.sidecar_store import read_root_metadata  # noqa: E402


HIGH_EDGES = np.asarray(bh.RECOIL_PT_BINS, dtype=float)
LOW_EDGES = np.asarray(
    [250.0, 300.0, 350.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0],
    dtype=float,
)
HIGH_GROUPS = ("Nb1", "Nb2", "Nb3plus")
LOW_GROUPS = ("Nb1", "Nb2plus")
REGIONS = ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR")
LOW_VALUE_BRANCH = {
    "LLCR": "met",
    "QCDCR": "met",
    "GCR": "recoil_gcr",
    "DY2E": "recoil_dy2e",
    "DY2M": "recoil_dy2m",
    "SR": "met",
}
HIGH_NB_BRANCH = {
    "LLCR": "nb_medium",
    "QCDCR": "nb_medium",
    "GCR": "nb_photon_clean",
    "DY2E": "nb_lepton_clean",
    "DY2M": "nb_lepton_clean",
    "SR": "nb_medium",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empty_leaf(nbin: int) -> dict[str, Any]:
    return {
        "sumw": [0.0] * nbin,
        "sumw2": [0.0] * nbin,
        "entries": [0] * nbin,
    }


def fill_leaf(
    leaf: dict[str, Any],
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
) -> None:
    valid = (
        mask
        & np.isfinite(values)
        & bh.finite_hist_weight_mask(weights)
        & (values >= edges[0])
        & (values <= edges[-1])
    )
    if not np.any(valid):
        return
    index = np.searchsorted(edges, values[valid], side="right") - 1
    index = np.minimum(index, len(edges) - 2)
    selected_weights = weights[valid]
    leaf["sumw"] = (
        np.asarray(leaf["sumw"], dtype=float)
        + np.bincount(index, weights=selected_weights, minlength=len(edges) - 1)
    ).tolist()
    leaf["sumw2"] = (
        np.asarray(leaf["sumw2"], dtype=float)
        + np.bincount(
            index, weights=selected_weights * selected_weights, minlength=len(edges) - 1
        )
    ).tolist()
    leaf["entries"] = (
        np.asarray(leaf["entries"], dtype=np.int64)
        + np.bincount(index, minlength=len(edges) - 1)
    ).tolist()


def fill_index_leaf(
    leaf: dict[str, Any],
    indices: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> None:
    valid = (
        mask
        & bh.finite_hist_weight_mask(weights)
        & (indices >= 0)
        & (indices < len(leaf["sumw"]))
    )
    if not np.any(valid):
        return
    idx = indices[valid].astype(np.int64, copy=False)
    selected_weights = weights[valid]
    leaf["sumw"] = (
        np.asarray(leaf["sumw"], dtype=float)
        + np.bincount(idx, weights=selected_weights, minlength=len(leaf["sumw"]))
    ).tolist()
    leaf["sumw2"] = (
        np.asarray(leaf["sumw2"], dtype=float)
        + np.bincount(
            idx, weights=selected_weights * selected_weights, minlength=len(leaf["sumw2"])
        )
    ).tolist()
    leaf["entries"] = (
        np.asarray(leaf["entries"], dtype=np.int64)
        + np.bincount(idx, minlength=len(leaf["entries"]))
    ).tolist()


def nb_masks(values: np.ndarray, highdm: bool) -> dict[str, np.ndarray]:
    if highdm:
        return {
            "Nb1": values == 1,
            "Nb2": values == 2,
            "Nb3plus": values >= 3,
        }
    return {
        "Nb1": values == 1,
        "Nb2plus": values >= 2,
    }


def nested_leaf(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    nbin: int,
) -> dict[str, Any]:
    target = payload
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    return target.setdefault(keys[-1], empty_leaf(nbin))


def process_root(
    root_name: str,
    repo_name: str,
    normalization_name: str,
    dy_policy: str,
    step_size: int,
    include_data: bool,
    nominal_only: bool,
    lowdm_llcr_nb1_only: bool,
    analysis_sf_components: tuple[str, ...],
    regions: tuple[str, ...],
) -> dict[str, Any]:
    root_path = Path(root_name)
    repo = Path(repo_name)
    norm = read_json(Path(normalization_name))
    output: dict[str, Any] = {
        "highdm": {"recoil": {}, "sr_components": {}},
        "lowdm": {"recoil": {}, "search_components": {}},
        "summary": {
            "input_root": str(root_path),
            "events_scanned": 0,
            "events_weighted": 0,
            "datasets": {},
        },
    }
    if not root_path.exists():
        output["summary"]["missing"] = True
        return output
    try:
        meta = read_root_metadata(root_path)
    except FileNotFoundError:
        output["summary"]["missing"] = True
        return output
    with uproot.open(root_path) as root_file:
        tree = root_file["Events"]
        present = set(tree.keys())
        branches = [branch for branch in bh.READ_BRANCHES if branch in present]
        iterator = bh.iterate_tree_with_dy_policy(
            tree,
            branches,
            meta,
            dy_policy,
            output["summary"],
            step_size,
        )
        for chunk in iterator:
            n_chunk = len(chunk["dataset_id"])
            output["summary"]["events_scanned"] += n_chunk
            dataset_ids = np.asarray(chunk["dataset_id"], dtype=np.int64)
            for dataset_id in sorted(set(int(value) for value in dataset_ids)):
                selected_dataset = dataset_ids == dataset_id
                sub = {
                    name: chunk[name][selected_dataset] for name in ak.fields(chunk)
                }
                dataset, process, is_data, is_signal = bh.dataset_label(
                    meta, dataset_id
                )
                if (
                    is_signal
                    or (is_data and not include_data)
                    or (
                        is_data
                        and not any(
                            bh.data_process_allowed(process, region)
                            for region in regions
                        )
                    )
                    or not bh.dy_ptll_dataset_allowed(
                    dataset, process, dy_policy
                    )
                ):
                    continue
                arrays, inputs = bh.flat_arrays_for_weights(sub)
                year_values = np.asarray(sub["year"], dtype=int)
                year = str(int(year_values[0])) if len(year_values) else "2024"
                norm_values = bh.norm_vector(
                    norm,
                    sub,
                    dataset_id,
                    dataset,
                    is_data=is_data,
                    is_signal=False,
                    require_normalization=not is_data,
                )
                if is_data:
                    variations = {"nominal": np.ones(inputs["n"], dtype=float)}
                else:
                    _generator, variations, status = bh.compute_weight_bundle(
                        arrays,
                        repo,
                        dataset,
                        process,
                        year,
                        inputs["n"],
                        inputs["jet_pt"],
                        inputs["jet_eta"],
                        inputs["jet_hadflav"],
                        inputs["b_med"],
                        inputs["e_eta"],
                        inputs["e_delta_eta_sc"],
                        inputs["e_pt"],
                        inputs["e_phi"],
                        inputs["e_veto"],
                        inputs["e_med"],
                        inputs["n_e_veto"],
                        inputs["n_e_med"],
                        inputs["m_eta"],
                        inputs["m_pt"],
                        inputs["m_phi"],
                        inputs["m_loose"],
                        inputs["m_med"],
                        inputs["n_m_loose"],
                        inputs["n_m_med"],
                        inputs["p_eta"],
                        inputs["p_pt"],
                        inputs["p_phi"],
                        inputs["p_med"],
                        inputs["gcr_mask"],
                        met_pt=inputs["met_pt"],
                        met_trigger_mask=inputs["met_trigger_mask"],
                        analysis_sf_components=analysis_sf_components,
                    )
                    btag_status = (
                        (status.get("components") or {}).get("btagSF") or {}
                    )
                    if not btag_status.get("applied"):
                        raise RuntimeError(
                            f"{root_path}: btagSF unavailable for {dataset}: "
                            f"{btag_status}"
                        )
                    for component in analysis_sf_components:
                        component_status = (
                            (status.get("components") or {}).get(component) or {}
                        )
                        if not component_status.get("applied"):
                            raise RuntimeError(
                                f"{root_path}: required analysis SF {component} "
                                f"unavailable for {dataset}: {component_status}"
                            )
                    variations = bh.histogram_variations(
                        variations,
                        nominal_only=nominal_only,
                    )
                label = bh.sample_label(
                    process,
                    is_data,
                    False,
                    sub,
                    dataset,
                )
                output["summary"]["datasets"][dataset] = (
                    int(output["summary"]["datasets"].get(dataset, 0)) + inputs["n"]
                )
                output["summary"]["events_weighted"] += inputs["n"]

                if lowdm_llcr_nb1_only:
                    if is_data and not bh.data_process_allowed(process, "LLCR"):
                        continue
                    low_index = bh.lowdm_nbge1_indices(
                        np.where(
                            bh.lowdm_region_mask(sub, "LLCR", inputs["n"]),
                            bh.int_field(
                                sub,
                                "lowdm_search_bin_LLCR",
                                inputs["n"],
                                -1,
                            ),
                            -1,
                        )
                    )
                    group_mask = (low_index >= 0) & (low_index < 16)
                    low_values = bh.finite_array(
                        sub["met"],
                        inputs["n"],
                        0.0,
                    )
                    for variation_name, raw_weight in variations.items():
                        weights = (
                            bh.finite_array(raw_weight, inputs["n"], 0.0)
                            * norm_values
                        )
                        fill_leaf(
                            nested_leaf(
                                output["lowdm"]["recoil"],
                                (
                                    "LLCR",
                                    "Nb1",
                                    label,
                                    variation_name,
                                ),
                                len(LOW_EDGES) - 1,
                            ),
                            low_values,
                            weights,
                            group_mask,
                            LOW_EDGES,
                        )
                    continue

                high_sr_index = (
                    bh.selected_an17_recoil60_indices(
                        sub,
                        inputs["n"],
                        bh.as_bool(sub["feature_SR"], inputs["n"]),
                    )
                    if "SR" in regions
                    else np.full(inputs["n"], -1, dtype=np.int64)
                )
                low_indices = {
                    region: bh.lowdm_nbge1_indices(
                        np.where(
                            bh.lowdm_region_mask(sub, region, inputs["n"]),
                            bh.int_field(
                                sub,
                                f"lowdm_search_bin_{region}",
                                inputs["n"],
                                -1,
                            ),
                            -1,
                        )
                    )
                    for region in regions
                }
                for variation_name, raw_weight in variations.items():
                    weights = (
                        bh.finite_array(raw_weight, inputs["n"], 0.0) * norm_values
                    )
                    for region in regions:
                        # Never record the blinded signal-region observation.
                        if is_data and region == "SR":
                            continue
                        if is_data and not bh.data_process_allowed(process, region):
                            continue
                        if region == "SR":
                            high_mask = high_sr_index >= 0
                            high_values = bh.finite_array(
                                sub["met"], inputs["n"], 0.0
                            )
                        else:
                            flag, value_branch = bh.REGION_VARIABLES[region]
                            high_mask = bh.region_mask(
                                sub, region, flag, inputs["n"]
                            )
                            high_values = bh.finite_array(
                                sub[value_branch], inputs["n"], 0.0
                            )
                        high_nb = bh.int_field(
                            sub,
                            HIGH_NB_BRANCH[region],
                            inputs["n"],
                            -1,
                        )
                        for group, group_mask in nb_masks(
                            high_nb, highdm=True
                        ).items():
                            fill_leaf(
                                nested_leaf(
                                    output["highdm"]["recoil"],
                                    (
                                        region,
                                        group,
                                        label,
                                        variation_name,
                                    ),
                                    len(HIGH_EDGES) - 1,
                                ),
                                high_values,
                                weights,
                                high_mask & group_mask,
                                HIGH_EDGES,
                            )
                            if region == "SR":
                                fill_index_leaf(
                                    nested_leaf(
                                        output["highdm"]["sr_components"],
                                        (
                                            group,
                                            label,
                                            variation_name,
                                        ),
                                        len(bh.selected_an17_recoil60_labels()),
                                    ),
                                    high_sr_index,
                                    weights,
                                    group_mask,
                                )

                        low_index = low_indices[region]
                        low_mask = low_index >= 0
                        low_values = bh.finite_array(
                            sub[LOW_VALUE_BRANCH[region]], inputs["n"], 0.0
                        )
                        low_group_masks = {
                            "Nb1": low_mask
                            & (low_index < 16),
                            "Nb2plus": low_mask
                            & (low_index >= 16),
                        }
                        for group, group_mask in low_group_masks.items():
                            fill_leaf(
                                nested_leaf(
                                    output["lowdm"]["recoil"],
                                    (
                                        region,
                                        group,
                                        label,
                                        variation_name,
                                    ),
                                    len(LOW_EDGES) - 1,
                                ),
                                low_values,
                                weights,
                                group_mask,
                                LOW_EDGES,
                            )
                            fill_index_leaf(
                                nested_leaf(
                                    output["lowdm"]["search_components"],
                                    (
                                        region,
                                        group,
                                        label,
                                        variation_name,
                                    ),
                                    len(bh.LOWDM_34BIN_LABELS),
                                ),
                                low_index,
                                weights,
                                group_mask,
                            )
    return output


def merge_leaf(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("sumw", "sumw2", "entries"):
        target[key] = (
            np.asarray(target[key])
            + np.asarray(source[key])
        ).tolist()


def merge_tree(target: dict[str, Any], source: dict[str, Any]) -> None:
    if set(source) >= {"sumw", "sumw2", "entries"}:
        merge_leaf(target, source)
        return
    for key, value in source.items():
        if isinstance(value, dict):
            if set(value) >= {"sumw", "sumw2", "entries"}:
                if key not in target:
                    target[key] = empty_leaf(len(value["sumw"]))
                merge_leaf(target[key], value)
            else:
                merge_tree(target.setdefault(key, {}), value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--step-size", type=int, default=100000)
    parser.add_argument("--dy-ptll-policy", default="ptll100_200")
    parser.add_argument("--include-data", action="store_true")
    parser.add_argument("--nominal-only", action="store_true")
    parser.add_argument("--lowdm-llcr-nb1-only", action="store_true")
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=REGIONS,
        default=list(REGIONS),
    )
    parser.add_argument("--exclude-root", action="append", default=[])
    parser.add_argument(
        "--analysis-sf-components",
        nargs="+",
        choices=bh.REQUIRED_ANALYSIS_SF_COMPONENTS,
        default=list(bh.REQUIRED_ANALYSIS_SF_COMPONENTS),
        help=(
            "Analysis-specific SF components to apply and retain as "
            "Up/Down variations."
        ),
    )
    args = parser.parse_args()

    roots = [
        line.strip()
        for line in args.input_list.read_text().splitlines()
        if line.strip() and line.strip() not in set(args.exclude_root)
    ]
    if len(set(roots)) != len(roots):
        raise SystemExit("duplicate ROOT paths in input list")
    merged: dict[str, Any] = {
        "schema_version": "nb_recoil_transfer_inputs_2024_v1",
        "status": "running",
        "highdm": {
            "recoil_edges": HIGH_EDGES.tolist(),
            "nb_groups": list(HIGH_GROUPS),
            "recoil": {},
            "sr_components": {},
        },
        "lowdm": {
            "recoil_edges": LOW_EDGES.tolist(),
            "nb_groups": list(LOW_GROUPS),
            "search_bin_labels": bh.LOWDM_34BIN_LABELS,
            "recoil": {},
            "search_components": {},
        },
        "summary": {
            "input_roots": len(roots),
            "completed_roots": 0,
            "missing_roots": [],
            "events_scanned": 0,
            "events_weighted": 0,
            "datasets": {},
        },
        "provenance": {
            "normalization_sha256": file_sha256(args.normalization),
            "dy_ptll_policy": args.dy_ptll_policy,
            "jobs": args.jobs,
            "step_size": args.step_size,
            "include_data": args.include_data,
            "nominal_only": args.nominal_only,
            "lowdm_llcr_nb1_only": args.lowdm_llcr_nb1_only,
            "analysis_sf_components": list(args.analysis_sf_components),
            "regions": list(args.regions),
            "excluded_roots": list(args.exclude_root),
        },
    }
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, args.jobs)
    ) as executor:
        futures = {
            executor.submit(
                process_root,
                root,
                str(args.repo),
                str(args.normalization),
                args.dy_ptll_policy,
                args.step_size,
                args.include_data,
                args.nominal_only,
                args.lowdm_llcr_nb1_only,
                tuple(args.analysis_sf_components),
                tuple(args.regions),
            ): root
            for root in roots
        }
        for future in concurrent.futures.as_completed(futures):
            root = futures[future]
            result = future.result()
            summary = result["summary"]
            if summary.get("missing"):
                merged["summary"]["missing_roots"].append(root)
            else:
                merged["summary"]["completed_roots"] += 1
            merged["summary"]["events_scanned"] += int(
                summary.get("events_scanned", 0)
            )
            merged["summary"]["events_weighted"] += int(
                summary.get("events_weighted", 0)
            )
            for dataset, count in (summary.get("datasets") or {}).items():
                merged["summary"]["datasets"][dataset] = (
                    int(merged["summary"]["datasets"].get(dataset, 0))
                    + int(count)
                )
            merge_tree(merged["highdm"]["recoil"], result["highdm"]["recoil"])
            merge_tree(
                merged["highdm"]["sr_components"],
                result["highdm"]["sr_components"],
            )
            merge_tree(merged["lowdm"]["recoil"], result["lowdm"]["recoil"])
            merge_tree(
                merged["lowdm"]["search_components"],
                result["lowdm"]["search_components"],
            )
            if merged["summary"]["completed_roots"] % 25 == 0:
                print(
                    json.dumps(
                        {
                            "completed_roots": merged["summary"][
                                "completed_roots"
                            ],
                            "total_roots": len(roots),
                            "events_weighted": merged["summary"][
                                "events_weighted"
                            ],
                        }
                    ),
                    flush=True,
                )
    merged["status"] = (
        "complete"
        if not merged["summary"]["missing_roots"]
        else "complete_with_missing_roots"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, separators=(",", ":")))
    print(
        json.dumps(
            {
                "status": merged["status"],
                "output": str(args.output),
                "completed_roots": merged["summary"]["completed_roots"],
                "events_weighted": merged["summary"]["events_weighted"],
            }
        )
    )
    return 0 if merged["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
