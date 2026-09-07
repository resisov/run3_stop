#!/usr/bin/env python3
"""Build one audited diagonal-v3 Low-dM control-region NN-out partial."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

try:
    from . import region_io as base
    from ..data import split_buckets_2_1_7
    from .diagonal_v3_region_features import feature_arrays
    from .rank005_numpy import Rank005Numpy
except ImportError:  # EOS worker payload execution.
    import region_io as base  # type: ignore[no-redef]
    from data import split_buckets_2_1_7  # type: ignore[no-redef]
    from diagonal_v3_region_features import feature_arrays  # type: ignore[no-redef]
    from rank005_numpy import Rank005Numpy  # type: ignore[no-redef]


SCHEMA = "gnn_lowdm_diagonal_v3_srcr_nnout_partial_v3"
CR_REGIONS = ("LLCR", "QCDCR", "GCR", "DY2E", "DY2M")
ALL_REGIONS = ("SR", *CR_REGIONS)
SR_CATEGORIES = (
    "Nb1_NISR0", "Nb1_NISR1", "Nb1_NISR2plus",
    "Nb2plus_NISR0", "Nb2plus_NISR1", "Nb2plus_NISR2plus",
)
TEST_FRACTION = 0.70
EXTRA_BRANCHES = (
    "jet_corrected_mass",
    "n_lowdm_isr",
    "lowdm_isr_pt",
    "lowdm_isr_eta",
    "lowdm_isr_phi",
    "lowdm_isr_dphi",
    "lowdm_isr_subjet_btag_max",
    "lowdm_met_sqrt_ht",
    "min_dphi4",
)
PHYSICS_VARIABLES = ("recoil", "njet", "nb", "ht")


def category_masks(block: base.RegionBlock, region: str) -> dict[str, np.ndarray]:
    if region != "SR":
        return base.category_masks(block)
    return {
        "Nb1_NISR0": (block.nb == 1) & (block.nisr == 0),
        "Nb1_NISR1": (block.nb == 1) & (block.nisr == 1),
        "Nb1_NISR2plus": (block.nb == 1) & (block.nisr >= 2),
        "Nb2plus_NISR0": (block.nb >= 2) & (block.nisr == 0),
        "Nb2plus_NISR1": (block.nb >= 2) & (block.nisr == 1),
        "Nb2plus_NISR2plus": (block.nb >= 2) & (block.nisr >= 2),
    }


def sr_score_edges(configuration: dict[str, Any]) -> dict[str, np.ndarray]:
    definition = configuration["sr_binning"]
    if definition["category_labels"] != list(SR_CATEGORIES) or definition["total_bins"] != 30:
        raise RuntimeError("SR configuration is not the adopted GNN30")
    output = {
        label: np.asarray(definition["edges_by_category"][label], dtype=float)
        for label in SR_CATEGORIES
    }
    for label, edges in output.items():
        if (len(edges) != 6 or edges[0] != 0.0 or edges[-1] != 1.0
                or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0)):
            raise RuntimeError(label + ": invalid frozen five-bin score edges")
    return output


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def campaign_year(manifest: dict[str, Any], audits: list[dict[str, Any]]) -> int:
    years = {int(item["trota_provenance"]["application_year"]) for item in audits}
    if manifest.get("year") is not None:
        years.add(int(manifest["year"]))
    if len(years) != 1 or not years <= {2024, 2025}:
        raise RuntimeError("missing or inconsistent campaign year: " + str(sorted(years)))
    return years.pop()


def counter(nbin: int) -> dict[str, list[float]]:
    return {
        "sumw": [0.0] * nbin,
        "sumw2": [0.0] * nbin,
        "entries": [0.0] * nbin,
    }


def fill(
    payload: dict[str, Any],
    variation: str,
    region: str,
    category: str,
    sample: str,
    score: np.ndarray,
    weight: np.ndarray,
    selected: np.ndarray,
    edges: np.ndarray,
) -> None:
    nbin = len(edges) - 1
    target = (
        payload.setdefault(variation, {})
        .setdefault(region, {})
        .setdefault(category, {})
        .setdefault(sample, {})
        .setdefault("gnn_score", counter(nbin))
    )
    clipped = np.clip(score[selected], edges[0] + 1.0e-8, edges[-1] - 1.0e-8)
    local_weight = weight[selected]
    for name, values in (
        ("sumw", np.histogram(clipped, edges, weights=local_weight)[0]),
        ("sumw2", np.histogram(clipped, edges, weights=np.square(local_weight))[0]),
        ("entries", np.histogram(clipped, edges)[0]),
    ):
        target[name] = (
            np.asarray(target[name], dtype=float) + np.asarray(values, dtype=float)
        ).tolist()


def fill_physics_variable(
    payload: dict[str, Any],
    variation: str,
    region: str,
    category: str,
    sample: str,
    variable: str,
    values: np.ndarray,
    weight: np.ndarray,
    selected: np.ndarray,
) -> None:
    """Fill a selected CR observable alongside the GNN template."""
    target = (
        payload.setdefault(variation, {})
        .setdefault(region, {})
        .setdefault(category, {})
        .setdefault(sample, {})
        .setdefault(
            variable,
            base.empty_counter(base.HISTOGRAMS[variable]["edges"]),
        )
    )
    base.fill_counter(
        target,
        values,
        weight,
        selected,
        base.HISTOGRAMS[variable],
    )


def fill_score_ut(
    payload: dict[str, Any], variation: str, region: str, category: str,
    sample: str, score: np.ndarray, recoil: np.ndarray, weight: np.ndarray,
    selected: np.ndarray, score_edges: np.ndarray, ut_edges: np.ndarray,
) -> None:
    values = np.asarray(recoil[selected], dtype=float)
    selected_scores = np.asarray(score[selected], dtype=float)
    selected_weights = np.asarray(weight[selected], dtype=float)
    if (not np.all(np.isfinite(values)) or not np.all(np.isfinite(selected_scores))
            or not np.all(np.isfinite(selected_weights))):
        raise RuntimeError("non-finite GNN-score x U_T input")
    if np.any(values < ut_edges[0]):
        raise RuntimeError("selected event is below the U_T histogram domain")
    shape = (len(score_edges) - 1, len(ut_edges) - 1)
    target = (payload.setdefault(variation, {}).setdefault(region, {})
              .setdefault(category, {}).setdefault(sample, {})
              .setdefault("gnn_score_ut", {
                  name: np.zeros(shape).tolist() for name in ("sumw", "sumw2", "entries")
              }))
    selected_scores = np.clip(selected_scores, score_edges[0] + 1.0e-8,
                              score_edges[-1] - 1.0e-8)
    values = np.minimum(values, np.nextafter(ut_edges[-1], -np.inf))
    for name, weights in (("sumw", selected_weights),
                          ("sumw2", np.square(selected_weights)), ("entries", None)):
        counts = np.histogram2d(selected_scores, values, (score_edges, ut_edges),
                                weights=weights)[0]
        target[name] = (np.asarray(target[name]) + counts).tolist()


def validate_score_ut(payload: dict[str, Any]) -> None:
    for by_region in payload.values():
        for by_category in by_region.values():
            for by_sample in by_category.values():
                for record in by_sample.values():
                    joint = record.get("gnn_score_ut")
                    if joint is None:
                        raise RuntimeError("GNN-score x U_T histogram is missing")
                    for field in ("sumw", "sumw2", "entries"):
                        values = np.asarray(joint[field], dtype=float)
                        score = np.asarray(record["gnn_score"][field], dtype=float)
                        if (values.ndim != 2 or values.shape[0] != len(score)
                                or not np.all(np.isfinite(values))
                                or not np.allclose(values.sum(axis=1), score,
                                                   rtol=1e-10, atol=1e-8)):
                            raise RuntimeError("GNN-score x U_T projection differs from GNN template")


def diagonal_region_masks(
    blocks: dict[str, base.RegionBlock], nres: np.ndarray
) -> dict[str, np.ndarray]:
    """Frozen diagonal selection: no MET/sqrt(HT), NISR, or High-dM veto."""
    return {
        region: (
            block.core
            & (block.nb >= 1)
            & (block.nt == 0)
            & (block.nw == 0)
            & (nres == 0)
        )
        for region, block in blocks.items()
    }


def merge_feature_audit(
    target: dict[str, dict[str, int]], region: str, source: dict[str, int]
) -> None:
    current = target.setdefault(region, {})
    for key, value in source.items():
        current[key] = int(current.get(key, 0)) + int(value)


def process_source(
    record: dict[str, Any],
    manifest: dict[str, Any],
    signal_norm: dict[tuple[str, int, int], float],
    stop_xsec: dict[int, float],
    repository: Path,
    model: Rank005Numpy,
    edges: np.ndarray,
    regions: tuple[str, ...],
    sr_test_only: bool,
    sr_edges: dict[str, np.ndarray] | None = None,
    raw_dy: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if "SR" in regions and sr_edges is None:
        raise RuntimeError("SR evaluation requires the frozen GNN30 configuration")
    root_path = str(record["root"])
    sidecar_path = str(record["sidecar"])
    sidecar = json.loads(Path(sidecar_path).read_text())
    trota_provenance = base.validate_trota_provenance(sidecar)
    with uproot.open(root_path) as root_file:
        from build_flat_boosted_recoil_hists import WEIGHT_BRANCHES, BACKGROUND_ESTIMATION_UT_BINS

        ut_edges = np.asarray(BACKGROUND_ESTIMATION_UT_BINS, dtype=float)

        tree = root_file["Events"]
        read_branches = tuple(
            dict.fromkeys((*base.SELECTION_BRANCHES, *EXTRA_BRANCHES, *WEIGHT_BRANCHES))
        )
        missing = sorted(set(read_branches) - set(tree.keys()))
        optional = {"electron_veto_eta", "electron_medium_eta"}
        hard_missing = [name for name in missing if name not in optional]
        if hard_missing:
            raise RuntimeError("missing flat branches: " + ", ".join(hard_missing))
        arrays = tree.arrays(
            [name for name in read_branches if name in tree.keys()], library="ak"
        )
        blocks, reconstruction_audit = base.build_region_blocks(arrays)
        eligible = np.zeros(len(arrays), dtype=bool)
        for region in regions:
            block = blocks[region]
            eligible |= (
                block.core
                & (block.nb >= 1)
                & (block.nt == 0)
                & (block.nw == 0)
            )
        nres, nres_audit = base.compute_nres(
            arrays, root_file["TROTA"], eligible
        )

    all_masks = diagonal_region_masks(blocks, nres)
    region_masks = {region: all_masks[region].copy() for region in regions}
    split_audit: dict[str, Any] = {"policy": "all_events"}
    if sr_test_only:
        split = split_buckets_2_1_7(
            np.asarray(arrays["physical_dataset_id"], dtype=np.int64),
            np.asarray(arrays["run"], dtype=np.int64),
            np.asarray(arrays["luminosityBlock"], dtype=np.int64),
            np.asarray(arrays["event"], dtype=np.int64),
        )
        before = int(np.count_nonzero(region_masks["SR"]))
        region_masks["SR"] &= split == 2
        split_audit = {
            "policy": "deterministic_2_1_7_test_only",
            "test_fraction": TEST_FRACTION,
            "sr_before_test_split": before,
            "sr_after_test_split": int(np.count_nonzero(region_masks["SR"])),
        }
    union = np.logical_or.reduce(list(region_masks.values()))
    histograms: dict[str, Any] = {}
    audit: dict[str, Any] = {
        "root": root_path,
        "sidecar": sidecar_path,
        "events": len(arrays),
        "selected_union_events": int(np.count_nonzero(union)),
        "selected_by_region": {
            region: int(np.count_nonzero(mask))
            for region, mask in region_masks.items()
        },
        "trota": nres_audit,
        "trota_provenance": trota_provenance,
        "reconstruction": reconstruction_audit,
        "feature_reconstruction": {},
        "data_stream_exclusions": {},
        "blinded_sr_data_events": 0,
        "split": split_audit,
        "weight_status": {},
    }
    if not np.any(union):
        return histograms, audit

    scores = {
        region: np.full(len(arrays), np.nan, dtype=np.float32)
        for region in regions
    }
    for region, selected in region_masks.items():
        if not np.any(selected):
            continue
        inputs = feature_arrays(arrays, blocks[region], region, selected)
        scores[region][selected] = model.predict(*inputs[:5], batch_size=256)
        merge_feature_audit(audit["feature_reconstruction"], region, inputs[5])

    dataset_ids = np.asarray(arrays["dataset_id"], dtype=np.int64)
    for dataset_id in sorted(set(int(value) for value in dataset_ids[union])):
        sidecar_dataset = base.dataset_record(sidecar, dataset_id)
        dataset = str(sidecar_dataset.get("dataset") or "unknown")
        raw_process = str(sidecar_dataset.get("process") or "unknown")
        process = base.canonical_process(raw_process, dataset)
        is_signal = bool(sidecar_dataset.get("is_signal"))
        if is_signal and "SR" not in regions:
            continue
        for group_mask in base.subgroup_masks(arrays, dataset_id, is_signal):
            active = group_mask & union
            if not np.any(active):
                continue
            sub_group = arrays[active]
            local_masks = {
                region: mask[active] for region, mask in region_masks.items()
            }
            local_all_masks = {
                region: mask[active] for region, mask in all_masks.items()
            }
            weight_variations, status = base.normalized_weight_variations(
                sub_group,
                sidecar_dataset,
                raw_process,
                dataset,
                manifest,
                signal_norm,
                stop_xsec,
                repository,
                local_all_masks,
            )
            sample = base.sample_name(sub_group, sidecar_dataset, process)
            audit["weight_status"].setdefault(sample, status)
            for region, local_region in local_masks.items():
                if is_signal and region != "SR":
                    continue
                if sidecar_dataset.get("is_data") and region == "SR":
                    audit["blinded_sr_data_events"] += int(
                        np.count_nonzero(local_region)
                    )
                    continue
                if sidecar_dataset.get("is_data") and process != base.DATA_STREAM[region]:
                    audit["data_stream_exclusions"][region] = int(
                        audit["data_stream_exclusions"].get(region, 0)
                    ) + int(np.count_nonzero(local_region))
                    continue
                if not np.any(local_region):
                    continue
                local_block = base.RegionBlock(
                    **{
                        field: getattr(blocks[region], field)[active]
                        for field in base.RegionBlock.__dataclass_fields__
                    }
                )
                local_score = scores[region][active]
                if np.any(~np.isfinite(local_score[local_region])):
                    raise RuntimeError(region + ": selected events have no score")
                physics_values = base.histogram_values(local_block)
                for category, category_mask in category_masks(local_block, region).items():
                    selected_category = local_region & category_mask
                    if not np.any(selected_category):
                        continue
                    local_edges = sr_edges[category] if region == "SR" else edges
                    for variation, weights in weight_variations.items():
                        template_weights = (
                            weights / TEST_FRACTION
                            if region == "SR" and sr_test_only
                            else weights
                        )
                        fill(
                            histograms,
                            variation,
                            region,
                            category,
                            sample,
                            local_score,
                            template_weights,
                            selected_category,
                            local_edges,
                        )
                        fill_score_ut(histograms, variation, region, category, sample,
                                      local_score, local_block.recoil, template_weights,
                                      selected_category, local_edges, ut_edges)
                        for variable in PHYSICS_VARIABLES:
                            fill_physics_variable(
                                histograms,
                                variation,
                                region,
                                category,
                                sample,
                                variable,
                                physics_values[variable],
                                template_weights,
                                selected_category,
                            )
                    if not raw_dy and region in base.RZ_FACTORS and sample == "DY":
                        nb_key = "Nb1" if category.startswith("Nb1_") else "Nb2plus"
                        for variation, weights in weight_variations.items():
                            rz_variation = f"{variation}_rz"
                            rz_weights = weights * base.RZ_FACTORS[region][nb_key]
                            fill(
                                histograms,
                                rz_variation,
                                region,
                                category,
                                sample,
                                local_score,
                                rz_weights,
                                selected_category,
                                local_edges,
                            )
                            fill_score_ut(histograms, rz_variation, region, category,
                                          sample, local_score, local_block.recoil,
                                          rz_weights, selected_category, local_edges, ut_edges)
                            for variable in PHYSICS_VARIABLES:
                                fill_physics_variable(
                                    histograms,
                                    rz_variation,
                                    region,
                                    category,
                                    sample,
                                    variable,
                                    physics_values[variable],
                                    rz_weights,
                                    selected_category,
                                )
    validate_score_ut(histograms)
    return histograms, audit


def merge(target: dict[str, Any], source: dict[str, Any], nbin: int) -> None:
    for variation, by_region in source.items():
        for region, by_category in by_region.items():
            for category, by_sample in by_category.items():
                for sample, by_variable in by_sample.items():
                    for variable, incoming in by_variable.items():
                        variable_nbin = len(incoming["sumw"])
                        if variable == "gnn_score" and variable_nbin != nbin:
                            raise RuntimeError("inconsistent GNN histogram size")
                        current = (
                            target.setdefault(variation, {})
                            .setdefault(region, {})
                            .setdefault(category, {})
                            .setdefault(sample, {})
                            .setdefault(variable, {
                                name: np.zeros_like(np.asarray(incoming[name], dtype=float)).tolist()
                                for name in ("sumw", "sumw2", "entries")
                            })
                        )
                        for name in ("sumw", "sumw2", "entries"):
                            if np.shape(current[name]) != np.shape(incoming[name]):
                                raise RuntimeError("inconsistent histogram shape: " + variable)
                            current[name] = (
                                np.asarray(current[name], dtype=float)
                                + np.asarray(incoming[name], dtype=float)
                            ).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--configuration", type=Path,
                        help="Frozen config.json; required when evaluating SR.")
    parser.add_argument("--raw-dy", action="store_true",
                        help="Do not write legacy RZ-scaled DY plotting variations.")
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=ALL_REGIONS,
        default=list(CR_REGIONS),
    )
    opts = parser.parse_args()
    regions = tuple(dict.fromkeys(opts.regions))
    request = json.loads(opts.request.read_text())
    selection = json.loads(opts.selection.read_text())
    configuration = json.loads(opts.configuration.read_text()) if opts.configuration else None
    edge_values = selection.get("score_edges")
    if edge_values is None and configuration is not None:
        edge_values = configuration["cr_binning"]["score_edges"]
    if edge_values is None:
        parser.error("score edges require --configuration or selection.score_edges")
    edges = np.asarray(edge_values, dtype=np.float64)
    if len(edges) != 6 or not np.all(np.diff(edges) > 0.0):
        raise RuntimeError("frozen diagonal-v3 score edges are invalid")
    sr_edges = None
    if configuration is not None:
        sr_edges = sr_score_edges(configuration)
        if configuration["cr_binning"]["score_edges"] != edges.tolist():
            raise RuntimeError("selection/CR configuration edges differ")
    if "SR" in regions and sr_edges is None:
        parser.error("--regions SR requires --configuration")
    manifest = json.loads(Path(request["manifest"]).read_text())
    from build_flat_boosted_recoil_hists import BACKGROUND_ESTIMATION_UT_BINS

    xsec_payload = json.loads(Path(request["stop_xsec"]).read_text())
    signal_norm = base.signal_normalization_map(manifest)
    stop_xsec = base.stop_xsec_map(xsec_payload)
    model = Rank005Numpy(opts.model)
    histograms: dict[str, Any] = {}
    audits: list[dict[str, Any]] = []
    valid: list[str] = []
    bad: list[dict[str, Any]] = []
    started = time.time()
    repository = Path(
        os.environ.get("LOWDM_TEST_REPOSITORY_OVERRIDE") or request["repository"]
    )
    for record in request["inputs"]:
        try:
            local, audit = process_source(
                record,
                manifest,
                signal_norm,
                stop_xsec,
                repository,
                model,
                edges,
                regions,
                False,
                sr_edges,
                opts.raw_dy,
            )
            merge(histograms, local, len(edges) - 1)
            audits.append(audit)
            valid.append(str(record["root"]))
        except Exception as error:
            bad.append(
                {
                    "file": str(record["root"]),
                    "failure_stage": "diagonal_v3_cr_nnout_source",
                    "exception_type": type(error).__name__,
                    "error": str(error)[:1000],
                    "traceback": traceback.format_exc(limit=20)[:8000],
                }
            )
    payload = {
        "schema_version": SCHEMA,
        "status": "complete" if not bad else "complete_with_bad_files",
        "year": campaign_year(manifest, audits),
        "kind": request["kind"],
        "batch": int(request["batch"]),
        "selection_contract": {
            "common": (
                "region core && Nb>=1 && Nt=0 && Nw=0 && Nres(TROTA)=0"
            ),
            "explicitly_not_applied": [
                "MET/sqrt(HT)>=10",
                "deltaPhi(ISR,recoil)>2",
                "NISR==1",
                "!feature_SR",
            ],
            "categories": {"SR": list(SR_CATEGORIES), "CR": list(base.CATEGORIES)}
                          if "SR" in regions else list(base.CATEGORIES),
            "data_streams": base.DATA_STREAM,
            "rz": {} if opts.raw_dy else base.RZ_FACTORS,
            "dy_mass_window_gev": [71.0, 111.0],
            "lepton_veto_pt_min_gev": {"electron": 10.0, "muon": 10.0},
            "lepton_veto_pt_comparison": ">",
            "analysis_sf_components": ["met_trigger", "photon_trigger"],
            "sr_data": "blinded",
        },
        "regions": list(regions),
        "partition": "all selected events; SR data blinded",
        "score_edges": {"CR": edges.tolist(),
                        "SR": {key: value.tolist() for key, value in sr_edges.items()}}
                       if "SR" in regions else edges.tolist(),
        "checkpoint": {
            "trial": selection["best_trial"],
            "epoch": int(selection["best_epoch"]),
            "model_sha256": hashlib.sha256(opts.model.read_bytes()).hexdigest(),
            "selection_sha256": hashlib.sha256(opts.selection.read_bytes()).hexdigest(),
            "configuration_sha256": hashlib.sha256(opts.configuration.read_bytes()).hexdigest()
                                    if opts.configuration else None,
        },
        "input_files_requested": len(request["inputs"]),
        "input_files_valid": len(valid),
        "input_files": valid,
        "histograms": histograms,
        "histogram_specs": {
            variable: base.HISTOGRAMS[variable]
            for variable in PHYSICS_VARIABLES
        },
        "source_audits": audits,
        "bad_files": bad,
        "runtime_seconds": time.time() - started,
    }
    payload["histogram_specs"]["gnn_score_ut"] = {
        "axes": ["gnn_score", "U_T"],
        "score_edges": payload["score_edges"],
        "ut_edges": list(BACKGROUND_ESTIMATION_UT_BINS),
        "ut_last_bin_open_ended": True,
        "process_separated": True,
    }
    validate_score_ut(histograms)
    write_json(opts.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "valid": len(valid),
                "bad": len(bad),
                "output": str(opts.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
