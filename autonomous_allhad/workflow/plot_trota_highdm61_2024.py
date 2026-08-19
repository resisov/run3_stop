#!/usr/bin/env python3
"""Combine the adopted 55-bin High-dM baseline with the TROTA Nres split.

The canonical histogram is read with bounded-memory mmap extraction.  The
first six bins are replaced by twelve full-weight Nres=0/Nres>=1 bins; the
remaining 49 adopted bins are copied without reinterpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
from typing import Any

import numpy as np

import plot_control_search_bins_style as plot


CANONICAL_SCHEME = "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR"
COMPONENT_SCHEME = "trota_nres_split12_nb1plus_nt0_nw0_SR"
OUTPUT_SCHEME = "highdm61_adopted55_trota_nres_split_SR"
CANONICAL_SHA256 = "2ad64b5236a23c03fbe0c21ec7be7e1a51cb035bf9868e73ef959e7510c178df"
COMPONENT_BTAG_SHA256 = "5a96f6b7dcd806a10c64dab7ecefd18a13767fa4645bc003bef5716798246563"
SAMPLES = (
    "data_obs",
    "QCD",
    "Zto2Nu",
    "WtoLNu",
    "ST",
    "TT",
    "DY",
    "GJ",
    "VV",
    "T2tt_mStop1000_mLSP1",
    "T2tt_mStop1200_mLSP1",
)
HIGH_MERGE_PAIRS = ((22, 23), (34, 35), (40, 41), (52, 53), (58, 59))
BASE_CATEGORY_KEYS = (
    "Nb1plus_T0_W0",
    "Nb1plus_T0_W1plus",
    "Nb1_T1plus_W0",
    "Nb1_T1plus_W1plus",
    "Nb2_T1_W0",
    "Nb2_T1_W1",
    "Nb2_Nt2plus_W0",
    "Nb3plus_T1_W0",
    "Nb3plus_T1_W1",
    "Nb3plus_T2_W0",
)
OUTPUT_LAYOUT = (
    ("Nb1plus_T0_W0_Nres0", 6),
    ("Nb1plus_T0_W0_Nres1plus", 6),
    ("Nb1plus_T0_W1plus", 6),
    ("Nb1_T1plus_W0", 6),
    ("Nb1_T1plus_W1plus", 5),
    ("Nb2_T1_W0", 6),
    ("Nb2_T1_W1", 5),
    ("Nb2_Nt2plus_W0", 5),
    ("Nb3plus_T1_W0", 6),
    ("Nb3plus_T1_W1", 5),
    ("Nb3plus_T2_W0", 5),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def drop_file_cache(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def json_value_end(buffer: mmap.mmap, start: int) -> int:
    opening = buffer[start]
    if opening not in (ord("{"), ord("[")):
        raise ValueError(f"expected JSON object/array at byte {start}")
    closing = ord("}") if opening == ord("{") else ord("]")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(buffer)):
        value = buffer[index]
        if in_string:
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                in_string = False
            continue
        if value == ord('"'):
            in_string = True
        elif value == opening:
            depth += 1
        elif value == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError(f"unterminated JSON value at byte {start}")


def extract_search_sample(path: Path, scheme: str, sample: str) -> dict[str, Any]:
    return extract_search_samples(path, scheme, (sample,))[sample]


def extract_search_samples(
    path: Path, scheme: str, samples: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as payload:
            section = payload.find(b'"search_bin_histograms":')
            if section < 0:
                raise ValueError(f"search_bin_histograms absent from {path}")
            scheme_marker = json.dumps(scheme).encode() + b":"
            scheme_start = payload.find(scheme_marker, section)
            if scheme_start < 0:
                raise ValueError(f"scheme {scheme} absent from {path}")
            scheme_value_start = scheme_start + len(scheme_marker)
            while payload[scheme_value_start] in b" \t\r\n":
                scheme_value_start += 1
            scheme_end = json_value_end(payload, scheme_value_start)
            for sample in samples:
                sample_marker = json.dumps(sample).encode() + b":"
                sample_start = payload.find(sample_marker, scheme_value_start, scheme_end)
                if sample_start < 0:
                    raise ValueError(f"sample {sample} absent from {scheme}")
                value_start = sample_start + len(sample_marker)
                while payload[value_start] in b" \t\r\n":
                    value_start += 1
                value_end = json_value_end(payload, value_start)
                result[sample] = json.loads(payload[value_start:value_end])
    drop_file_cache(path)
    return result


def adopted55_mapping() -> list[list[int]]:
    merge = dict(HIGH_MERGE_PAIRS)
    seconds = {second for _, second in HIGH_MERGE_PAIRS}
    result = []
    for index in range(60):
        if index in seconds:
            continue
        result.append([index, merge[index]] if index in merge else [index])
    if len(result) != 55 or sorted(sum(result, [])) != list(range(60)):
        raise AssertionError("invalid adopted High-dM 55-bin mapping")
    return result


def combine_bins(values: np.ndarray, mapping: list[list[int]]) -> np.ndarray:
    return np.asarray([float(np.sum(values[group])) for group in mapping], dtype=float)


def combine_record(record: dict[str, Any], mapping: list[list[int]]) -> dict[str, Any]:
    output = {}
    for field, dtype in (("sumw", float), ("sumw2", float), ("entries", int)):
        values = np.asarray(record[field], dtype=dtype)
        if len(values) != 60:
            raise ValueError(f"canonical {field} length {len(values)} != 60")
        combined = combine_bins(values, mapping)
        output[field] = combined.astype(dtype).tolist()
    return output


def merge_sample(
    sample: str,
    canonical: dict[str, Any],
    component: dict[str, Any],
    mapping: list[list[int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical_variations = set(canonical)
    component_variations = set(component)
    if canonical_variations != component_variations:
        raise RuntimeError(
            f"variation mismatch for {sample}: canonical-only="
            f"{sorted(canonical_variations - component_variations)}, component-only="
            f"{sorted(component_variations - canonical_variations)}"
        )
    output: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    for variation in sorted(canonical_variations):
        baseline = canonical[variation]
        split = component[variation]
        adopted = combine_record(baseline, mapping)
        for field, dtype in (("sumw", float), ("sumw2", float), ("entries", int)):
            split_values = np.asarray(split[field], dtype=dtype)
            if len(split_values) != 12:
                raise ValueError(f"component {sample}/{variation}/{field} length != 12")
            baseline_values = np.asarray(baseline[field], dtype=dtype)[:6]
            recombined = split_values[:6] + split_values[6:]
            if field == "entries":
                matched = np.array_equal(recombined, baseline_values)
            else:
                matched = np.allclose(
                    recombined, baseline_values, rtol=2.0e-10, atol=2.0e-8
                )
            if not matched:
                raise RuntimeError(
                    f"split does not reproduce canonical category-1 for "
                    f"{sample}/{variation}/{field}; max_abs="
                    f"{float(np.max(np.abs(recombined - baseline_values)))}"
                )
            merged = np.concatenate([split_values, np.asarray(adopted[field], dtype=dtype)[6:]])
            if len(merged) != 61:
                raise AssertionError(f"merged {field} length {len(merged)} != 61")
            output.setdefault(variation, {})[field] = merged.astype(dtype).tolist()
            audit.setdefault(variation, {})[field] = {
                "matched": True,
                "max_abs_difference": float(np.max(np.abs(recombined - baseline_values))),
            }
    return output, audit


def zero_component_like(canonical: dict[str, Any]) -> dict[str, Any]:
    for variation, record in canonical.items():
        for field in ("sumw", "sumw2", "entries"):
            values = np.asarray(record[field])[:6]
            if np.any(values != 0):
                raise RuntimeError(
                    f"cannot synthesize missing component: canonical "
                    f"{variation}/{field} first-category content is nonzero"
                )
    return {
        variation: {
            "sumw": [0.0] * 12,
            "sumw2": [0.0] * 12,
            "entries": [0] * 12,
        }
        for variation in canonical
    }


def recoil_labels(size: int) -> list[str]:
    if size == 6:
        return list(plot.RECOIL6_LABELS)
    if size == 5:
        return list(plot.RECOIL6_LABELS[:4]) + ["500-1500"]
    raise ValueError(f"unsupported recoil category size: {size}")


def block_label(key: str) -> str:
    if key == "Nb1plus_T0_W0_Nres0":
        return '$N_{b}\\geq1$, $N_{t}=0$\n$N_{W}=0$, $N_{\\mathrm{res}}=0$'
    if key == "Nb1plus_T0_W0_Nres1plus":
        return '$N_{b}\\geq1$, $N_{t}=0$\n$N_{W}=0$, $N_{\\mathrm{res}}\\geq1$'
    return plot.SELECTED_AN17_CATEGORY_LABELS.get(key, key)


def plot_payload(payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    record = plot.flat_search_record(
        payload, OUTPUT_SCHEME, "High-dM 61-bin TROTA study", allow_signal=True
    )
    if not record or int(record["nbin"]) != 61:
        raise RuntimeError("61-bin plotting record is missing or malformed")
    blocks = []
    offset = 0
    for index, (category, size) in enumerate(OUTPUT_LAYOUT):
        slc = slice(offset, offset + size)
        block = {
            "groups": {group: values[slc] for group, values in record["groups"].items()},
            "background": record["background"][slc],
            "background_unc": record["background_unc"][slc],
            "background_stat_unc": record["background_stat_unc"][slc],
            "background_systematic_totals": {
                source: {
                    direction: values[slc]
                    for direction, values in directions.items()
                }
                for source, directions in record["background_systematic_totals"].items()
            },
            "data": record["data"][slc],
            "data_unc": record["data_unc"][slc],
            "signals": {key: values[slc] for key, values in record["signals"].items()},
            "signal_specs": plot.SIGNAL_OVERLAYS,
            "label": block_label(category),
            "nbin": size,
            "blind_data": True,
            "label_box": True,
            "label_fontsize": 9.4,
            "label_box_pad": 0.16,
            "figure_width": 18.5,
        }
        if index == 0:
            block["annotation"] = (
                "Exploratory TROTA $N_{\\mathrm{res}}$ split (no dedicated TROTA SF)"
            )
        blocks.append(block)
        offset += size
    output_dir.mkdir(parents=True, exist_ok=True)
    result = plot.draw_flat_blocks(
        blocks,
        output_dir / "highdm_sr_trota_nres61_from_adopted55_bins",
        xlabel="Search bin",
        show_yields=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--component", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    canonical_sha256 = sha256(args.canonical)
    if canonical_sha256 != CANONICAL_SHA256:
        raise RuntimeError(
            f"canonical histogram checksum drift: {canonical_sha256} != {CANONICAL_SHA256}"
        )
    component_payload = json.loads(args.component.read_text())
    if component_payload.get("status") != "complete":
        raise RuntimeError("TROTA full-weight component is not complete")
    if int(component_payload.get("files_completed") or 0) != 5489:
        raise RuntimeError("TROTA component does not contain 5,489 completed inputs")
    if int((component_payload.get("totals") or {}).get("events") or 0) != 230_830_776:
        raise RuntimeError("TROTA component event total is not 230,830,776")
    if component_payload.get("failures"):
        raise RuntimeError("TROTA component contains failures")
    if component_payload.get("btag_efficiency_sha256") != COMPONENT_BTAG_SHA256:
        raise RuntimeError("TROTA component btag checksum mismatch")
    component = (
        (component_payload.get("search_bin_histograms") or {})
        .get(COMPONENT_SCHEME) or {}
    )

    mapping = adopted55_mapping()
    output_histograms: dict[str, Any] = {}
    split_audit: dict[str, Any] = {}
    canonical_samples = extract_search_samples(
        args.canonical, CANONICAL_SCHEME, SAMPLES
    )
    for sample in SAMPLES:
        canonical_sample = canonical_samples[sample]
        component_sample = component.get(sample)
        if not component_sample:
            component_sample = zero_component_like(canonical_sample)
        output_histograms[sample], split_audit[sample] = merge_sample(
            sample, canonical_sample, component_sample, mapping
        )

    labels = [
        f"{category}_recoil_{label}"
        for category, size in OUTPUT_LAYOUT
        for label in recoil_labels(size)
    ]
    payload = {
        "schema_version": "trota_highdm61_plot_payload_2024_v1",
        "status": "complete",
        "year": "2024",
        "canonical_histogram": str(args.canonical),
        "canonical_histogram_sha256": canonical_sha256,
        "component": str(args.component),
        "component_sha256": sha256(args.component),
        "physics_status": "exploratory; dedicated TROTA efficiency SF unavailable",
        "construction": (
            "adopted High-dM 55-bin baseline; replace unchanged first six recoil "
            "bins with Nres=0 and Nres>=1 copies, retain the other 49 bins"
        ),
        "adopted55_mapping": mapping,
        "split_recombination_audit": split_audit,
        "search_bin_schemes": {
            OUTPUT_SCHEME: {
                "bin_labels": labels,
                "category_sizes": [size for _, size in OUTPUT_LAYOUT],
                "category_layout": [[category, size] for category, size in OUTPUT_LAYOUT],
                "recoil_pt_bins": [250.0, 300.0, 350.0, 400.0, 500.0, 800.0, 1500.0],
                "bin_count": 61,
                "baseline_bin_count": 55,
                "trota_scale_factor": "not applied; unavailable",
            }
        },
        "search_bin_histograms": {OUTPUT_SCHEME: output_histograms},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact_path = args.output_dir / "highdm61_histograms_2024.json"
    write_json(compact_path, payload)
    plot_result = plot_payload(payload, args.output_dir)
    summary = {
        "status": "complete",
        "bin_count": 61,
        "baseline_bin_count": 55,
        "canonical_histogram_sha256": canonical_sha256,
        "component_sha256": sha256(args.component),
        "compact_histogram": str(compact_path),
        "compact_histogram_sha256": sha256(compact_path),
        "plot": plot_result,
        "validation": {
            "samples": len(output_histograms),
            "variations_per_mc_sample": len(output_histograms["TT"]),
            "split_recombines_to_canonical_first_six_bins": True,
            "unchanged_adopted_bins": 49,
            "finite_histogram_contents": all(
                np.all(np.isfinite(np.asarray(record[field], dtype=float)))
                for sample in output_histograms.values()
                for record in sample.values()
                for field in ("sumw", "sumw2")
            ),
        },
        "trota_scale_factor": "unavailable; plot is exploratory",
    }
    summary_path = args.output_dir / "highdm61_plot_summary_2024.json"
    write_json(summary_path, summary)
    print(json.dumps({
        "status": "complete",
        "bin_count": 61,
        "png": plot_result.get("png"),
        "pdf": plot_result.get("pdf"),
        "summary": str(summary_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
