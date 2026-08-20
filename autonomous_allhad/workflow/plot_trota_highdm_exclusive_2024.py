#!/usr/bin/env python3
"""Draw blinded 2024 High-dM comparisons for the TROTA Nres proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from plot_control_search_bins_style import (
    RECOIL6_LABELS,
    SELECTED_AN17_CATEGORY_LABELS,
    SELECTED_AN17_CATEGORY_ORDER,
    SIGNAL_OVERLAYS,
    draw_flat_blocks,
    flat_search_record,
)


SCHEMA_VERSION = "trota_highdm_exclusive_2024_plot_summary_v2"
BASELINE_SCHEME = "highdm55_mtb175_inclusive_nres_SR"
EXCLUSIVE85_SCHEME = "highdm85_mtb175_exclusive_nres_SR"
TAILMERGED80_SCHEME = "highdm80_mtb175_exclusive_nres_tailmerged_SR"
BASELINE_CATEGORY_SIZES = (6, 6, 6, 5, 6, 5, 5, 6, 5, 5)
NRES_TOPOLOGY_LAYOUT = (
    ("resolved1_only", "$N_{t}=0$, $N_{W}=0$\n$N_{res}=1$"),
    ("resolved2plus_only", "$N_{t}=0$, $N_{W}=0$\n$N_{res}\\geq2$"),
    ("top_resolved", "$N_{t}\\geq1$, $N_{W}=0$\n$N_{res}\\geq1$"),
    ("w_resolved", "$N_{t}=0$, $N_{W}\\geq1$\n$N_{res}\\geq1$"),
    ("top_w_resolved", "$N_{t}\\geq1$, $N_{W}\\geq1$\n$N_{res}\\geq1$"),
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


def slice_record(record: dict, start: int, size: int, label: str, xlabels: list[str]) -> dict:
    stop = start + size
    slc = slice(start, stop)
    if stop > int(record["nbin"]):
        raise ValueError(f"block {label} exceeds {record['nbin']} bins")
    systematics = {
        source: {
            direction: np.asarray(values, dtype=float)[slc]
            for direction, values in varied.items()
        }
        for source, varied in (record.get("background_systematic_totals") or {}).items()
    }
    return {
        "groups": {
            group: np.asarray(values, dtype=float)[slc]
            for group, values in record["groups"].items()
        },
        "background": np.asarray(record["background"], dtype=float)[slc],
        "background_unc": np.asarray(record["background_unc"], dtype=float)[slc],
        "background_stat_unc": np.asarray(
            record.get("background_stat_unc", record["background_unc"]), dtype=float
        )[slc],
        "background_systematic_totals": systematics,
        "data": np.asarray(record["data"], dtype=float)[slc],
        "data_unc": np.asarray(record["data_unc"], dtype=float)[slc],
        "signals": {
            key: np.asarray(values, dtype=float)[slc]
            for key, values in (record.get("signals") or {}).items()
        },
        "signal_specs": record.get("signal_specs") or SIGNAL_OVERLAYS,
        "label": label,
        "nbin": size,
        "xlabels": xlabels,
        "blind_data": True,
        "label_box": True,
        "label_fontsize": 9.0,
        "label_box_pad": 0.16,
        "figure_width": 24.0,
    }


def baseline_blocks(record: dict, *, nres_zero: bool) -> list[dict]:
    if sum(BASELINE_CATEGORY_SIZES) != int(record["nbin"]):
        raise ValueError("the adopted 55-bin category layout does not match the histogram")
    blocks = []
    offset = 0
    for category, size in zip(SELECTED_AN17_CATEGORY_ORDER, BASELINE_CATEGORY_SIZES):
        label = SELECTED_AN17_CATEGORY_LABELS[category]
        if nres_zero:
            label += "\n$N_{res}=0$"
        labels = RECOIL6_LABELS[:size]
        blocks.append(slice_record(record, offset, size, label, list(labels)))
        offset += size
    return blocks


def exclusive_blocks(record: dict, *, tail_merged: bool) -> list[dict]:
    expected = 80 if tail_merged else 85
    if int(record["nbin"]) != expected:
        raise ValueError(f"expected {expected} bins, found {record['nbin']}")
    baseline_record = {
        **record,
        "nbin": 55,
        "groups": {key: value[:55] for key, value in record["groups"].items()},
        "background": record["background"][:55],
        "background_unc": record["background_unc"][:55],
        "background_stat_unc": record["background_stat_unc"][:55],
        "background_systematic_totals": {
            source: {direction: value[:55] for direction, value in varied.items()}
            for source, varied in (record.get("background_systematic_totals") or {}).items()
        },
        "data": record["data"][:55],
        "data_unc": record["data_unc"][:55],
        "signals": {key: value[:55] for key, value in record["signals"].items()},
    }
    blocks = baseline_blocks(baseline_record, nres_zero=True)
    offset = 55
    size = 5 if tail_merged else 6
    recoil = list(RECOIL6_LABELS[:4]) + (["500-1500"] if tail_merged else list(RECOIL6_LABELS[4:]))
    for _key, label in NRES_TOPOLOGY_LAYOUT:
        blocks.append(slice_record(record, offset, size, label, recoil))
        offset += size
    if offset != expected:
        raise AssertionError("exclusive plot layout does not cover the scheme")
    return blocks


def ordered_exclusive79_blocks(record: dict) -> list[dict]:
    """Return the 85-bin contents in Nb/Nt/NW/Nres display order.

    The stored Nres>0 extension is inclusive in Nb, so those blocks are
    labelled explicitly as Nb>=1 and placed before the exact-Nb baseline
    blocks.  The statistically empty Nt>=1, NW>=1, Nres>=1 block is omitted
    from this display only; no histogram contents are reassigned.
    """
    blocks = exclusive_blocks(record, tail_merged=False)
    ordered_layout = (
        (0, None),
        (10, "$N_{b}\\geq1$, $N_{t}=0$\n$N_{W}=0$, $N_{res}=1$"),
        (11, "$N_{b}\\geq1$, $N_{t}=0$\n$N_{W}=0$, $N_{res}\\geq2$"),
        (1, None),
        (13, "$N_{b}\\geq1$, $N_{t}=0$\n$N_{W}\\geq1$, $N_{res}\\geq1$"),
        (12, "$N_{b}\\geq1$, $N_{t}\\geq1$\n$N_{W}=0$, $N_{res}\\geq1$"),
        (2, None),
        (3, None),
        (4, None),
        (5, None),
        (6, None),
        (7, None),
        (8, None),
        (9, None),
    )
    result = []
    for index, label in ordered_layout:
        block = blocks[index]
        if label is not None:
            block["label"] = label
        result.append(block)
    if sum(int(block["nbin"]) for block in result) != 79:
        raise AssertionError("ordered Nres display must contain 79 bins")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError("merged TROTA histogram is not complete")
    plot_payload = {"search_bin_histograms": payload.get("histograms") or {}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specifications = (
        (BASELINE_SCHEME, "highdm55_inclusive_nres", False),
        (EXCLUSIVE85_SCHEME, "highdm85_exclusive_nres", False),
        (TAILMERGED80_SCHEME, "highdm80_exclusive_nres_tailmerged", True),
    )
    plots = []
    for scheme, name, tail_merged in specifications:
        record = flat_search_record(plot_payload, scheme, scheme, allow_signal=True)
        if not record:
            raise RuntimeError(f"missing scheme {scheme}")
        blocks = (
            exclusive_blocks(record, tail_merged=tail_merged)
            if scheme != BASELINE_SCHEME
            else baseline_blocks(record, nres_zero=False)
        )
        # Repeated recoil-edge strings become unreadable once the complete
        # 55/80/85-bin topology lattice is shown.  The category boxes and
        # separators retain the topology mapping; use stable global search-bin
        # numbers on the axis and retain the recoil definitions in metadata.
        for block in blocks:
            block["xlabels"] = []
        plots.append(draw_flat_blocks(
            blocks,
            args.output_dir / name,
            xlabel="Search bin",
            show_yields=True,
        ))
    ordered_record = flat_search_record(
        plot_payload,
        EXCLUSIVE85_SCHEME,
        EXCLUSIVE85_SCHEME,
        allow_signal=True,
    )
    if not ordered_record:
        raise RuntimeError(f"missing scheme {EXCLUSIVE85_SCHEME}")
    ordered_blocks = ordered_exclusive79_blocks(ordered_record)
    for block in ordered_blocks:
        block["xlabels"] = []
    plots.append(draw_flat_blocks(
        ordered_blocks,
        args.output_dir / "highdm79_exclusive_nres_nb_nt_nw_nres_ordered",
        xlabel="Search bin",
        show_yields=True,
    ))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "physics_status": payload.get("physics_status"),
        "data_policy": "signal-region data blinded",
        "axis_policy": "global search-bin numbers; topology boxes and separators shown",
        "recoil_labels": list(RECOIL6_LABELS),
        "schemes": {BASELINE_SCHEME: 55, EXCLUSIVE85_SCHEME: 85, TAILMERGED80_SCHEME: 80},
        "display_variants": {
            "highdm79_exclusive_nres_nb_nt_nw_nres_ordered": {
                "bins": 79,
                "order": ["Nb", "Nt", "NW", "Nres"],
                "nres_positive_nb_policy": "stored extension is inclusive Nb>=1",
                "omitted_topology": "Nt>=1, NW>=1, Nres>=1",
            }
        },
        "plots": plots,
    }
    write_json(args.output_dir / "plot_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
