#!/usr/bin/env python3
"""Build compact 2024 TF inputs from selected leaves of the merged histograms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SAMPLES = ("ST", "TT", "WtoLNu", "QCD")
HIGH_REGIONS = {
    "SR": "SR",
    "LLCR": "LLCR",
    "QCDCR": "QCDCR",
}
LOW_REGIONS = {
    "SR": "cat7_SR_lowDeltaM",
    "LLCR": "cat2_LLCR_lowDeltaM",
    "QCDCR": "cat3_QCDCR_lowDeltaM",
}
HIGH_SR_SCHEME = "boosted_an17_selected_recoil60_nb2_nt2plus_w0_SR"


def assign(tree: dict[str, Any], path: list[str], value: Any) -> None:
    node = tree
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_selected(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            key, encoded = line.split("\t", 1)
            assign(result, key.split("/"), json.loads(encoded))
        except Exception as error:
            raise ValueError(f"invalid selected histogram line {line_number}") from error
    return result


def nominal(source: dict[str, Any], *path: str) -> dict[str, list[float]]:
    node: Any = source
    for key in path:
        node = (node or {}).get(key)
    if not isinstance(node, dict):
        return {}
    return {
        quantity: [float(value) for value in node.get(quantity, [])]
        for quantity in ("sumw", "sumw2")
    }


def fold_high_sr(source: dict[str, Any], sample: str, n_recoil: int) -> dict[str, list[float]]:
    leaf = nominal(
        source,
        "search_bin_histograms",
        HIGH_SR_SCHEME,
        sample,
        "nominal",
    )
    result = {"sumw": [0.0] * n_recoil, "sumw2": [0.0] * n_recoil}
    for quantity in result:
        values = leaf.get(quantity, [])
        for index, value in enumerate(values):
            result[quantity][index % n_recoil] += float(value)
    return result


def require_length(leaf: dict[str, list[float]], nbin: int, context: str) -> None:
    for quantity in ("sumw", "sumw2"):
        if len(leaf.get(quantity, [])) != nbin:
            raise ValueError(
                f"{context}/{quantity}: found {len(leaf.get(quantity, []))}, expected {nbin}"
            )


def merge_highdm_nb2plus(exacts: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    if not exacts:
        raise ValueError("no High-dM exact inputs were supplied")
    edges = [
        float(value)
        for value in ((exacts[0].get("highdm") or {}).get("recoil_edges") or [])
    ]
    if len(edges) < 2:
        raise ValueError("High-dM exact input has no recoil edges")
    nbin = len(edges) - 1
    output: dict[str, Any] = {}
    for region in ("SR", "LLCR", "QCDCR"):
        output[region] = {"Nb1": {}, "Nb2plus": {}}
        for sample in SAMPLES:
            merged = {
                group: {quantity: [0.0] * nbin for quantity in ("sumw", "sumw2")}
                for group in ("Nb1", "Nb2", "Nb3plus")
            }
            for input_index, exact in enumerate(exacts):
                if exact.get("status") != "complete":
                    raise ValueError(
                        f"High-dM exact input {input_index} is not complete: "
                        f"{exact.get('status')}"
                    )
                current_edges = [
                    float(value)
                    for value in ((exact.get("highdm") or {}).get("recoil_edges") or [])
                ]
                if current_edges != edges:
                    raise ValueError(
                        f"High-dM exact input {input_index} has inconsistent recoil edges"
                    )
                source = (exact.get("highdm") or {}).get("recoil") or {}
                for group in merged:
                    leaf = nominal(source, region, group, sample, "nominal")
                    if not leaf:
                        if input_index == 0:
                            raise ValueError(
                                f"base High-dM exact input lacks {region}/{group}/{sample}"
                            )
                        continue
                    require_length(
                        leaf,
                        nbin,
                        f"highdm-exact-{input_index}/{region}/{group}/{sample}",
                    )
                    for quantity in ("sumw", "sumw2"):
                        merged[group][quantity] = [
                            left + right
                            for left, right in zip(
                                merged[group][quantity], leaf[quantity]
                            )
                        ]
            output[region]["Nb1"][sample] = {"nominal": merged["Nb1"]}
            output[region]["Nb2plus"][sample] = {
                "nominal": {
                    quantity: [
                        left + right
                        for left, right in zip(
                            merged["Nb2"][quantity], merged["Nb3plus"][quantity]
                        )
                    ]
                    for quantity in ("sumw", "sumw2")
                }
            }
    return edges, output


def merge_lowdm_nb2plus(exacts: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    """Merge exact Low-dM yields into the same Nb-only U_T model as High-dM."""
    if not exacts:
        raise ValueError("no Low-dM exact inputs were supplied")
    edges = [
        float(value)
        for value in ((exacts[0].get("lowdm") or {}).get("recoil_edges") or [])
    ]
    if len(edges) < 2:
        raise ValueError("Low-dM exact input has no recoil edges")
    nbin = len(edges) - 1
    output: dict[str, Any] = {}
    for region in ("SR", "LLCR", "QCDCR"):
        output[region] = {"Nb1": {}, "Nb2plus": {}}
        for sample in SAMPLES:
            merged = {
                group: {quantity: [0.0] * nbin for quantity in ("sumw", "sumw2")}
                for group in ("Nb1", "Nb2plus")
            }
            for input_index, exact in enumerate(exacts):
                if exact.get("status") != "complete":
                    raise ValueError(
                        f"Low-dM exact input {input_index} is not complete: "
                        f"{exact.get('status')}"
                    )
                current_edges = [
                    float(value)
                    for value in ((exact.get("lowdm") or {}).get("recoil_edges") or [])
                ]
                if current_edges != edges:
                    raise ValueError(
                        f"Low-dM exact input {input_index} has inconsistent recoil edges"
                    )
                source = (exact.get("lowdm") or {}).get("recoil") or {}
                for group in merged:
                    leaf = nominal(source, region, group, sample, "nominal")
                    if not leaf:
                        if input_index == 0:
                            raise ValueError(
                                f"base Low-dM exact input lacks {region}/{group}/{sample}"
                            )
                        continue
                    require_length(
                        leaf,
                        nbin,
                        f"lowdm-exact-{input_index}/{region}/{group}/{sample}",
                    )
                    for quantity in ("sumw", "sumw2"):
                        merged[group][quantity] = [
                            left + right
                            for left, right in zip(
                                merged[group][quantity], leaf[quantity]
                            )
                        ]
            for group in merged:
                output[region][group][sample] = {"nominal": merged[group]}
    return edges, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--source-hists", type=Path)
    parser.add_argument(
        "--highdm-exact",
        type=Path,
        nargs="+",
        help=(
            "Optional exact High-dM Nb x U_T inputs to add; Low-dM remains "
            "histogram-derived."
        ),
    )
    parser.add_argument(
        "--lowdm-exact",
        type=Path,
        nargs="+",
        help=(
            "Optional exact Low-dM Nb x U_T inputs. When supplied, replace "
            "the 34 search-bin TF model with Nb=1 and Nb>=2 U_T factors."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--campaign-year", choices=("2024", "2025"), default="2024"
    )
    args = parser.parse_args()

    if not args.selected and not (args.highdm_exact and args.lowdm_exact):
        parser.error(
            "--selected is required unless both --highdm-exact and "
            "--lowdm-exact are supplied"
        )
    selected = read_selected(args.selected) if args.selected else {}
    source_status = str(selected.get("status", "exact_only"))
    if args.selected and not source_status.startswith("complete"):
        raise ValueError(f"source histogram status is not complete: {source_status!r}")

    edges: list[float] = []
    high_recoil: dict[str, Any] = {}
    high_groups: list[str] = []
    high_exact_provenance = None
    if args.highdm_exact:
        exacts = [json.loads(path.read_text()) for path in args.highdm_exact]
        edges, high_recoil = merge_highdm_nb2plus(exacts)
        high_groups = ["Nb1", "Nb2plus"]
        high_exact_provenance = [
            {
                "input": str(path),
                "sha256": file_sha256(path),
                "provenance": exact.get("provenance"),
                "summary": exact.get("summary"),
            }
            for path, exact in zip(args.highdm_exact, exacts)
        ]
    else:
        edges = [float(value) for value in selected.get("recoil_pt_bins", [])]
        if len(edges) < 2:
            raise ValueError("recoil_pt_bins are absent")
        high_nbin = len(edges) - 1
        for region, source_region in HIGH_REGIONS.items():
            high_recoil[region] = {"inclusive": {}}
            for sample in SAMPLES:
                leaf = nominal(
                    selected,
                    "histograms",
                    source_region,
                    sample,
                    "nominal",
                )
                if region == "SR" and not leaf.get("sumw"):
                    leaf = fold_high_sr(selected, sample, high_nbin)
                require_length(leaf, high_nbin, f"highdm/{region}/{sample}")
                high_recoil[region]["inclusive"][sample] = {"nominal": leaf}
        high_groups = ["inclusive"]

    low_exact_provenance = None
    if args.lowdm_exact:
        low_exacts = [json.loads(path.read_text()) for path in args.lowdm_exact]
        low_edges, low_recoil = merge_lowdm_nb2plus(low_exacts)
        low_payload = {
            "recoil_edges": low_edges,
            "nb_groups": ["Nb1", "Nb2plus"],
            "recoil": low_recoil,
        }
        low_exact_provenance = [
            {
                "input": str(path),
                "sha256": file_sha256(path),
                "provenance": exact.get("provenance"),
                "summary": exact.get("summary"),
            }
            for path, exact in zip(args.lowdm_exact, low_exacts)
        ]
    else:
        labels = list(
            (((selected.get("search_bin_schemes") or {}).get("cat7_SR_lowDeltaM") or {}).get("bin_labels") or [])
        )
        if len(labels) != 34:
            raise ValueError(f"expected 34 Low-dM labels, found {len(labels)}")
        low_nbin = len(labels)
        low_components: dict[str, Any] = {}
        for region, scheme in LOW_REGIONS.items():
            low_components[region] = {"Nb1": {}, "Nb2plus": {}}
            for sample in SAMPLES:
                leaf = nominal(
                    selected,
                    "search_bin_histograms",
                    scheme,
                    sample,
                    "nominal",
                )
                require_length(leaf, low_nbin, f"lowdm/{region}/{sample}")
                for group, prefix in (("Nb1", "Nb1_"), ("Nb2plus", "Nb2plus_")):
                    masked = {
                        quantity: [
                            value if label.startswith(prefix) else 0.0
                            for label, value in zip(labels, leaf[quantity])
                        ]
                        for quantity in ("sumw", "sumw2")
                    }
                    low_components[region][group][sample] = {"nominal": masked}
        low_payload = {
            "recoil_edges": edges,
            "nb_groups": ["Nb1", "Nb2plus"],
            "search_bin_labels": labels,
            "search_components": low_components,
        }

    output = {
        "schema_version": f"histogram_tf_inputs_{args.campaign_year}_v2",
        "status": "complete",
        "highdm": {
            "recoil_edges": edges,
            "nb_groups": high_groups,
            "recoil": high_recoil,
        },
        "lowdm": low_payload,
        "summary": {
            "selected_processes": list(SAMPLES),
            "source_kind": (
                "exact normalized feature ROOT yields"
                if not args.selected
                else "merged normalized histograms"
            ),
        },
        "provenance": {
            "source_hists": str(args.source_hists) if args.source_hists else None,
            "source_schema_version": selected.get("schema_version"),
            "source_status": source_status,
            "normalization": (
                selected.get("normalization")
                if args.selected
                else (high_exact_provenance or [{}])[0]
                .get("provenance", {})
                .get("normalization_sha256")
            ),
            "selected_histogram_leaves": str(args.selected) if args.selected else None,
            "campaign_year": args.campaign_year,
            "include_data": False,
            "regions": ["SR", "LLCR", "QCDCR"],
            "histogram_derived": bool(args.selected),
            "sample_policy": {
                "top": "TT + ST",
                "qcd": "QCD-4Jets HT-binned current merged process",
                "dy": "DYto2E/Mu/Tau-4Jets; PTLL excluded from current production",
            },
            "highdm_category_policy": (
                "Nb=1 and Nb>=2 in six native U_T bins"
                if args.highdm_exact
                else "inclusive across the stored High-dM search categories; six native U_T bins"
            ),
            "highdm_exact": high_exact_provenance,
            "lowdm_category_policy": (
                "Nb=1 and Nb>=2 in eight native U_T bins; no pTISR/pTb/Nj subdivision"
                if args.lowdm_exact
                else "all 34 native Low-dM search bins"
            ),
            "lowdm_exact": low_exact_provenance,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
