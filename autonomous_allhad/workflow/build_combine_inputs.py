#!/usr/bin/env python3
"""Build the adopted Run-3 LLCR/QCDCR/GCR transfer-factor likelihood.

The dilepton regions are external measurements of R_Z, not Poisson channels.
The photon control region remains in the simultaneous fit.  Its residual
normalization is shared with the matched invisible-Z signal-region template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import re
import sys
from array import array
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from autonomous_allhad.search_bin_categorization import (
    configured_bin_position_groups,
    configured_exclusive_mapping,
)

from background_process_groups import (
    BACKGROUND_PROCESS_ORDER,
    background_grouping_contract,
)


HIGH_CONTROL_REGIONS = ("LLCR", "QCDCR", "GCR")
LOW_CONTROL_REGIONS = HIGH_CONTROL_REGIONS
HIGH_PHYSICAL_GROUPS = {
    "Nb1": ("Nb1",),
    "Nb2plus": ("Nb2", "Nb3plus"),
}
RARE_PROCESSES = ("VV_VVV", "DY", "PhotonJet")
CONTROLLED_PROCESSES = ("Top", "WtoLNu", "QCD", "Zto2Nu")
HIGH_SCHEME = "highdm_search_bins"
LOW_SCHEME = "cat7_SR_lowDeltaM"
RZ_CATEGORIES = (
    "highdm_Nb1",
    "highdm_Nb2plus",
    "lowdm_Nb1",
    "lowdm_Nb2plus",
)
NPS_LUMI_NAME = "lumi_13p6TeV_2024"
ANALYSIS_NUISANCE_PREFIX = "CMS_NPS26012"
CAMPAIGN_YEAR = "2024"
LUMI_LNN = 1.016
MIN_BIN = 1.0e-9
MIN_VARIATION_RATIO = 1.0e-3
MAX_TOP_W_COMPOSITION_LOG_SIGMA = 1.0
TOP_W_COMPOSITION_DIAGNOSTICS: dict[str, dict[str, Any]] = {}
RAW_COMPONENTS = {
    "VV_VVV": ("VV",),
    "Top": ("ST", "TT"),
    "DY": ("DY",),
    "PhotonJet": ("GJ",),
    "WtoLNu": ("WtoLNu",),
    "Zto2Nu": ("Zto2Nu",),
    "QCD": ("QCD",),
    "Other": (),
}
CANONICAL_BACKGROUND_SAMPLES = frozenset(
    {"data_obs"}.union(*(set(samples) for samples in RAW_COMPONENTS.values()))
)


def nps_nuisance_name(name: str) -> str:
    fixed = {
        "btagSF_bc_correlated": "CMS_btag_mediumWP_bc_correlated",
        "btagSF_light_correlated": "CMS_btag_mediumWP_light_correlated",
    }
    yearly = {
        "btagSF_bc_uncorrelated": "CMS_btag_mediumWP_bc_uncorrelated",
        "btagSF_light_uncorrelated": "CMS_btag_mediumWP_light_uncorrelated",
        "electron_hlt": f"{ANALYSIS_NUISANCE_PREFIX}_eff_e_trigger",
        "electron_id": "CMS_eff_e_id_13p6TeV",
        "loose_muon_5to10": f"{ANALYSIS_NUISANCE_PREFIX}_eff_m_loose_5to10",
        "met_trigger": f"{ANALYSIS_NUISANCE_PREFIX}_eff_met_trigger",
        "muon_hlt": f"{ANALYSIS_NUISANCE_PREFIX}_eff_m_trigger",
        "muon_id": f"{ANALYSIS_NUISANCE_PREFIX}_eff_m_id",
        "photon_id": "CMS_eff_g_id_13p6TeV",
        "photon_trigger": f"{ANALYSIS_NUISANCE_PREFIX}_eff_g_trigger",
        "pileup": "CMS_pileup",
        "veto_electron_5to10": f"{ANALYSIS_NUISANCE_PREFIX}_eff_e_veto_5to10",
    }
    if name in fixed:
        return fixed[name]
    if name in yearly:
        return f"{yearly[name]}_{CAMPAIGN_YEAR}"
    return name


def enforce_downstream_input_boundary(args: argparse.Namespace) -> None:
    """Reject every event-level input after canonical histogram promotion."""
    if args.hists.name != "hists.json":
        raise SystemExit(
            "downstream input boundary requires the promoted canonical hists.json"
        )
    forbidden = ("nanoaod", "flat2024", "flat2025", "outputs/nominal")
    inspected = {
        "hists": args.hists,
        "sgamma": args.sgamma,
        "rz_high": args.rz_high,
        "rz_low": args.rz_low,
        "zgamma_double_ratio": args.zgamma_double_ratio,
        "search_bin_config": args.search_bin_config,
        "output_dir": args.output_dir,
    }
    for label, path in inspected.items():
        lowered = str(path).lower()
        if any(token in lowered for token in forbidden):
            raise SystemExit(
                f"forbidden event-level path in downstream {label}: {path}"
            )
    for label in ("sgamma", "rz_high", "rz_low", "zgamma_double_ratio"):
        path = inspected[label]
        if path.suffix != ".json":
            raise SystemExit(f"{label} must be a machine-derived JSON: {path}")


def apply_configured_highdm_bin_merges(
    hists: dict[str, Any],
    exact: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Project bounded canonical High-dM records into configured final bins.

    This runs only after the streaming canonical-histogram extraction.  Every
    signal and background component remains separated by its physical group
    and native recoil bin; only the visible search-bin axis is added.  Hence
    TF, RZ, and Sgamma parameter scopes are unchanged by the projection.
    """
    high = exact["highdm"]
    raw_labels = [str(value) for value in high["search_bin_labels"]]
    input_count = len(raw_labels)
    configured_source_count = len(configured_exclusive_mapping(configuration))
    position_groups = configured_bin_position_groups(configuration)
    final_count = len(position_groups)
    if input_count not in {configured_source_count, final_count}:
        raise ValueError(
            "High-dM configuration/source mismatch: "
            f"expected {configured_source_count} source bins or "
            f"{final_count} already-projected bins, but found "
            f"{input_count} canonical labels"
        )

    def validate_leaf(leaf: dict[str, Any], expected: int) -> None:
        for field in ("entries", "sumw", "sumw2"):
            if field not in leaf:
                continue
            values = leaf.get(field) or []
            if len(values) != expected:
                raise ValueError(
                    f"High-dM {field} has {len(values)} bins; "
                    f"expected {expected}"
                )

    def validate_tree(value: Any, expected: int) -> None:
        if not isinstance(value, dict):
            return
        if "sumw" in value:
            validate_leaf(value, expected)
            return
        for child in value.values():
            validate_tree(child, expected)

    input_trees = (
        hists["search_bin_histograms"][HIGH_SCHEME],
        high["sr_components"],
    )
    if input_count == final_count:
        for tree in input_trees:
            validate_tree(tree, final_count)
        high["bin_projection"] = {
            "source_bin_count": configured_source_count,
            "input_bin_count": input_count,
            "final_bin_count": final_count,
            "already_projected": True,
            "position_groups_zero_based": [
                list(group) for group in position_groups
            ],
            "bin_merges_1based": list(
                configuration.get("bin_merges_1based") or []
            ),
        }
        return high["bin_projection"]

    def rebin_leaf(leaf: dict[str, Any]) -> dict[str, Any]:
        output = dict(leaf)
        for field in ("entries", "sumw", "sumw2"):
            if field not in leaf:
                continue
            values = leaf.get(field) or []
            if len(values) != configured_source_count:
                raise ValueError(
                    f"High-dM {field} has {len(values)} bins; "
                    f"expected {configured_source_count}"
                )
            output[field] = [
                sum(values[position] for position in positions)
                for positions in position_groups
            ]
        return output

    def rebin_tree(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "sumw" in value:
            return rebin_leaf(value)
        return {key: rebin_tree(child) for key, child in value.items()}

    hists["search_bin_histograms"][HIGH_SCHEME] = rebin_tree(
        hists["search_bin_histograms"][HIGH_SCHEME]
    )
    high["sr_components"] = rebin_tree(high["sr_components"])
    high["source_search_bin_labels"] = raw_labels
    high["search_bin_labels"] = [
        "__plus__".join(raw_labels[position] for position in positions)
        for positions in position_groups
    ]
    high["bin_projection"] = {
        "source_bin_count": configured_source_count,
        "input_bin_count": input_count,
        "final_bin_count": final_count,
        "already_projected": False,
        "position_groups_zero_based": [list(group) for group in position_groups],
        "bin_merges_1based": list(configuration.get("bin_merges_1based") or []),
    }
    return high["bin_projection"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def sanitize(name: str) -> str:
    output = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return output or "unnamed"


def signal_process_name(mass_key: str) -> str:
    return "sig_" + sanitize(mass_key)


def parse_mass_key(key: str) -> tuple[int, int]:
    match = re.fullmatch(r"mStop(\d+)_mLSP(\d+)", key)
    if not match:
        raise ValueError(f"invalid mass key: {key}")
    return int(match.group(1)), int(match.group(2))


def stable_path(path: Path) -> str:
    return str(path.resolve()).replace(
        "/eos/home-t/taiwoo", "/eos/user/t/taiwoo"
    )


def make_hist(
    name: str,
    values: np.ndarray,
    sumw2: np.ndarray,
    edges: np.ndarray,
):
    import ROOT

    hist = ROOT.TH1D(
        name,
        name,
        len(values),
        array("d", [float(value) for value in edges]),
    )
    hist.Sumw2()
    for index, value in enumerate(values, start=1):
        hist.SetBinContent(index, float(max(value, MIN_BIN)))
        variance = float(sumw2[index - 1]) if index - 1 < len(sumw2) else 0.0
        hist.SetBinError(index, math.sqrt(max(variance, 0.0)))
    return hist


def write_hist(
    directory,
    name: str,
    values: np.ndarray,
    sumw2: np.ndarray,
    edges: np.ndarray,
) -> None:
    directory.cd()
    make_hist(name, values, sumw2, edges).Write(name, 1)


def leaf_arrays(
    by_sample: dict[str, Any],
    process: str,
    variation: str,
    nbin: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(nbin, dtype=float)
    sumw2 = np.zeros(nbin, dtype=float)
    for sample in RAW_COMPONENTS[process]:
        variations = by_sample.get(sample) or {}
        record = variations.get(variation) or variations.get("nominal") or {}
        current = np.asarray(record.get("sumw") or [], dtype=float)
        current_sumw2 = np.asarray(record.get("sumw2") or [], dtype=float)
        if len(current) != nbin or len(current_sumw2) != nbin:
            if not len(current) and not len(current_sumw2):
                continue
            raise ValueError(
                f"{process}/{sample}/{variation}: expected {nbin} bins, "
                f"found {len(current)}/{len(current_sumw2)}"
            )
        values += current
        sumw2 += current_sumw2
    return values, sumw2


def variation_pairs(
    by_sample: dict[str, Any], process: str, nbin: int
) -> dict[str, dict[str, np.ndarray]]:
    names = {
        name
        for sample in RAW_COMPONENTS[process]
        for name in (by_sample.get(sample) or {})
        if name != "nominal"
    }
    bases = sorted(
        {
            name[:-2]
            for name in names
            if name.endswith("Up") and name[:-2] + "Down" in names
        }
    )
    return {
        nps_nuisance_name(base): {
            "up": leaf_arrays(by_sample, process, base + "Up", nbin)[0],
            "down": leaf_arrays(by_sample, process, base + "Down", nbin)[0],
        }
        for base in bases
    }


def one_bin_background(
    by_sample: dict[str, Any],
    process: str,
    source_bin: int,
    nbin: int,
) -> dict[str, Any] | None:
    nominal, sumw2 = leaf_arrays(by_sample, process, "nominal", nbin)
    value = float(nominal[source_bin])
    variance = float(sumw2[source_bin])
    if value <= 0.0 and variance <= 0.0:
        return None
    nominal_value = max(value, MIN_BIN)
    floor = max(MIN_BIN, nominal_value * MIN_VARIATION_RATIO)
    variations = {}
    for nuisance, pair in variation_pairs(by_sample, process, nbin).items():
        up_value = max(float(pair["up"][source_bin]), floor)
        down_value = max(float(pair["down"][source_bin]), floor)
        if np.isclose(
            up_value, nominal_value, rtol=1.0e-12, atol=1.0e-15
        ) and np.isclose(
            down_value, nominal_value, rtol=1.0e-12, atol=1.0e-15
        ):
            continue
        variations[nuisance] = {
            "up": np.asarray([up_value]),
            "down": np.asarray([down_value]),
        }
    return {
        "nominal": np.asarray([nominal_value]),
        "sumw2": np.asarray([max(variance, 0.0)]),
        "variations": variations,
    }


def sum_one_bin_backgrounds(
    records: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    retained = [record for record in records if record is not None]
    if not retained:
        return None
    nominal = sum(
        (record["nominal"] for record in retained), np.zeros(1)
    )
    sumw2 = sum((record["sumw2"] for record in retained), np.zeros(1))
    nuisances = sorted(
        {
            nuisance
            for record in retained
            for nuisance in record["variations"]
        }
    )
    variations = {}
    for nuisance in nuisances:
        up = np.zeros(1)
        down = np.zeros(1)
        for record in retained:
            pair = record["variations"].get(nuisance)
            up += pair["up"] if pair else record["nominal"]
            down += pair["down"] if pair else record["nominal"]
        variations[nuisance] = {"up": up, "down": down}
    return {"nominal": nominal, "sumw2": sumw2, "variations": variations}


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


def object_member_bounds(
    buffer: mmap.mmap,
    bounds: tuple[int, int],
) -> list[tuple[str, tuple[int, int]]]:
    """Return direct child bounds without decoding the parent object."""
    start, end = bounds
    if buffer[start] != ord("{"):
        raise ValueError(f"expected JSON object at byte {start}")
    cursor = start + 1
    members: list[tuple[str, tuple[int, int]]] = []
    while cursor < end:
        while cursor < end and buffer[cursor] in b" \t\r\n,":
            cursor += 1
        if cursor >= end or buffer[cursor] == ord("}"):
            break
        if buffer[cursor] != ord('"'):
            raise ValueError(f"expected JSON member name at byte {cursor}")
        key_start = cursor
        cursor += 1
        escaped = False
        while cursor < end:
            value = buffer[cursor]
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                cursor += 1
                break
            cursor += 1
        key = json.loads(buffer[key_start:cursor])
        while cursor < end and buffer[cursor] in b" \t\r\n":
            cursor += 1
        if cursor >= end or buffer[cursor] != ord(":"):
            raise ValueError(f"expected ':' after JSON member {key!r}")
        cursor += 1
        while cursor < end and buffer[cursor] in b" \t\r\n":
            cursor += 1
        value_start = cursor
        value_end = json_value_end(buffer, value_start)
        members.append((str(key), (value_start, value_end)))
        cursor = value_end
    return members


def named_member_bounds(
    buffer: mmap.mmap,
    bounds: tuple[int, int],
    name: str,
) -> tuple[int, int]:
    for key, child_bounds in object_member_bounds(buffer, bounds):
        if key == name:
            return child_bounds
    raise ValueError(f"member {name!r} is absent from bounded JSON object")


def project_sample_tree(
    buffer: mmap.mmap,
    bounds: tuple[int, int],
    levels_before_samples: int,
    keep_sample,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, child_bounds in object_member_bounds(buffer, bounds):
        if levels_before_samples:
            child = project_sample_tree(
                buffer,
                child_bounds,
                levels_before_samples - 1,
                keep_sample,
            )
            if child:
                output[name] = child
        elif keep_sample(name):
            output[name] = json.loads(buffer[child_bounds[0]:child_bounds[1]])
    return output


def extract_top_level_object(path: Path, name: str) -> dict[str, Any]:
    """Decode one bounded top-level object without parsing the full histogram."""
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as payload:
            marker = json.dumps(name).encode() + b":"
            member = payload.find(marker)
            if member < 0:
                raise ValueError(f"top-level member {name} absent from {path}")
            value_start = member + len(marker)
            while payload[value_start] in b" \t\r\n":
                value_start += 1
            value_end = json_value_end(payload, value_start)
            value = json.loads(payload[value_start:value_end])
    if not isinstance(value, dict):
        raise ValueError(f"top-level member {name} is not an object")
    return value


def extract_search_scheme(
    path: Path,
    scheme: str,
    topology: str | None = None,
) -> dict[str, Any]:
    """Project backgrounds and one optional signal topology from a scheme."""
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as payload:
            section_marker = b'"search_bin_histograms":'
            section = payload.find(section_marker)
            if section < 0:
                raise ValueError(f"search_bin_histograms absent from {path}")
            section_start = section + len(section_marker)
            while payload[section_start] in b" \t\r\n":
                section_start += 1
            section_end = json_value_end(payload, section_start)
            scheme_bounds = named_member_bounds(
                payload,
                (section_start, section_end),
                scheme,
            )
            signal_prefix = f"{topology}_" if topology else None
            return project_sample_tree(
                payload,
                scheme_bounds,
                0,
                lambda sample: sample in CANONICAL_BACKGROUND_SAMPLES
                or bool(signal_prefix and sample.startswith(signal_prefix)),
            )


def extract_component_tree(
    path: Path,
    section_name: str,
    levels_before_samples: int,
    member: str | None = None,
) -> dict[str, Any]:
    """Project only background/data leaves from a canonical component tree."""
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as payload:
            marker = json.dumps(section_name).encode() + b":"
            section_member = payload.find(marker)
            if section_member < 0:
                raise ValueError(f"section {section_name} absent from {path}")
            section_start = section_member + len(marker)
            while payload[section_start] in b" \t\r\n":
                section_start += 1
            section_bounds = (
                section_start,
                json_value_end(payload, section_start),
            )
            if member is not None:
                section_bounds = named_member_bounds(
                    payload,
                    section_bounds,
                    member,
                )
            return project_sample_tree(
                payload,
                section_bounds,
                levels_before_samples,
                lambda sample: sample in CANONICAL_BACKGROUND_SAMPLES,
            )


def split_lowdm_group(
    variations: dict[str, Any], group: str, nbin: int
) -> dict[str, Any]:
    selected = (
        np.arange(nbin) < 16
        if group == "Nb1"
        else np.arange(nbin) >= 16
    )
    output = {}
    for variation, record in variations.items():
        output[variation] = {
            field: np.where(
                selected,
                np.asarray(record.get(field) or [0] * nbin),
                0,
            ).tolist()
            for field in ("sumw", "sumw2", "entries")
        }
    return output


def extract_current_histogram_input(
    path: Path, topology: str
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Materialize only bounded canonical sections needed by Combine.

    No ROOT input is consulted.  Background CR/SR components were filled in
    the same canonical histogram pass and are validated during its merge.
    """
    schemes = extract_top_level_object(path, "search_bin_schemes")
    high_metadata = schemes[HIGH_SCHEME]
    high_labels = list(high_metadata.get("bin_labels") or [])
    low_labels = list(schemes[LOW_SCHEME].get("bin_labels") or [])
    if not high_labels or not low_labels:
        raise ValueError("canonical search-bin labels are missing")

    high_scheme = extract_search_scheme(path, HIGH_SCHEME, topology)
    low_scheme = extract_search_scheme(path, LOW_SCHEME, topology)
    signal_prefix = f"{topology}_"
    high_signals = {
        sample: variations
        for sample, variations in high_scheme.items()
        if sample.startswith(signal_prefix)
    }
    low_signals = {
        sample: variations
        for sample, variations in low_scheme.items()
        if sample.startswith(signal_prefix)
    }
    common_signals = sorted(set(high_signals) & set(low_signals))
    if not common_signals:
        raise ValueError(f"no {topology} signals in canonical histogram")
    hists = {
        "search_bin_histograms": {
            HIGH_SCHEME: {
                sample: high_signals[sample] for sample in common_signals
            },
            LOW_SCHEME: {
                sample: low_signals[sample] for sample in common_signals
            },
        }
    }

    control_components = extract_component_tree(
        path,
        "highdm_control_components",
        levels_before_samples=2,
    )
    search_components = extract_component_tree(
        path,
        "highdm_search_bin_components",
        levels_before_samples=2,
        member=HIGH_SCHEME,
    )
    low_backgrounds = {
        sample: variations
        for sample, variations in low_scheme.items()
        if sample == "data_obs" or not sample.startswith(("T2tt_", "T2bW_", "T2tb_"))
    }
    low_components: dict[str, Any] = {}
    for region, scheme in (
        ("LLCR", "cat2_LLCR_lowDeltaM"),
        ("QCDCR", "cat3_QCDCR_lowDeltaM"),
        ("GCR", "cat4_GCR_lowDeltaM"),
        ("DY2E", "cat5_DY2E_lowDeltaM"),
        ("DY2M", "cat6_DY2M_lowDeltaM"),
        ("SR", LOW_SCHEME),
    ):
        region_tree = low_backgrounds if scheme == LOW_SCHEME else {
            sample: variations
            for sample, variations in extract_search_scheme(path, scheme).items()
            if sample == "data_obs" or not sample.startswith(("T2tt_", "T2bW_", "T2tb_"))
        }
        low_components[region] = {
            group: {
                sample: split_lowdm_group(variations, group, len(low_labels))
                for sample, variations in region_tree.items()
            }
            for group in ("Nb1", "Nb2plus")
        }
    exact = {
        "schema_version": f"canonical_card_inputs_{CAMPAIGN_YEAR}_v1",
        "status": "complete",
        "highdm": {
            "recoil_edges": [250.0, 300.0, 350.0, 400.0, 500.0, 800.0, 1500.0],
            "recoil_last_bin_open_ended": True,
            "recoil_display_cap_gev": 1500.0,
            "recoil": control_components,
            "sr_components": search_components,
            "search_bin_labels": high_labels,
        },
        "lowdm": {
            "search_bin_labels": low_labels,
            "search_components": low_components,
        },
    }
    return hists, exact, common_signals


def scaled_record(
    record: dict[str, Any] | None, scale: float
) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "nominal": record["nominal"] * scale,
        "sumw2": record["sumw2"] * scale * scale,
        "variations": {
            nuisance: {
                "up": pair["up"] * scale,
                "down": pair["down"] * scale,
            }
            for nuisance, pair in record["variations"].items()
        },
    }


def require_positive(record: dict[str, Any], label: str, key: str = "value") -> float:
    if record.get("status") != "complete":
        raise ValueError(f"{label} is not complete: {record}")
    value = float(record[key])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} is nonpositive/nonfinite: {value}")
    return value


def logical_records(
    sources: dict[str, dict[str, Any]],
    logical_group: str,
    process: str,
    source_bin: int,
    nbin: int,
    scales: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    records = []
    for physical_group in HIGH_PHYSICAL_GROUPS[logical_group]:
        record = one_bin_background(
            sources[physical_group], process, source_bin, nbin
        )
        if scales is not None:
            record = scaled_record(record, scales[physical_group])
        records.append(record)
    return sum_one_bin_backgrounds(records)


def low_family(label: str) -> str:
    return re.sub(r"_recoil_[0-9]+$", "", label)


def low_sgamma(sgamma: dict[str, Any], label: str) -> tuple[float, float, str, int]:
    family = low_family(label)
    payload = sgamma["lowdm_families"][family]
    source_bin = int(re.search(r"_recoil_([0-9]+)$", label).group(1)) - 1
    if source_bin < 0 or source_bin >= len(payload["bins"]):
        raise ValueError(f"invalid Low-dM recoil bin for {label}")
    q = require_positive(payload["Q"], f"Qgamma/{family}")
    shape = require_positive(
        payload["bins"][source_bin]["Sgamma"],
        f"Sgamma/{family}/bin{source_bin}",
    )
    return q, shape, family, source_bin


def low_sgamma_models(
    sgamma: dict[str, Any], labels: list[str]
) -> dict[int, dict[str, Any]]:
    """Build identifiable Low-dM Sgamma shape parameters.

    A bin with nonpositive photon ``data - other`` cannot define a positive
    multiplicative shape parameter. Such a bin is pooled with the adjacent
    bin that gives the larger combined photon-MC denominator, repeating only
    if needed. The SR search bins stay separate; only the GCR shape parameter
    is shared across the low-statistics pool.
    """
    source_by_family: dict[str, list[int]] = {}
    for source_bin, label in enumerate(labels):
        source_by_family.setdefault(low_family(label), []).append(source_bin)

    output: dict[int, dict[str, Any]] = {}
    for family, source_bins in source_by_family.items():
        payload = sgamma["lowdm_families"][family]
        factor_bins = payload["bins"]
        if len(factor_bins) != len(source_bins):
            raise ValueError(
                f"Sgamma bin count mismatch for {family}: "
                f"{len(factor_bins)} versus {len(source_bins)}"
            )
        qgamma = require_positive(payload["Q"], f"Qgamma/{family}")

        def sums(block: list[int]) -> tuple[float, float]:
            numerator = 0.0
            denominator = 0.0
            for local_bin in block:
                record = factor_bins[local_bin]["Sgamma"]
                current_numerator = float(record.get("numerator", float("nan")))
                current_denominator = float(
                    record.get("denominator", float("nan"))
                )
                if not np.isfinite(current_numerator) or not np.isfinite(
                    current_denominator
                ):
                    raise ValueError(
                        f"nonfinite Sgamma ingredients for {family}/bin{local_bin}"
                    )
                numerator += current_numerator
                denominator += current_denominator
            return numerator, denominator

        blocks = [[index] for index in range(len(factor_bins))]
        while True:
            invalid_index = next(
                (
                    index
                    for index, block in enumerate(blocks)
                    if sums(block)[0] <= 0.0 or sums(block)[1] <= 0.0
                ),
                None,
            )
            if invalid_index is None:
                break
            neighbors = [
                index
                for index in (invalid_index - 1, invalid_index + 1)
                if 0 <= index < len(blocks)
            ]
            if not neighbors:
                raise ValueError(f"cannot pool unavailable Sgamma family {family}")
            neighbor = max(
                neighbors,
                key=lambda index: sums(blocks[invalid_index] + blocks[index])[1],
            )
            left = min(invalid_index, neighbor)
            right = max(invalid_index, neighbor)
            blocks[left] = blocks[left] + blocks[right]
            blocks.pop(right)

        for block in blocks:
            numerator, denominator = sums(block)
            shape = numerator / denominator
            if not np.isfinite(shape) or shape <= 0.0:
                raise ValueError(
                    f"invalid pooled Sgamma for {family}/{block}: {shape}"
                )
            global_bins = [source_bins[index] for index in block]
            parameter_bin: int | str = (
                global_bins[0]
                if len(global_bins) == 1
                else f"{global_bins[0]}to{global_bins[-1]}"
            )
            group = "Nb1" if global_bins[0] < 16 else "Nb2plus"
            parameter = rate_parameter(
                "sgamma_shape", "lowdm", group, parameter_bin
            )
            for local_bin, source_bin in zip(block, global_bins):
                output[source_bin] = {
                    "qgamma": qgamma,
                    "sgamma": float(shape),
                    "family": family,
                    "family_recoil_bin_zero_based": local_bin,
                    "pool_local_bins_zero_based": list(block),
                    "pool_source_bins_zero_based": list(global_bins),
                    "parameter": parameter,
                }
    if set(output) != set(range(len(labels))):
        raise ValueError("Low-dM Sgamma model does not cover every search bin")
    return output


def high_sgamma(
    sgamma: dict[str, Any], physical_group: str, recoil_bin: int
) -> tuple[float, float]:
    payload = sgamma["highdm"][physical_group]
    factor_bins = payload["bins"]
    if not factor_bins:
        raise ValueError(f"Sgamma/highdm/{physical_group} has no bins")
    # The adopted measurement merges the two native High-dM tail bins into
    # one 500-inf shape measurement.  Both native GCR/SR bins must therefore
    # use the same final Sgamma entry rather than inventing a second shape
    # parameter.
    factor_bin = min(recoil_bin, len(factor_bins) - 1)
    q = require_positive(payload["Q"], f"Qgamma/highdm/{physical_group}")
    shape = require_positive(
        factor_bins[factor_bin]["Sgamma"],
        f"Sgamma/highdm/{physical_group}/bin{factor_bin}",
    )
    return q, shape


def high_sgamma_parameter_bin(sgamma: dict[str, Any], recoil_bin: int) -> int:
    counts = {
        len((sgamma["highdm"][physical] or {}).get("bins") or [])
        for physical in ("Nb1", "Nb2", "Nb3plus")
    }
    if len(counts) != 1 or not counts:
        raise ValueError(f"inconsistent High-dM Sgamma bin counts: {counts}")
    count = next(iter(counts))
    if count <= 0:
        raise ValueError("High-dM Sgamma has no measurement bins")
    return min(recoil_bin, count - 1)


def high_effective_sgamma(
    sgamma: dict[str, Any],
    gcr_sources: dict[str, dict[str, Any]],
    logical_group: str,
    recoil_bin: int,
    nbin: int,
) -> float | None:
    """Return the identifiable Sgamma shape parameter for a logical group.

    ``Nb2`` and ``Nb3plus`` share one aggregated ``Nb2plus`` photon-control
    channel. A separate free Sgamma parameter for each physical component
    would therefore be underconstrained. The single shared parameter is the
    Qgamma-scaled photon-MC weighted average, so its initial expectation is
    exactly the sum of the measured physical-category photon predictions.
    """
    numerator = 0.0
    denominator = 0.0
    for physical_group in HIGH_PHYSICAL_GROUPS[logical_group]:
        qgamma, shape = high_sgamma(sgamma, physical_group, recoil_bin)
        record = one_bin_background(
            gcr_sources[physical_group], "PhotonJet", recoil_bin, nbin
        )
        if record is None:
            continue
        weight = qgamma * float(record["nominal"][0])
        numerator += weight * shape
        denominator += weight
    if denominator <= 0.0:
        return None
    value = numerator / denominator
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "invalid effective Sgamma for "
            f"highdm/{logical_group}/bin{recoil_bin}: {value}"
        )
    return float(value)


def high_effective_sgamma_grouped(
    sgamma: dict[str, Any],
    gcr_sources: dict[str, dict[str, Any]],
    logical_group: str,
    recoil_bins: list[int],
    nbin: int,
) -> float | None:
    """Photon-weighted Sgamma for native bins sharing one measurement bin."""
    numerator = 0.0
    denominator = 0.0
    for physical_group in HIGH_PHYSICAL_GROUPS[logical_group]:
        for recoil_bin in recoil_bins:
            qgamma, shape = high_sgamma(sgamma, physical_group, recoil_bin)
            record = one_bin_background(
                gcr_sources[physical_group], "PhotonJet", recoil_bin, nbin
            )
            if record is None:
                continue
            weight = qgamma * float(record["nominal"][0])
            numerator += weight * shape
            denominator += weight
    if denominator <= 0.0:
        return None
    value = numerator / denominator
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            "invalid grouped effective Sgamma for "
            f"highdm/{logical_group}/{recoil_bins}: {value}"
        )
    return float(value)


def build_rz_covariance(
    high_summary: dict[str, Any], low_summary: dict[str, Any]
) -> dict[str, Any]:
    low_rz = low_summary.get("rz_low") or low_summary.get("rz_low_feature")
    if not isinstance(low_rz, dict):
        raise ValueError("Low-dM RZ measurement is missing")
    records = {
        "highdm_Nb1": high_summary["rz_high"]["combined"]["Nb1"],
        "highdm_Nb2plus": high_summary["rz_high"]["combined"]["Nb2plus"],
        "lowdm_Nb1": low_rz["combined"]["Nb1"],
        "lowdm_Nb2plus": low_rz["combined"]["Nb2plus"],
    }
    central = np.asarray(
        [require_positive(records[key], f"RZ/{key}", "RZ") for key in RZ_CATEGORIES]
    )
    errors = np.asarray([float(records[key]["RZ_stat"]) for key in RZ_CATEGORIES])
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError(f"invalid RZ statistical errors: {errors}")
    covariance_r = np.diag(errors * errors)
    covariance_eta = covariance_r / np.outer(central, central)
    cholesky = np.linalg.cholesky(covariance_eta)
    nuisances = []
    for index, category in enumerate(RZ_CATEGORIES):
        nuisances.append(
            {
                "name": f"{ANALYSIS_NUISANCE_PREFIX}_RZstat_{category}_{CAMPAIGN_YEAR}",
                "category": category,
                "log_coefficient": float(cholesky[index, index]),
            }
        )
    return {
        "schema_version": f"rz_covariance_{CAMPAIGN_YEAR}_v1",
        "status": "temporary_statistical_only",
        "warning": (
            "Diagonal statistical-only covariance. Replace with the final "
            "documented cross-category covariance before the final result."
        ),
        "categories": list(RZ_CATEGORIES),
        "central": central.tolist(),
        "statistical_errors": errors.tolist(),
        "covariance_r": covariance_r.tolist(),
        "covariance_log_r": covariance_eta.tolist(),
        "cholesky_log_r": cholesky.tolist(),
        "nuisances": nuisances,
    }


def rz_value(covariance: dict[str, Any], category: str) -> float:
    return float(covariance["central"][covariance["categories"].index(category)])


def rz_nuisances(covariance: dict[str, Any], category: str) -> list[dict[str, Any]]:
    index = covariance["categories"].index(category)
    output = []
    for column, nuisance in enumerate(covariance["nuisances"]):
        coefficient = float(covariance["cholesky_log_r"][index][column])
        if coefficient == 0.0:
            continue
        output.append(
            {
                "name": nuisance["name"],
                "down": math.exp(-coefficient),
                "up": math.exp(coefficient),
            }
        )
    return output


def closure_record(
    double_ratio: dict[str, Any], regime: str, low: float, high: float
) -> tuple[str, float, list[int]]:
    selected = []
    for index, record in enumerate(double_ratio[regime]["bins"]):
        if record.get("status") != "complete":
            continue
        if float(record["high"]) <= low or float(record["low"]) >= high:
            continue
        value = float(record["double_ratio"])
        if np.isfinite(value):
            selected.append((index, abs(value - 1.0)))
    if not selected:
        return "", 0.0, []
    delta = max(value for _, value in selected)
    high_label = "Inf" if not np.isfinite(high) else str(int(high))
    name = (
        f"{ANALYSIS_NUISANCE_PREFIX}_zgammaNonclosure_{regime}_"
        f"u{int(low)}to{high_label}_{CAMPAIGN_YEAR}"
    )
    return name, delta, [index for index, _ in selected]


def low_geometry(label: str) -> tuple[float, float]:
    family = low_family(label)
    local_bin = int(re.search(r"_recoil_([0-9]+)$", label).group(1)) - 1
    if "PISR300to500" in family:
        low = 300.0 + 100.0 * local_bin
    elif "PISR500plus" in family:
        low = 450.0 + 100.0 * local_bin
    else:
        raise ValueError(f"cannot infer Low-dM U_T geometry for {label}")
    family_bins = 4 if family.startswith("Nb1_") else 3
    high = low + 100.0 if local_bin + 1 < family_bins else float("inf")
    return low, high


def low_control_groups(
    labels: list[str],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Aggregate Low-dM CRs like the High-dM Nb x recoil model.

    The 34 Low-dM SR bins retain their full pTISR/pTb/Nj categorization.  A
    control observation, however, is not split once per SR bin.  Bins with
    the same Nb requirement and the same physical recoil interval share one
    LLCR, QCDCR, and GCR channel and therefore one normalization parameter.
    This is the Low-dM analogue of the High-dM model, where many exclusive SR
    categories share an Nb x recoil control constraint.
    """
    grouped: dict[tuple[str, float, float], list[int]] = {}
    for source_bin, label in enumerate(labels):
        group = "Nb1" if str(label).startswith("Nb1_") else "Nb2plus"
        low, high = low_geometry(str(label))
        grouped.setdefault((group, low, high), []).append(source_bin)

    groups: list[dict[str, Any]] = []
    by_source: dict[int, dict[str, Any]] = {}
    for (group, low, high), source_bins in grouped.items():
        high_label = "Inf" if not np.isfinite(high) else str(int(high))
        key = f"u{int(low)}to{high_label}"
        record = {
            "nb_group": group,
            "recoil_low": float(low),
            "recoil_high": float(high),
            "key": key,
            "source_bins_zero_based": list(source_bins),
        }
        groups.append(record)
        for source_bin in source_bins:
            by_source[source_bin] = record
    if set(by_source) != set(range(len(labels))):
        raise ValueError("Low-dM control grouping does not cover every SR bin")
    return groups, by_source


def summed_low_background(
    by_sample: dict[str, Any],
    process: str,
    source_bins: list[int],
    nbin: int,
    scale: float | None = None,
) -> dict[str, Any] | None:
    records = [
        one_bin_background(by_sample, process, source_bin, nbin)
        for source_bin in source_bins
    ]
    if scale is not None:
        records = [scaled_record(record, scale) for record in records]
    return sum_one_bin_backgrounds(records)


def low_gcr_shape_initial(
    observation: float,
    backgrounds: dict[str, Any],
    context: str,
) -> float:
    photon = backgrounds.get("PhotonJet")
    if photon is None:
        raise ValueError(f"{context} has no Qgamma-scaled PhotonJet template")
    denominator = float(photon["nominal"][0])
    numerator = observation - sum(
        float(record["nominal"][0])
        for process, record in backgrounds.items()
        if process != "PhotonJet"
    )
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise ValueError(f"nonfinite Low-dM GCR ingredients for {context}")
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError(
            f"unidentifiable Low-dM GCR shape for {context}: "
            f"numerator={numerator}, denominator={denominator}"
        )
    return float(numerator / denominator)


def rate_parameter(
    kind: str, regime: str, group: str, bin_index: int | str
) -> str:
    return f"{kind}_{regime}_{group}_bin{bin_index}_{CAMPAIGN_YEAR}"


def add_extra(
    channel: dict[str, Any], process: str, records: list[dict[str, Any]]
) -> None:
    if records:
        channel.setdefault("extra_lnN", {}).setdefault(process, []).extend(records)


def top_w_composition(
    top: dict[str, Any] | None,
    wjets: dict[str, Any] | None,
    name: str,
) -> dict[str, list[dict[str, float | str]]]:
    """MC-statistical log-ratio prior with no first-order total-yield change.

    The delta-method Gaussian approximation is not valid once its log-ratio
    sigma reaches unity.  Those sparse control bins are already handled by
    ``autoMCStats``; adding an unbounded log-normal composition prior would
    both double count that uncertainty and overflow numerically.
    """

    if top is None or wjets is None:
        TOP_W_COMPOSITION_DIAGNOSTICS[name] = {"status": "missing_process"}
        return {}
    top_yield = float(top["nominal"][0])
    w_yield = float(wjets["nominal"][0])
    if top_yield <= 0.0 or w_yield <= 0.0:
        TOP_W_COMPOSITION_DIAGNOSTICS[name] = {
            "status": "nonpositive_yield",
            "top_yield": top_yield,
            "w_yield": w_yield,
        }
        return {}
    sigma = math.sqrt(
        max(float(top["sumw2"][0]), 0.0) / (top_yield * top_yield)
        + max(float(wjets["sumw2"][0]), 0.0) / (w_yield * w_yield)
    )
    if not np.isfinite(sigma) or sigma <= 0.0:
        TOP_W_COMPOSITION_DIAGNOSTICS[name] = {
            "status": "invalid_sigma",
            "sigma": float(sigma) if np.isfinite(sigma) else None,
            "top_yield": top_yield,
            "w_yield": w_yield,
        }
        return {}
    if sigma >= MAX_TOP_W_COMPOSITION_LOG_SIGMA:
        TOP_W_COMPOSITION_DIAGNOSTICS[name] = {
            "status": "autoMCStats_only_low_effective_count",
            "sigma": float(sigma),
            "top_yield": top_yield,
            "w_yield": w_yield,
        }
        return {}
    top_fraction = top_yield / (top_yield + w_yield)
    top_shift = (1.0 - top_fraction) * sigma
    w_shift = top_fraction * sigma
    result = {
        "Top": [{"name": name, "down": math.exp(-top_shift), "up": math.exp(top_shift)}],
        "WtoLNu": [{"name": name, "down": math.exp(w_shift), "up": math.exp(-w_shift)}],
    }
    TOP_W_COMPOSITION_DIAGNOSTICS[name] = {
        "status": "applied",
        "sigma": float(sigma),
        "top_yield": top_yield,
        "w_yield": w_yield,
    }
    return result


def composition_name(regime: str, group: str, bin_index: int | str) -> str:
    return (
        f"{ANALYSIS_NUISANCE_PREFIX}_topWComposition_{regime}_{group}_"
        f"bin{bin_index}_{CAMPAIGN_YEAR}"
    )


def build_channels(
    exact: dict[str, Any],
    sgamma: dict[str, Any],
    rz_covariance: dict[str, Any],
    double_ratio: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any], list[str]]:
    channels: list[dict[str, Any]] = []
    bin_map: dict[str, Any] = {"highdm": [], "lowdm": []}
    high = exact["highdm"]
    high_nbin = len(high["recoil_edges"]) - 1
    high_shape_bin = {
        recoil_bin: high_sgamma_parameter_bin(sgamma, recoil_bin)
        for recoil_bin in range(high_nbin)
    }
    native_bins_by_shape: dict[int, list[int]] = {}
    for recoil_bin, shape_bin in high_shape_bin.items():
        native_bins_by_shape.setdefault(shape_bin, []).append(recoil_bin)
    grouped_high_shape_initial = {
        (group, shape_bin): high_effective_sgamma_grouped(
            sgamma,
            high["recoil"]["GCR"],
            group,
            native_bins,
            high_nbin,
        )
        for group in HIGH_PHYSICAL_GROUPS
        for shape_bin, native_bins in native_bins_by_shape.items()
    }
    high_shape_initial = {
        (group, recoil_bin): grouped_high_shape_initial[
            (group, high_shape_bin[recoil_bin])
        ]
        for group in HIGH_PHYSICAL_GROUPS
        for recoil_bin in range(high_nbin)
    }

    for region in HIGH_CONTROL_REGIONS:
        sources = high["recoil"][region]
        for group in HIGH_PHYSICAL_GROUPS:
            for recoil_bin in range(high_nbin):
                backgrounds: dict[str, Any] = {}
                for process in BACKGROUND_PROCESS_ORDER:
                    scales = None
                    if region == "GCR" and process == "PhotonJet":
                        scales = {
                            physical: high_sgamma(sgamma, physical, recoil_bin)[0]
                            for physical in HIGH_PHYSICAL_GROUPS[group]
                        }
                    record = logical_records(
                        sources, group, process, recoil_bin, high_nbin, scales
                    )
                    if record is not None:
                        backgrounds[process] = record
                rate_params: dict[str, str] = {}
                if region == "LLCR":
                    parameter = rate_parameter("ll_norm", "highdm", group, recoil_bin)
                    for process in ("Top", "WtoLNu"):
                        if process in backgrounds:
                            rate_params[process] = parameter
                elif region == "QCDCR" and "QCD" in backgrounds:
                    rate_params["QCD"] = rate_parameter(
                        "qcd_norm", "highdm", group, recoil_bin
                    )
                elif region == "GCR" and "PhotonJet" in backgrounds:
                    shape_initial = high_shape_initial[(group, recoil_bin)]
                    if shape_initial is None:
                        raise ValueError(
                            "GCR PhotonJet exists without a measurable Sgamma: "
                            f"highdm/{group}/bin{recoil_bin}"
                        )
                    rate_params["PhotonJet"] = rate_parameter(
                        "sgamma_shape",
                        "highdm",
                        group,
                        high_shape_bin[recoil_bin],
                    )
                observation = sum(
                    float(
                        sources[physical]["data_obs"]["nominal"]["sumw"][recoil_bin]
                    )
                    for physical in HIGH_PHYSICAL_GROUPS[group]
                )
                channel = {
                        "name": f"{region}_highdm_{group}_bin{recoil_bin}",
                        "kind": "highdm_control",
                        "regime": "highdm",
                        "region": region,
                        "nb_group": group,
                        "source_bin": recoil_bin,
                        "backgrounds": backgrounds,
                        "rate_params": rate_params,
                        "rate_initial": {
                            process: (
                                high_shape_initial[(group, recoil_bin)]
                                if region == "GCR" and process == "PhotonJet"
                                else 1.0
                            )
                            for process in rate_params
                        },
                        "signal_source": None,
                        "observation": observation,
                        "extra_lnN": {},
                    }
                if region == "LLCR":
                    for process, records in top_w_composition(
                        backgrounds.get("Top"),
                        backgrounds.get("WtoLNu"),
                        composition_name("highdm", group, recoil_bin),
                    ).items():
                        add_extra(channel, process, records)
                channels.append(channel)

    high_components = high["sr_components"]
    rz_scale = {
        group: rz_value(rz_covariance, f"highdm_{group}")
        for group in HIGH_PHYSICAL_GROUPS
    }
    high_closure = []
    for recoil_bin in range(high_nbin):
        low, high_edge = high["recoil_edges"][recoil_bin : recoil_bin + 2]
        high_closure.append(closure_record(double_ratio, "highdm", low, high_edge))
    high_search_bins = len(high["search_bin_labels"])
    for output_bin, output_label in enumerate(high["search_bin_labels"]):
        channel = {
            "name": f"SR_highdm_bin{output_bin}",
            "kind": "highdm_signal_searchbin",
            "regime": "highdm",
            "region": "SR",
            "source_bin": output_bin,
            "bin_label": output_label,
            "backgrounds": {},
            "rate_params": {},
            "rate_initial": {},
            "signal_source": ("highdm", output_bin),
            "observation": None,
            "extra_lnN": {},
        }
        for group, physical_groups in HIGH_PHYSICAL_GROUPS.items():
            for recoil_bin in range(high_nbin):
                for process in ("Top", "WtoLNu", "QCD"):
                    record = sum_one_bin_backgrounds(
                        [
                            one_bin_background(
                                ((high_components.get(physical) or {}).get(
                                    f"recoil{recoil_bin}"
                                ))
                                or {},
                                process,
                                output_bin,
                                high_search_bins,
                            )
                            for physical in physical_groups
                        ]
                    )
                    if record is None:
                        continue
                    component = f"{process}_{group}_u{recoil_bin}"
                    channel["backgrounds"][component] = record
                    kind = (
                        "ll_norm"
                        if process in ("Top", "WtoLNu")
                        else "qcd_norm"
                        if process == "QCD"
                        else "sgamma_shape"
                    )
                    parameter = rate_parameter(kind, "highdm", group, recoil_bin)
                    channel["rate_params"][component] = parameter
                    channel["rate_initial"][component] = 1.0
                    if process in ("Top", "WtoLNu"):
                        composition = top_w_composition(
                            logical_records(
                                high["recoil"]["LLCR"],
                                group,
                                "Top",
                                recoil_bin,
                                high_nbin,
                            ),
                            logical_records(
                                high["recoil"]["LLCR"],
                                group,
                                "WtoLNu",
                                recoil_bin,
                                high_nbin,
                            ),
                            composition_name("highdm", group, recoil_bin),
                        )
                        add_extra(channel, component, composition.get(process, []))
                z_record = sum_one_bin_backgrounds(
                    [
                        scaled_record(
                            one_bin_background(
                                ((high_components.get(physical) or {}).get(
                                    f"recoil{recoil_bin}"
                                ))
                                or {},
                                "Zto2Nu",
                                output_bin,
                                high_search_bins,
                            ),
                            rz_scale[group],
                        )
                        for physical in physical_groups
                    ]
                )
                if z_record is not None:
                    shape_initial = high_shape_initial[(group, recoil_bin)]
                    if shape_initial is None:
                        raise ValueError(
                            "Z SR contribution exists without a photon-control "
                            f"Sgamma measurement: highdm/{group}/bin{recoil_bin}"
                        )
                    component = f"Zto2Nu_{group}_u{recoil_bin}"
                    channel["backgrounds"][component] = z_record
                    parameter = rate_parameter(
                        "sgamma_shape",
                        "highdm",
                        group,
                        high_shape_bin[recoil_bin],
                    )
                    channel["rate_params"][component] = parameter
                    channel["rate_initial"][component] = shape_initial
                    add_extra(
                        channel,
                        component,
                        rz_nuisances(rz_covariance, f"highdm_{group}"),
                    )
                    name, delta, source = high_closure[recoil_bin]
                    if delta > 0.0:
                        add_extra(
                            channel,
                            component,
                            [{"name": name, "down": 1.0 / (1.0 + delta), "up": 1.0 + delta}],
                        )
        for process in RARE_PROCESSES:
            record = sum_one_bin_backgrounds(
                [
                    one_bin_background(
                        ((high_components.get(group) or {}).get(
                            f"recoil{recoil_bin}"
                        ))
                        or {},
                        process,
                        output_bin,
                        high_search_bins,
                    )
                    for physical_groups in HIGH_PHYSICAL_GROUPS.values()
                    for group in physical_groups
                    for recoil_bin in range(high_nbin)
                ]
            )
            if record is not None:
                channel["backgrounds"][process] = record
        channels.append(channel)
        bin_map["highdm"].append(
            {
                "channel": channel["name"],
                "source_bin_zero_based": output_bin,
                "label": output_label,
                "native_recoil_bins": [
                    recoil_bin
                    for recoil_bin in range(high_nbin)
                    if any(
                        sum(
                            int(value)
                            for value in (
                                ((
                                    ((high_components.get(group) or {}).get(
                                        f"recoil{recoil_bin}"
                                    ) or {}).get(sample) or {}
                                ).get("nominal") or {}).get("entries")
                                or []
                            )
                        ) > 0
                        for physical_groups in HIGH_PHYSICAL_GROUPS.values()
                        for group in physical_groups
                        for sample in (
                            ((high_components.get(group) or {}).get(
                                f"recoil{recoil_bin}"
                            ) or {})
                        )
                    )
                ],
            }
        )

    low = exact["lowdm"]
    labels = low["search_bin_labels"]
    low_nbin = len(labels)
    low_groups, low_group_by_source = low_control_groups(labels)
    low_qgamma = {
        group: require_positive(
            sgamma["lowdm_families"][group]["Q"],
            f"Qgamma/lowdm/{group}",
        )
        for group in ("Nb1", "Nb2plus")
    }
    low_sr_qcd_groups = {
        (str(control_group["nb_group"]), str(control_group["key"]))
        for control_group in low_groups
        if summed_low_background(
            low["search_components"]["SR"][
                str(control_group["nb_group"])
            ],
            "QCD",
            list(control_group["source_bins_zero_based"]),
            low_nbin,
        )
        is not None
    }
    low_shape_initial: dict[tuple[str, str], float] = {}
    for region in LOW_CONTROL_REGIONS:
        source_payload = low["search_components"]
        for control_group in low_groups:
            group = str(control_group["nb_group"])
            key = str(control_group["key"])
            source_bins = list(control_group["source_bins_zero_based"])
            by_sample = source_payload[region][group]
            q = low_qgamma[group]
            backgrounds: dict[str, Any] = {}
            for process in BACKGROUND_PROCESS_ORDER:
                record = summed_low_background(
                    by_sample,
                    process,
                    source_bins,
                    low_nbin,
                    q if region == "GCR" and process == "PhotonJet" else None,
                )
                if record is not None:
                    backgrounds[process] = record
            rate_params: dict[str, str] = {}
            if region == "LLCR":
                parameter = rate_parameter("ll_norm", "lowdm", group, key)
                for process in ("Top", "WtoLNu"):
                    if process in backgrounds:
                        rate_params[process] = parameter
            elif (
                region == "QCDCR"
                and "QCD" in backgrounds
                and (group, key) in low_sr_qcd_groups
            ):
                rate_params["QCD"] = rate_parameter(
                    "qcd_norm", "lowdm", group, key
                )
            elif region == "GCR" and "PhotonJet" in backgrounds:
                rate_params["PhotonJet"] = rate_parameter(
                    "sgamma_shape", "lowdm", group, key
                )
            data_values = by_sample["data_obs"]["nominal"]["sumw"]
            observation = float(
                sum(float(data_values[index]) for index in source_bins)
            )
            if region == "GCR" and "PhotonJet" in backgrounds:
                low_shape_initial[(group, key)] = low_gcr_shape_initial(
                    observation,
                    backgrounds,
                    f"lowdm/{group}/{key}",
                )
            shape = low_shape_initial.get((group, key), 1.0)
            channel = {
                    "name": f"{region}_lowdm_{group}_{key}",
                    "kind": "lowdm_control",
                    "regime": "lowdm",
                    "region": region,
                    "nb_group": group,
                    "control_group": key,
                    "source_bins_zero_based": source_bins,
                    "recoil_low": control_group["recoil_low"],
                    "recoil_high": control_group["recoil_high"],
                    "backgrounds": backgrounds,
                    "rate_params": rate_params,
                    "rate_initial": {
                        process: (
                            shape
                            if region == "GCR" and process == "PhotonJet"
                            else 1.0
                        )
                        for process in rate_params
                    },
                    "signal_source": None,
                    "observation": observation,
                    "extra_lnN": {},
                }
            if region == "LLCR":
                for process, records in top_w_composition(
                    backgrounds.get("Top"),
                    backgrounds.get("WtoLNu"),
                    composition_name("lowdm", group, key),
                ).items():
                    add_extra(channel, process, records)
            channels.append(channel)

    missing_shape_groups = [
        f"{record['nb_group']}/{record['key']}"
        for record in low_groups
        if (str(record["nb_group"]), str(record["key"]))
        not in low_shape_initial
    ]
    if missing_shape_groups:
        raise ValueError(
            "Low-dM GCR does not constrain every control group: "
            + ", ".join(missing_shape_groups)
        )

    for source_bin, label in enumerate(labels):
        control_group = low_group_by_source[source_bin]
        group = str(control_group["nb_group"])
        key = str(control_group["key"])
        source_bins = list(control_group["source_bins_zero_based"])
        by_sample = low["search_components"]["SR"][group]
        shape = low_shape_initial[(group, key)]
        q = low_qgamma[group]
        sgamma_parameter = rate_parameter(
            "sgamma_shape", "lowdm", group, key
        )
        backgrounds = {}
        rate_params = {}
        rate_initial = {}
        channel = {
            "name": f"SR_lowdm_bin{source_bin}",
            "kind": "lowdm_signal_searchbin",
            "regime": "lowdm",
            "region": "SR",
            "nb_group": group,
            "control_group": key,
            "source_bin": source_bin,
            "bin_label": label,
            "backgrounds": backgrounds,
            "rate_params": rate_params,
            "rate_initial": rate_initial,
            "signal_source": ("lowdm", source_bin),
            "observation": None,
            "extra_lnN": {},
        }
        for process in BACKGROUND_PROCESS_ORDER:
            if process == "Zto2Nu":
                continue
            record = one_bin_background(by_sample, process, source_bin, low_nbin)
            if record is None:
                continue
            if process in ("Top", "WtoLNu"):
                parameter = rate_parameter("ll_norm", "lowdm", group, key)
            elif process == "QCD":
                parameter = rate_parameter("qcd_norm", "lowdm", group, key)
            else:
                parameter = None
            backgrounds[process] = record
            if parameter:
                rate_params[process] = parameter
                rate_initial[process] = 1.0
                if process in ("Top", "WtoLNu"):
                    llcr = low["search_components"]["LLCR"][group]
                    composition = top_w_composition(
                        summed_low_background(
                            llcr, "Top", source_bins, low_nbin
                        ),
                        summed_low_background(
                            llcr, "WtoLNu", source_bins, low_nbin
                        ),
                        composition_name("lowdm", group, key),
                    )
                    add_extra(channel, process, composition.get(process, []))
        z_record = one_bin_background(
            by_sample, "Zto2Nu", source_bin, low_nbin
        )
        if z_record is not None:
            z_record = scaled_record(
                z_record,
                rz_value(rz_covariance, f"lowdm_{group}"),
            )
            backgrounds["Zto2Nu"] = z_record
            rate_params["Zto2Nu"] = sgamma_parameter
            rate_initial["Zto2Nu"] = shape
            add_extra(
                channel,
                "Zto2Nu",
                rz_nuisances(rz_covariance, f"lowdm_{group}"),
            )
            low_edge, high_edge = low_geometry(label)
            name, delta, source = closure_record(
                double_ratio, "lowdm", low_edge, high_edge
            )
            if delta > 0.0:
                add_extra(
                    channel,
                    "Zto2Nu",
                    [{"name": name, "down": 1.0 / (1.0 + delta), "up": 1.0 + delta}],
                )
        channels.append(channel)
        bin_map["lowdm"].append(
            {
                "channel": channel["name"],
                "nb_group": group,
                "source_bin_zero_based": source_bin,
                "label": label,
                "control_group": key,
                "control_group_source_bins_zero_based": source_bins,
                "control_recoil_low": control_group["recoil_low"],
                "control_recoil_high": (
                    float(control_group["recoil_high"])
                    if np.isfinite(control_group["recoil_high"])
                    else None
                ),
                "control_last_bin_open_ended": not np.isfinite(
                    control_group["recoil_high"]
                ),
                "sgamma": shape,
                "qgamma": q,
                "sgamma_parameter": sgamma_parameter,
            }
        )

    dropped_empty_control_channels: list[str] = []
    retained_channels: list[dict[str, Any]] = []
    for channel in channels:
        if channel["backgrounds"] or channel["region"] == "SR":
            retained_channels.append(channel)
            continue
        observation = float(channel.get("observation") or 0.0)
        if observation != 0.0:
            raise ValueError(
                "control channel has data but no contributing process: "
                f"{channel['name']} observation={observation}"
            )
        dropped_empty_control_channels.append(str(channel["name"]))
    channels = retained_channels

    scopes: dict[str, set[str]] = {}
    for channel in channels:
        scope = "sr" if channel["region"] == "SR" else "cr"
        for parameter in channel["rate_params"].values():
            scopes.setdefault(parameter, set()).add(scope)
    invalid = sorted(
        parameter for parameter, scope in scopes.items() if scope != {"cr", "sr"}
    )
    invalid_set = set(invalid)
    for channel in channels:
        channel["rate_params"] = {
            process: parameter
            for process, parameter in channel["rate_params"].items()
            if parameter not in invalid_set
        }
        channel["rate_initial"] = {
            process: value
            for process, value in channel["rate_initial"].items()
            if process in channel["rate_params"]
        }
    parameter_initials: dict[str, float] = {}
    for channel in channels:
        for process, parameter in channel["rate_params"].items():
            initial = float(channel["rate_initial"][process])
            previous = parameter_initials.setdefault(parameter, initial)
            if not math.isclose(previous, initial, rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise ValueError(
                    f"rate parameter {parameter} has inconsistent initial values: "
                    f"{previous} versus {initial} in {channel['name']}/{process}"
                )
    return channels, invalid, bin_map, dropped_empty_control_channels


def overwrite_observations(
    output_root: Path,
    channels: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    import ROOT

    root_file = ROOT.TFile(str(output_root), "UPDATE")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"cannot update {output_root}")
    try:
        for channel in channels:
            observation = channel.get("observation")
            if observation is None:
                continue
            directory = root_file.GetDirectory(channel["name"])
            if not directory:
                raise RuntimeError(f"missing ROOT directory {channel['name']}")
            value = np.asarray([max(float(observation), 0.0)])
            write_hist(directory, "data_obs", value, value, np.asarray([0.0, 1.0]))
            summary["channels"][channel["name"]]["observation"] = float(observation)
            summary["channels"][channel["name"]]["observation_source"] = (
                f"{CAMPAIGN_YEAR} data"
            )
    finally:
        root_file.Close()


def signal_histogram(
    hists: dict[str, Any],
    regime: str,
    mass_key: str,
    topology: str,
) -> dict[str, Any]:
    scheme = HIGH_SCHEME if regime == "highdm" else LOW_SCHEME
    return (
        ((hists.get("search_bin_histograms") or {}).get(scheme) or {}).get(
            f"{topology}_{mass_key}"
        )
        or {}
    )


def signal_leaf(
    hists: dict[str, Any],
    regime: str,
    mass_key: str,
    variation: str,
    topology: str,
) -> tuple[np.ndarray, np.ndarray]:
    variations = signal_histogram(hists, regime, mass_key, topology)
    record = variations.get(variation) or variations.get("nominal") or {}
    return (
        np.asarray(record.get("sumw") or [], dtype=float),
        np.asarray(record.get("sumw2") or [], dtype=float),
    )


def signal_variations(
    hists: dict[str, Any],
    regime: str,
    mass_key: str,
    topology: str,
) -> list[str]:
    names = set(signal_histogram(hists, regime, mass_key, topology))
    return sorted(
        {
            name[:-2]
            for name in names
            if name.endswith("Up") and name[:-2] + "Down" in names
        }
    )


def mass_points(
    hists: dict[str, Any],
    only: list[str] | None,
    max_mstop: int,
    topology: str,
) -> list[str]:
    selected: set[str] = set()
    for scheme in (HIGH_SCHEME, LOW_SCHEME):
        samples = (hists.get("search_bin_histograms") or {}).get(scheme) or {}
        for sample, variations in samples.items():
            match = re.fullmatch(
                rf"{re.escape(topology)}_(mStop\d+_mLSP\d+)", sample
            )
            if not match:
                continue
            mass_key = match.group(1)
            if only and mass_key not in only:
                continue
            mstop, mlsp = parse_mass_key(mass_key)
            nominal = (variations.get("nominal") or {}).get("sumw") or []
            if mstop <= max_mstop and mlsp < mstop and sum(nominal) > 0.0:
                selected.add(mass_key)
    return sorted(selected, key=parse_mass_key)


def build_root(
    channels: list[dict[str, Any]],
    hists: dict[str, Any],
    masses: list[str],
    output_root: Path,
    topology: str,
) -> dict[str, Any]:
    import ROOT

    output_root.parent.mkdir(parents=True, exist_ok=True)
    root_file = ROOT.TFile(str(output_root), "RECREATE")
    summary: dict[str, Any] = {
        "channels": {},
        "signals": {},
        "background_grouping_contract": background_grouping_contract(),
    }
    try:
        for channel in channels:
            directory = root_file.mkdir(channel["name"])
            background_total = np.zeros(1)
            background_sumw2 = np.zeros(1)
            background_summary = {}
            for process, record in sorted(channel["backgrounds"].items()):
                write_hist(
                    directory,
                    process,
                    record["nominal"],
                    record["sumw2"],
                    np.asarray([0.0, 1.0]),
                )
                background_total += record["nominal"]
                background_sumw2 += record["sumw2"]
                background_summary[process] = {
                    "yield": float(record["nominal"][0]),
                    "weight_nuisances": sorted(record["variations"]),
                    "nuisance_factors": {
                        nuisance: {
                            "down": float(
                                pair["down"][0] / record["nominal"][0]
                            ),
                            "up": float(
                                pair["up"][0] / record["nominal"][0]
                            ),
                        }
                        for nuisance, pair in record["variations"].items()
                    },
                }
            write_hist(
                directory,
                "data_obs",
                background_total,
                background_sumw2,
                np.asarray([0.0, 1.0]),
            )
            summary["channels"][channel["name"]] = {
                "kind": channel["kind"],
                "regime": channel["regime"],
                "region": channel["region"],
                "source_bin": channel.get("source_bin"),
                "background_yield": float(background_total[0]),
                "backgrounds": background_summary,
                "rate_params": channel["rate_params"],
            }
            for field in (
                "nb_group",
                "bin_label",
                "control_group",
                "source_bins_zero_based",
                "recoil_low",
                "recoil_high",
            ):
                if channel.get(field) is not None:
                    value = channel[field]
                    if field == "recoil_high" and not np.isfinite(value):
                        summary["channels"][channel["name"]][
                            "last_bin_open_ended"
                        ] = True
                        value = None
                    summary["channels"][channel["name"]][field] = value

            for mass_key in masses:
                process = signal_process_name(mass_key)
                signal = np.zeros(1)
                signal_sumw2 = np.zeros(1)
                signal_factors: dict[str, dict[str, float]] = {}
                sources = channel.get("signal_sources")
                if sources is None:
                    sources = (
                        [channel["signal_source"]]
                        if channel.get("signal_source")
                        else []
                    )
                if sources:
                    regimes = {regime for regime, _ in sources}
                    if len(regimes) != 1:
                        raise ValueError(
                            f"mixed signal regimes in {channel['name']}: "
                            f"{sorted(regimes)}"
                        )
                    regime = next(iter(regimes))
                    source_bins = [source_bin for _, source_bin in sources]
                    nominal, sumw2 = signal_leaf(
                        hists, regime, mass_key, "nominal", topology
                    )
                    if all(index < len(nominal) for index in source_bins):
                        signal[0] = max(
                            float(sum(nominal[index] for index in source_bins)),
                            MIN_BIN,
                        )
                        signal_sumw2[0] = max(
                            float(sum(sumw2[index] for index in source_bins)),
                            0.0,
                        )
                    for nuisance in signal_variations(
                        hists, regime, mass_key, topology
                    ):
                        up = signal_leaf(
                            hists, regime, mass_key, nuisance + "Up", topology
                        )[0]
                        down = signal_leaf(
                            hists, regime, mass_key, nuisance + "Down", topology
                        )[0]
                        if any(
                            index >= len(up) or index >= len(down)
                            for index in source_bins
                        ):
                            continue
                        floor = max(MIN_BIN, signal[0] * MIN_VARIATION_RATIO)
                        up_value = max(
                            float(sum(up[index] for index in source_bins)), floor
                        )
                        down_value = max(
                            float(sum(down[index] for index in source_bins)),
                            floor,
                        )
                        if np.isclose(
                            up_value, signal[0], rtol=1.0e-12, atol=1.0e-15
                        ) and np.isclose(
                            down_value,
                            signal[0],
                            rtol=1.0e-12,
                            atol=1.0e-15,
                        ):
                            continue
                        signal_factors[nps_nuisance_name(nuisance)] = {
                            "down": float(down_value / signal[0]),
                            "up": float(up_value / signal[0]),
                        }
                write_hist(
                    directory,
                    process,
                    signal,
                    signal_sumw2,
                    np.asarray([0.0, 1.0]),
                )
                signal_summary = summary["signals"].setdefault(
                    mass_key,
                    {
                        "topology": topology,
                        "process": process,
                        "channels": {},
                        "weight_nuisances": {},
                        "nuisance_factors": {},
                    },
                )
                signal_summary["channels"][channel["name"]] = float(signal[0])
                signal_summary["nuisance_factors"][
                    channel["name"]
                ] = signal_factors
                for nuisance in signal_factors:
                    signal_summary["weight_nuisances"].setdefault(
                        nuisance, []
                    ).append(channel["name"])
    finally:
        root_file.Close()
    return summary


def write_parallel_runner(
    cards: dict[str, str],
    output_dir: Path,
    runner: Path,
    jobs: int,
    point_timeout: int,
) -> None:
    validator = runner.with_name("validate_limit_outputs.py")
    validator.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import ROOT

EXPECTED_QUANTILES = (0.025, 0.16, 0.5, 0.84, 0.975)
MASS_RE = re.compile(
    r"^higgsCombine_(mStop[0-9]+_mLSP[0-9]+)\\.AsymptoticLimits.*\\.root$"
)


def valid_limit(path: Path) -> bool:
    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        return False
    try:
        tree = root_file.Get("limit")
        if not tree or int(tree.GetEntries()) != len(EXPECTED_QUANTILES):
            return False
        rows = [
            (float(entry.quantileExpected), float(entry.limit))
            for entry in tree
        ]
        return all(
            math.isfinite(quantile)
            and math.isfinite(value)
            and value >= 0.0
            and math.isclose(
                quantile,
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-5,
            )
            for (quantile, value), expected in zip(rows, EXPECTED_QUANTILES)
        )
    finally:
        root_file.Close()


output_dir = Path(sys.argv[1])
valid = []
for path in sorted(output_dir.glob("higgsCombine_*.AsymptoticLimits*.root")):
    match = MASS_RE.match(path.name)
    if match and valid_limit(path):
        valid.append(match.group(1))
if valid:
    print("\\n".join(sorted(set(valid))))
"""
    )
    validator.chmod(0o755)
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "export PYTHONNOUSERSITE=1",
        "COMBINE=${COMBINE:-combine}",
        f"OUTDIR={stable_path(output_dir)}",
        f"VALIDATOR={stable_path(validator)}",
        f"EXPECTED_POINTS={len(cards)}",
        f"MAX_JOBS={int(jobs)}",
        f"POINT_TIMEOUT={int(point_timeout)}",
        'mkdir -p "$OUTDIR"',
        'VALID_OUTPUTS="$OUTDIR/valid_limit_outputs.txt"',
        "validate_outputs() {",
        '  python3 "$VALIDATOR" "$OUTDIR" > "${VALID_OUTPUTS}.tmp"',
        '  mv "${VALID_OUTPUTS}.tmp" "$VALID_OUTPUTS"',
        "}",
        "validate_outputs",
        "run_one() {",
        '  local mass="$1"',
        '  local card="$2"',
        '  if grep -Fxq "$mass" "$VALID_OUTPUTS"; then echo "[combine-skip] ${mass}"; return 0; fi',
        '  echo "[combine-start] ${mass}"',
        '  (cd "$OUTDIR" && timeout "$POINT_TIMEOUT" "$COMBINE" -M AsymptoticLimits --run blind -n "_${mass}" "$card") > "$OUTDIR/log_${mass}.txt" 2>&1',
        "  local rc=$?",
        '  echo "[combine-rc] ${mass} rc=${rc}"',
        "  return ${rc}",
        "}",
        "fail=0",
        "running=0",
    ]
    for mass_key, card in sorted(cards.items()):
        lines.append(
            f"run_one {mass_key} {stable_path(Path(card))} || fail=1 &"
        )
        lines.append("running=$((running + 1))")
        lines.append(
            'if [ "$running" -ge "$MAX_JOBS" ]; then wait -n || fail=1; running=$((running - 1)); fi'
        )
    lines.extend(
        [
            "wait || fail=1",
            "validate_outputs || fail=1",
            'VALID_COUNT=$(wc -l < "$VALID_OUTPUTS")',
            'if [ "$VALID_COUNT" -ne "$EXPECTED_POINTS" ]; then',
            '  echo "[combine-validation] valid=${VALID_COUNT} expected=${EXPECTED_POINTS}"',
            "  fail=1",
            "fi",
            "exit $fail",
        ]
    )
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("\n".join(lines) + "\n")
    runner.chmod(0o755)


def datacard_text(
    template_root: Path,
    channels: list[dict[str, Any]],
    mass_key: str,
    summary: dict[str, Any],
    auto_mc_stats: int,
) -> str:
    signal = signal_process_name(mass_key)
    channel_names = [channel["name"] for channel in channels]
    backgrounds = sorted(
        {
            process
            for channel in channels
            for process in channel["backgrounds"]
        }
    )
    background_ids = {
        process: index + 1 for index, process in enumerate(backgrounds)
    }
    channel_map = {channel["name"]: channel for channel in channels}
    columns: list[tuple[str, str, int]] = []
    for channel in channels:
        name = channel["name"]
        if summary["signals"][mass_key]["channels"].get(name, 0.0) > 0.0:
            columns.append((name, signal, 0))
        for process in sorted(channel["backgrounds"]):
            columns.append((name, process, background_ids[process]))
    nuisances = sorted(
        {
            nuisance
            for channel in channels
            for record in channel["backgrounds"].values()
            for nuisance in record["variations"]
        }
        | set(summary["signals"][mass_key]["weight_nuisances"])
    )
    lines = [
        f"# {CAMPAIGN_YEAR} adopted likelihood: simultaneous LLCR/QCDCR/GCR; external RZ; canonical High-dM + Low-dM34.",
        "# Z(inv): RZ is the external normalization; Qgamma-normalized GCR data constrain the shared Sgamma shape parameter.",
        "# RZ covariance is explicitly temporary diagonal statistical-only pending the final cross-category matrix.",
        "imax * number of channels",
        "jmax * number of backgrounds",
        "kmax * number of nuisance parameters",
        "------------",
        f"shapes * * {stable_path(template_root)} $CHANNEL/$PROCESS",
        "------------",
        "bin " + " ".join(channel_names),
        "observation " + " ".join(["-1"] * len(channel_names)),
        "------------",
        "bin " + " ".join(item[0] for item in columns),
        "process " + " ".join(item[1] for item in columns),
        "process " + " ".join(str(item[2]) for item in columns),
        "rate " + " ".join(["-1"] * len(columns)),
        "------------",
    ]
    signal_factors = summary["signals"][mass_key]["nuisance_factors"]
    for nuisance in nuisances:
        mask = []
        for channel_name, process, _ in columns:
            if process == signal:
                pair = (signal_factors.get(channel_name) or {}).get(nuisance)
            else:
                pair = (
                    (
                        summary["channels"][channel_name]["backgrounds"].get(process)
                        or {}
                    ).get("nuisance_factors")
                    or {}
                ).get(nuisance)
            mask.append("-" if not pair else f"{pair['down']:.8g}/{pair['up']:.8g}")
        lines.append(nuisance + " lnN " + " ".join(mask))
    lines.append(
        NPS_LUMI_NAME
        + " lnN "
        + " ".join(
            (
                f"{LUMI_LNN:.3f}"
                if process == signal
                or (
                    process in {"VV_VVV", "DY", "PhotonJet"}
                    and process not in channel_map[channel_name]["rate_params"]
                )
                else "-"
            )
            for channel_name, process, _ in columns
        )
    )
    extra_names = sorted(
        {
            item["name"]
            for channel in channels
            for records in channel.get("extra_lnN", {}).values()
            for item in records
        }
    )
    for nuisance in extra_names:
        mask = []
        for channel_name, process, _ in columns:
            item = next(
                (
                    record
                    for record in channel_map[channel_name]
                    .get("extra_lnN", {})
                    .get(process, [])
                    if record["name"] == nuisance
                ),
                None,
            )
            mask.append("-" if item is None else f"{item['down']:.8g}/{item['up']:.8g}")
        lines.append(nuisance + " lnN " + " ".join(mask))
    rate_lines = []
    for channel in channels:
        for process, parameter in sorted(channel["rate_params"].items()):
            initial = min(max(float(channel["rate_initial"].get(process, 1.0)), 1.0e-4), 9.999)
            rate_lines.append(
                f"{parameter} rateParam {channel['name']} {process} {initial:.8g} [0,10]"
            )
    lines.extend(rate_lines)
    if auto_mc_stats >= 0:
        lines.append(f"* autoMCStats {auto_mc_stats}")
    lines.extend(
        [
            "# Top=TT+ST. One ll_norm is shared by Top and W in each Nb x recoil CR group and all matched exclusive SR categories.",
            "# A shared MC-derived anti-correlated Top/W composition nuisance accompanies each grouped ll_norm constraint.",
            "# GCR PhotonJet is pre-scaled only by Qgamma; the shared Sgamma rateParam is shape-only and starts at its measured value.",
            "# Zto2Nu SR is pre-scaled only by RZ; no extra Z-normalization rateParam is present.",
            "# GCR statistics enter only through its Poisson channel; no Sgamma statistical Gaussian is duplicated.",
            "# DY2E/DY2M are excluded as Poisson channels and enter only through external RZ and Z/gamma closure.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_cards(
    channels: list[dict[str, Any]],
    masses: list[str],
    template_root: Path,
    summary: dict[str, Any],
    output_dir: Path,
    auto_mc_stats: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = {}
    for mass_key in masses:
        card = output_dir / f"datacard_{mass_key}.txt"
        card.write_text(
            datacard_text(template_root, channels, mass_key, summary, auto_mc_stats)
        )
        cards[mass_key] = str(card)
    return cards


def main() -> int:
    global CAMPAIGN_YEAR, NPS_LUMI_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--hists", type=Path, required=True)
    parser.add_argument("--hists-sha256")
    parser.add_argument("--campaign-year", choices=("2024", "2025"), required=True)
    parser.add_argument("--topology", choices=("T2tt", "T2bW", "T2tb"), required=True)
    parser.add_argument("--sgamma", type=Path, required=True)
    parser.add_argument("--rz-high", type=Path, required=True)
    parser.add_argument("--rz-low", type=Path, required=True)
    parser.add_argument("--zgamma-double-ratio", type=Path, required=True)
    parser.add_argument(
        "--search-bin-config",
        type=Path,
        required=True,
        help="Adopted year-specific High-dM search-bin configuration",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--max-mstop", type=int, default=1800)
    parser.add_argument("--auto-mc-stats", type=int, default=10)
    parser.add_argument("--runner-jobs", type=int, default=4)
    parser.add_argument("--point-timeout", type=int, default=1800)
    args = parser.parse_args()

    enforce_downstream_input_boundary(args)

    CAMPAIGN_YEAR = str(args.campaign_year)
    NPS_LUMI_NAME = f"lumi_13p6TeV_{CAMPAIGN_YEAR}"

    input_paths = {
        "hists": args.hists,
        "sgamma": args.sgamma,
        "rz_high": args.rz_high,
        "rz_low": args.rz_low,
        "zgamma_double_ratio": args.zgamma_double_ratio,
        "search_bin_config": args.search_bin_config,
    }
    sgamma = read_json(args.sgamma)
    rz_high = read_json(args.rz_high)
    rz_low = read_json(args.rz_low)
    double_ratio = read_json(args.zgamma_double_ratio)
    search_bin_configuration = read_json(args.search_bin_config)
    for label, payload in (
        ("Sgamma", sgamma),
        ("RZ high", rz_high),
        ("Z/gamma double ratio", double_ratio),
    ):
        if payload.get("status") != "complete":
            raise SystemExit(f"{label} input incomplete: {payload.get('status')}")
    if rz_low.get("status") not in {"complete", "feature_stage_complete"}:
        raise SystemExit(f"RZ low input incomplete: {rz_low.get('status')}")
    if search_bin_configuration.get("schema_version") != "search_bin_scheme_v1":
        raise SystemExit("unsupported High-dM search-bin configuration")
    if search_bin_configuration.get("scheme_name") != HIGH_SCHEME:
        raise SystemExit(f"search-bin configuration must name {HIGH_SCHEME}")
    if str(search_bin_configuration.get("campaign_year")) != CAMPAIGN_YEAR:
        raise SystemExit(
            "search-bin configuration year does not match --campaign-year"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    hists, exact, extracted_signals = extract_current_histogram_input(
        args.hists, args.topology
    )
    highdm_bin_projection = apply_configured_highdm_bin_merges(
        hists,
        exact,
        search_bin_configuration,
    )
    rz_covariance = build_rz_covariance(rz_high, rz_low)
    write_json(output_dir / "rz_covariance.json", rz_covariance)
    (
        channels,
        unmatched_rate_parameters,
        bin_map,
        dropped_empty_control_channels,
    ) = build_channels(exact, sgamma, rz_covariance, double_ratio)
    write_json(output_dir / "bin_map.json", bin_map)
    low_control_group_summary: list[dict[str, Any]] = []
    seen_low_control_groups: set[tuple[str, str]] = set()
    for record in bin_map["lowdm"]:
        group_key = (
            str(record["nb_group"]),
            str(record["control_group"]),
        )
        if group_key in seen_low_control_groups:
            continue
        seen_low_control_groups.add(group_key)
        low_control_group_summary.append(
            {
                "nb_group": group_key[0],
                "key": group_key[1],
                "recoil_low": record["control_recoil_low"],
                "recoil_high": record["control_recoil_high"],
                "source_bins_zero_based": record[
                    "control_group_source_bins_zero_based"
                ],
            }
        )
    masses = mass_points(
        hists,
        args.only,
        args.max_mstop,
        topology=args.topology,
    )
    if not masses:
        raise SystemExit("no signal mass points selected")
    template_root = output_dir / "templates.root"
    card_dir = output_dir / "cards"
    limit_dir = output_dir / "limits"
    runner = output_dir / "run_limits.sh"
    summary = build_root(
        channels,
        hists,
        masses,
        template_root,
        topology=args.topology,
    )
    overwrite_observations(template_root, channels, summary)
    cards = write_cards(
        channels, masses, template_root, summary, card_dir, args.auto_mc_stats
    )
    write_parallel_runner(
        cards, limit_dir, runner, args.runner_jobs, args.point_timeout
    )
    rate_parameters = sorted(
        {
            parameter
            for channel in channels
            for parameter in channel["rate_params"].values()
        }
    )
    scopes = {
        parameter: sorted(
            {
                "sr" if channel["region"] == "SR" else "cr"
                for channel in channels
                if parameter in channel["rate_params"].values()
            }
        )
        for parameter in rate_parameters
    }
    channel_counts = {
        "total": len(channels),
        "highdm_control": sum(item["kind"] == "highdm_control" for item in channels),
        "highdm_signal": sum(item["kind"] == "highdm_signal_searchbin" for item in channels),
        "lowdm_control": sum(item["kind"] == "lowdm_control" for item in channels),
        "lowdm_signal": sum(item["kind"] == "lowdm_signal_searchbin" for item in channels),
    }
    manifest = {
        "schema_version": f"canonical_tf_rz_sgamma_model_{CAMPAIGN_YEAR}_v2",
        "status": "combine_inputs_ready",
        "model": {
            "simultaneous_control_regions": list(HIGH_CONTROL_REGIONS),
            "dilepton_poisson_channels": False,
            "lost_lepton": (
                "one free ll_norm shared by Top and W per Nb x recoil "
                "control group; multiple exclusive SR categories share it"
            ),
            "top_w_composition": (
                "MC-derived log-ratio nuisance, anti-correlated at fixed total "
                "to first order when sigma<1; sparse bins use autoMCStats only"
            ),
            "top_w_composition_max_log_sigma": MAX_TOP_W_COMPOSITION_LOG_SIGMA,
            "top_w_composition_diagnostics": {
                "counts": {
                    status: sum(
                        record.get("status") == status
                        for record in TOP_W_COMPOSITION_DIAGNOSTICS.values()
                    )
                    for status in sorted(
                        {
                            str(record.get("status"))
                            for record in TOP_W_COMPOSITION_DIAGNOSTICS.values()
                        }
                    )
                },
                "records": TOP_W_COMPOSITION_DIAGNOSTICS,
            },
            "qcd": (
                "one free qcd_norm per Nb x recoil control group; multiple "
                "exclusive SR categories share it"
            ),
            "zinv": (
                "RZ-only nominal Z normalization; Qgamma-only nominal GCR "
                "normalization; one shared GCR-constrained Sgamma shape "
                "rate parameter initialized at the measured Sgamma value"
            ),
            "zinv_free_normalization_rate_parameter": False,
            "sgamma_role": "shape only, shared between matched GCR and Z SR",
            "rz_covariance": rz_covariance["status"],
            "zgamma_nonclosure": "central abs(D-1), Z SR only",
            "highdm_bins": len(exact["highdm"]["search_bin_labels"]),
            "highdm_source_bins": highdm_bin_projection["source_bin_count"],
            "highdm_bin_projection": highdm_bin_projection,
            "lowdm_bins": 34,
            "lowdm_control_connectivity": {
                "model": "Nb x physical recoil interval shared across categories",
                "control_group_count": len(low_control_group_summary),
                "per_sr_bin_control_parameters": False,
                "groups": low_control_group_summary,
            },
            "signal_topology": args.topology,
            "histogram_access": (
                "bounded mmap extraction from canonical hists.json only; "
                "no intermediate ROOT or NanoAOD access"
            ),
            "input_boundary_enforced": True,
            "dropped_empty_control_channels": dropped_empty_control_channels,
        },
        "inputs": {
            label: {
                "path": str(path),
                "sha256": (
                    args.hists_sha256
                    if label == "hists" and args.hists_sha256
                    else sha256(path)
                ),
            }
            for label, path in input_paths.items()
        },
        "template_root": str(template_root),
        "cards": cards,
        "runner": str(runner),
        "limit_validator": str(runner.with_name("validate_limit_outputs.py")),
        "limit_success_gate": (
            "five finite nonnegative expected quantiles with canonical values"
        ),
        "mass_points": masses,
        "extracted_signal_samples": extracted_signals,
        "channels": channel_counts,
        "rate_parameter_count": len(rate_parameters),
        "rate_parameters": rate_parameters,
        "rate_parameter_scopes": scopes,
        "unmatched_rate_parameters_dropped": unmatched_rate_parameters,
        "auto_mc_stats": args.auto_mc_stats,
        "background_grouping": background_grouping_contract(),
        "root_summary": summary,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "channels": channel_counts,
                "rate_parameters": len(rate_parameters),
                "unmatched": unmatched_rate_parameters,
                "mass_points": masses,
                "output_dir": str(output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
