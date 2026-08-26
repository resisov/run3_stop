"""NanoAOD tag-and-probe histogram counting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .config import Expression
from .weights import WeightSet, required_fields


def read_file_list(path: Path | str) -> list[str]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return [str(item["file_path"]) for item in payload.get("records", [])]


def _delta_phi(left: Any, right: Any) -> Any:
    return (left - right + np.pi) % (2.0 * np.pi) - np.pi


def _mass(left: Any, right: Any) -> Any:
    def components(item: Any) -> tuple[Any, Any, Any, Any]:
        px = item.pt * np.cos(item.phi)
        py = item.pt * np.sin(item.phi)
        pz = item.pt * np.sinh(item.eta)
        energy = np.sqrt(np.maximum(0.0, item.mass**2 + px**2 + py**2 + pz**2))
        return px, py, pz, energy

    px1, py1, pz1, e1 = components(left)
    px2, py2, pz2, e2 = components(right)
    squared = (e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    return np.sqrt(np.maximum(0.0, squared))


def _lumimask(path: Path | str) -> dict[int, tuple[tuple[int, int], ...]]:
    payload = json.loads(Path(path).read_text())
    return {
        int(run): tuple((int(low), int(high)) for low, high in ranges)
        for run, ranges in payload.items()
    }


def _good_lumi(
    runs: np.ndarray, lumis: np.ndarray, mask: Mapping[int, Iterable[tuple[int, int]]]
) -> np.ndarray:
    return np.fromiter(
        (
            any(low <= int(lumi) <= high for low, high in mask.get(int(run), ()))
            for run, lumi in zip(runs, lumis)
        ),
        dtype=bool,
        count=len(runs),
    )


def _role_fields(config: Mapping[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(["pt", "eta", "phi", "mass", *map(str, config["fields"])])
    )


def required_branches(config: Mapping[str, Any], sample: str) -> set[str]:
    branches = set(
        config.get("input", {}).get("event_id", ["run", "luminosityBlock", "event"])
    )
    branches.update(config.get("input", {}).get("event_filters", []))
    trigger = config.get("reference_trigger", {})
    applies = bool(
        trigger.get(
            "apply_to_data" if sample == "data" else "apply_to_mc", sample == "data"
        )
    )
    if applies:
        branches.update(trigger.get("paths", []))
    for role in ("tag", "probe", "spectator"):
        if role not in config:
            continue
        collection = str(config[role]["collection"])
        branches.update(f"{collection}_{field}" for field in _role_fields(config[role]))
    if trigger.get("match_tag"):
        branches.update(
            {
                "TrigObj_pt",
                "TrigObj_eta",
                "TrigObj_phi",
                "TrigObj_id",
                "TrigObj_filterBits",
            }
        )
    if sample == "mc":
        branches.update(required_fields(config.get("weights", {})))
    return branches


def _trigger_match(arrays: Any, role: Any, config: Mapping[str, Any]) -> Any:
    import awkward as ak

    trigger = config.get("reference_trigger", {})
    if not trigger.get("match_tag"):
        return ak.ones_like(role["pt"], dtype=bool)
    valid = abs(arrays["TrigObj_id"]) == int(trigger["object_id"])
    bits = trigger.get("filter_bits")
    if bits is not None:
        valid = valid & ((arrays["TrigObj_filterBits"] & int(bits)) != 0)
    deta = role["eta"][:, :, None] - arrays["TrigObj_eta"][:, None, :]
    dphi = _delta_phi(role["phi"][:, :, None], arrays["TrigObj_phi"][:, None, :])
    radius = float(trigger.get("max_delta_r", 0.1))
    return ak.any(
        valid[:, None, :] & (deta * deta + dphi * dphi < radius * radius), axis=2
    )


def _objects(arrays: Any, role_name: str, config: Mapping[str, Any]) -> Any:
    import awkward as ak

    role = config[role_name]
    collection = str(role["collection"])
    values = {field: arrays[f"{collection}_{field}"] for field in _role_fields(role)}
    selected = Expression(str(role["selection"])).evaluate(values)
    records: dict[str, Any] = dict(values)
    records["index"] = ak.local_index(values["pt"], axis=1)
    if role_name == "tag":
        selected = selected & _trigger_match(arrays, values, config)
    if role_name == "probe":
        records["passing"] = Expression(str(role["pass"])).evaluate(values)
        records["bin_pt"] = Expression(str(role["pt"])).evaluate(values)
        records["bin_eta"] = Expression(str(role["eta"])).evaluate(values)
    return ak.zip(records)[selected]


def _pair_context(pairs: Any, fields: Iterable[str]) -> dict[str, Any]:
    context = {
        "mass": _mass(pairs.tag, pairs.probe),
        "delta_r": np.sqrt(
            (pairs.tag.eta - pairs.probe.eta) ** 2
            + _delta_phi(pairs.tag.phi, pairs.probe.phi) ** 2
        ),
    }
    for role in ("tag", "probe"):
        item = pairs[role]
        available = set(item.fields)
        for field in fields:
            if field in available:
                context[f"{role}_{field}"] = item[field]
    return context


def _pairs(arrays: Any, config: Mapping[str, Any]) -> tuple[Any, Any]:
    import awkward as ak

    tags = _objects(arrays, "tag", config)
    probes = _objects(arrays, "probe", config)
    pairs = ak.cartesian({"tag": tags, "probe": probes}, axis=1)
    mask = ak.ones_like(pairs.tag.pt, dtype=bool)
    if config["tag"]["collection"] == config["probe"]["collection"]:
        mask = mask & (pairs.tag.index != pairs.probe.index)
    context = _pair_context(pairs, {"charge", "pt", "eta", "phi"})
    mask = mask & Expression(str(config["pair"].get("selection", "True"))).evaluate(
        context
    )
    if "spectator" in config:
        spectators = _objects(arrays, "spectator", config)
        eligible = ak.ones_like(spectators.pt[:, None, :], dtype=bool)
        if config["spectator"].get("distinct_from_pair", True):
            if config["spectator"]["collection"] == config["tag"]["collection"]:
                eligible = eligible & (
                    spectators.index[:, None, :] != pairs.tag.index[:, :, None]
                )
            if config["spectator"]["collection"] == config["probe"]["collection"]:
                eligible = eligible & (
                    spectators.index[:, None, :] != pairs.probe.index[:, :, None]
                )
        mask = mask & ak.any(eligible, axis=2)
    pairs = pairs[mask]
    masses = _mass(pairs.tag, pairs.probe)
    low, high = map(float, config["pair"]["mass_window_gev"])
    selected = (masses >= low) & (masses <= high)
    return pairs[selected], masses[selected]


def _empty(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    return {
        f"{state}_{moment}": np.zeros(shape, dtype=float)
        for state in ("pass", "fail")
        for moment in ("sumw", "sumw2")
    }


def _fill(
    target: dict[str, np.ndarray],
    pairs: Any,
    masses: Any,
    weights: Any,
    edges: list[np.ndarray],
) -> int:
    import awkward as ak

    _, broadcast = ak.broadcast_arrays(masses, weights)
    values = np.column_stack(
        [
            np.abs(ak.to_numpy(ak.flatten(pairs.probe.bin_eta, axis=1))),
            ak.to_numpy(ak.flatten(pairs.probe.bin_pt, axis=1)),
            ak.to_numpy(ak.flatten(masses, axis=1)),
        ]
    )
    passing = np.asarray(
        ak.to_numpy(ak.flatten(pairs.probe.passing, axis=1)), dtype=bool
    )
    flat_weights = np.asarray(ak.to_numpy(ak.flatten(broadcast, axis=1)), dtype=float)
    for state, mask in (("pass", passing), ("fail", ~passing)):
        target[f"{state}_sumw"] += np.histogramdd(
            values[mask], bins=edges, weights=flat_weights[mask]
        )[0]
        target[f"{state}_sumw2"] += np.histogramdd(
            values[mask], bins=edges, weights=flat_weights[mask] ** 2
        )[0]
    return len(values)


def _serialise(target: Mapping[str, np.ndarray]) -> dict[str, list[list[float]]]:
    return {
        key: values.reshape(-1, values.shape[-1]).astype(float).tolist()
        for key, values in target.items()
    }


def _merge(target: Mapping[str, np.ndarray], source: Mapping[str, np.ndarray]) -> None:
    for key in target:
        target[key] += source[key]


def count_files(
    config: Mapping[str, Any],
    files: Iterable[str],
    sample: str,
    *,
    step_size: int = 100_000,
    base_dir: Path | str = ".",
) -> dict[str, Any]:
    import awkward as ak
    import uproot

    if sample not in {"data", "mc"}:
        raise ValueError("sample must be data or mc")
    eta_edges = np.asarray(config["axes"]["abseta_edges"], dtype=float)
    pt_edges = np.asarray(config["axes"]["pt_edges_gev"], dtype=float)
    low, high = map(float, config["pair"]["mass_window_gev"])
    mass_edges = np.linspace(low, high, int(config["fit"]["mass_bins"]) + 1)
    edges = [eta_edges, pt_edges, mass_edges]
    shape = tuple(len(axis) - 1 for axis in edges)
    weight_set = (
        WeightSet(config.get("weights", {}), base_dir) if sample == "mc" else None
    )
    variation_names = ["nominal"]
    if sample == "mc":
        variation_names.extend(config.get("weights", {}).get("mc_variations", {}))
        variation_names.extend(
            name
            for correction in config.get("weights", {}).get("corrections", [])
            for name in correction.get("variations", {})
        )
    variations = {name: _empty(shape) for name in dict.fromkeys(variation_names)}
    stats = {
        "files_expected": 0,
        "files_processed": 0,
        "files_failed": [],
        "events_read": 0,
        "pairs_selected": 0,
    }
    trigger = config.get("reference_trigger", {})
    apply_trigger = bool(
        trigger.get(
            "apply_to_data" if sample == "data" else "apply_to_mc", sample == "data"
        )
    )
    lumi_path = config.get("lumimask") if sample == "data" else None
    lumi = _lumimask(Path(base_dir) / str(lumi_path)) if lumi_path else None
    seen: set[tuple[int, int, int]] = set()
    requested_base = required_branches(config, sample)
    for file_path in files:
        stats["files_expected"] += 1
        file_variations = {name: _empty(shape) for name in variations}
        file_seen: set[tuple[int, int, int]] = set()
        file_events = 0
        file_pairs = 0
        try:
            tree = uproot.open(str(file_path))[
                str(config.get("input", {}).get("tree", "Events"))
            ]
            present = set(tree.keys())
            paths = [path for path in trigger.get("paths", []) if path in present]
            if apply_trigger and not paths:
                raise RuntimeError("no configured reference trigger is present")
            requested = requested_base - set(trigger.get("paths", [])) | set(paths)
            missing = sorted(requested - present)
            if missing:
                raise RuntimeError(f"required branches missing: {', '.join(missing)}")
            for arrays in tree.iterate(
                sorted(requested), step_size=step_size, library="ak"
            ):
                size = len(arrays["event"])
                file_events += size
                event_mask = np.ones(size, dtype=bool)
                for flag in config.get("input", {}).get("event_filters", []):
                    event_mask &= np.asarray(arrays[flag], dtype=bool)
                if apply_trigger:
                    fired = np.zeros(size, dtype=bool)
                    for path in paths:
                        fired |= np.asarray(arrays[path], dtype=bool)
                    event_mask &= fired
                if sample == "data":
                    runs = np.asarray(arrays["run"])
                    lumis = np.asarray(arrays["luminosityBlock"])
                    events = np.asarray(arrays["event"])
                    if lumi is not None:
                        event_mask &= _good_lumi(runs, lumis, lumi)
                    for index in np.flatnonzero(event_mask):
                        identity = (
                            int(runs[index]),
                            int(lumis[index]),
                            int(events[index]),
                        )
                        if identity in seen or identity in file_seen:
                            event_mask[index] = False
                        else:
                            file_seen.add(identity)
                selected = arrays[event_mask]
                if not len(selected["event"]):
                    continue
                pairs, masses = _pairs(selected, config)
                if sample == "data":
                    event_weights = {
                        "nominal": np.ones(len(selected["event"]), dtype=float)
                    }
                else:
                    event_weights = weight_set.evaluate(selected)
                for name, weight in event_weights.items():
                    if np.isscalar(weight):
                        weight = np.full(
                            len(selected["event"]), float(weight), dtype=float
                        )
                    selected_pairs = _fill(
                        file_variations[name], pairs, masses, ak.Array(weight), edges
                    )
                    if name == "nominal":
                        file_pairs += selected_pairs
            for name, target in variations.items():
                _merge(target, file_variations[name])
            seen.update(file_seen)
            stats["events_read"] += file_events
            stats["pairs_selected"] += file_pairs
            stats["files_processed"] += 1
        except Exception as error:  # noqa: BLE001
            stats["files_failed"].append(
                {"path": str(file_path), "error": f"{type(error).__name__}: {error}"}
            )
    samples = {
        sample if name == "nominal" else f"{sample}__{name}": _serialise(value)
        for name, value in variations.items()
    }
    blockers = (
        []
        if lumi is not None or sample != "data"
        else ["data were counted without a luminosity mask"]
    )
    if stats["files_failed"]:
        blockers.append(f"{len(stats['files_failed'])} ROOT files failed")
    return {
        "schema_version": 1,
        "measurement": config["measurement"],
        "year": str(config["year"]),
        "profile": config.get("profile"),
        "sample": sample,
        "probe_collection": config["probe"]["collection"],
        "probe_selection": config["probe"]["selection"],
        "pass_selection": config["probe"]["pass"],
        "probe_abseta_edges": eta_edges.tolist(),
        "probe_pt_edges_gev": pt_edges.tolist(),
        "mass_edges_gev": mass_edges.tolist(),
        "fit": config["fit"],
        "correction": config["correction"],
        "samples": samples,
        "processing": stats,
        "adoption_blockers": blockers,
        "status": "complete"
        if stats["files_expected"] == stats["files_processed"]
        else "incomplete",
    }
