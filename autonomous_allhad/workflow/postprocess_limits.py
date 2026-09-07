#!/usr/bin/env python3
"""Collect and plot a canonical High-dM + Low-dM limit grid."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_RUNTIME = {
    "python": "3.8.20",
    "numpy": "1.23.5",
    "scipy": "1.10.1",
    "matplotlib": "3.7.3",
    "mplhep": "0.4.1",
    "uproot": "4.3.7",
}


def runtime_versions() -> dict[str, str]:
    actual = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
    }
    for package in ("numpy", "scipy", "matplotlib", "mplhep", "uproot"):
        actual[package] = metadata.version(package)
    return actual


def validate_runtime() -> dict[str, str]:
    actual = runtime_versions()
    mismatches = {
        name: {"expected": expected, "actual": actual.get(name)}
        for name, expected in CANONICAL_RUNTIME.items()
        if actual.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "limit plotting runtime does not match the canonical EOS py38 "
            f"environment: {json.dumps(mismatches, sort_keys=True)}"
        )
    return actual


REPOSITORY = Path(__file__).resolve().parents[2]
RUN2_REFERENCE_DIR = REPOSITORY / "autonomous_allhad" / "gnn_lowdm" / "inputs"
DEFAULT_STOP_XSEC = (
    REPOSITORY / "autonomous_allhad" / "signals" / "stop_xsec_13p6TeV.json"
)

# The canonical limit plotter must be self-contained and reproducible.  Keep
# every SUS-19-010 overlay in the repository instead of selecting an EOS file
# or a result-directory sidecar at runtime.
RUN2_CONTOURS = {
    topology: RUN2_REFERENCE_DIR / f"run2_sus19010_{topology.lower()}_contours.json"
    for topology in ("T2tt", "T2tb", "T2bW")
}

RUN2_COMPRESSED_CONTOURS = {
    "T2tt": RUN2_REFERENCE_DIR / "run2_sus19010_t2ttc_contours.json",
    "T2tb": None,
    "T2bW": RUN2_REFERENCE_DIR / "run2_sus19010_t2bwc_contours.json",
}

DECAY_LABELS = {
    "T2tt": (
        r"$pp\rightarrow \tilde{t}_1\bar{\tilde{t}}_1,\ "
        r"\tilde{t}_1\rightarrow t\tilde{\chi}_1^0$"
    ),
    "T2tb": (
        r"$pp\rightarrow \tilde{t}_1\bar{\tilde{t}}_1,\ "
        r"\tilde{t}_1\rightarrow b\tilde{\chi}_1^+"
        r"\rightarrow bW^{+*}\tilde{\chi}_1^0\ (50\%),$"
        "\n"
        r"$"
        r"\bar{\tilde{t}}_1\rightarrow\bar{t}\tilde{\chi}_1^0\ (50\%)"
        r"$"
    ),
    "T2bW": (
        r"$pp\rightarrow \tilde{t}_1\bar{\tilde{t}}_1,\ "
        r"\tilde{t}_1\rightarrow b\tilde{\chi}_1^+"
        r"\rightarrow bW^+\tilde{\chi}_1^0$"
    ),
}

LUMINOSITY_LABELS = {
    "2024": r"109.82 fb$^{-1}$ (13.6 TeV)",
    "2025": r"110.84 fb$^{-1}$ (13.6 TeV)",
    "2024_2025": r"220.66 fb$^{-1}$ (13.6 TeV)",
}


# Interpolation boundaries are expressed in Delta m = m(stop) - m(LSP).
# Keeping these definitions next to the canonical plotter makes the physics
# choice explicit and leaves the historical global interpolation as the
# default, rollback-safe behaviour.
INTERPOLATION_REGIME_BOUNDARIES = {
    "T2tt": (80.4 + 4.8, 172.5),
    "T2bW": (2.0 * 80.4,),
    "T2tb": (80.4 + 4.8, 2.0 * 80.4, 172.5),
}

TOP_MASS_GEV = 172.5
W_MASS_GEV = 80.4
B_MASS_GEV = 4.8
T2TB_TOP_BRANCH_WEIGHT = 0.5


def signed_two_body_phase_space(
    parent_mass: np.ndarray,
    visible_mass: float,
    invisible_mass: np.ndarray,
) -> np.ndarray:
    """Return a dimensionless signed Kallen phase-space coordinate."""

    parent = np.asarray(parent_mass, dtype=float)
    invisible = np.asarray(invisible_mass, dtype=float)
    parent, invisible = np.broadcast_arrays(parent, invisible)
    coordinate = np.full(parent.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(parent)
        & np.isfinite(invisible)
        & (parent > 0.0)
        & (invisible >= 0.0)
    )
    first_factor = parent[valid] ** 2 - (
        float(visible_mass) + invisible[valid]
    ) ** 2
    second_factor = parent[valid] ** 2 - (
        float(visible_mass) - invisible[valid]
    ) ** 2
    kallen = first_factor * second_factor
    coordinate[valid] = (
        np.sign(kallen) * np.sqrt(np.abs(kallen)) / parent[valid] ** 2
    )
    return coordinate


def topology_phase_space_coordinate(
    topology: str,
    stop_mass: np.ndarray,
    lsp_mass: np.ndarray,
    *,
    top_branch_weight: float = T2TB_TOP_BRANCH_WEIGHT,
) -> np.ndarray:
    """Return the topology-specific continuous interpolation coordinate."""

    stop = np.asarray(stop_mass, dtype=float)
    lsp = np.asarray(lsp_mass, dtype=float)
    stop, lsp = np.broadcast_arrays(stop, lsp)
    top_coordinate = signed_two_body_phase_space(
        stop, TOP_MASS_GEV, lsp
    )
    chargino = 0.5 * (stop + lsp)
    w_coordinate = signed_two_body_phase_space(
        chargino, W_MASS_GEV, lsp
    )
    if topology == "T2tt":
        return top_coordinate
    if topology == "T2bW":
        return w_coordinate
    if topology == "T2tb":
        if not 0.0 <= top_branch_weight <= 1.0:
            raise ValueError("T2tb top-branch weight must lie in [0, 1]")
        return (
            top_branch_weight * top_coordinate
            + (1.0 - top_branch_weight) * w_coordinate
        )
    raise ValueError(f"unsupported topology: {topology}")


def load_excluded_points(input_dir: Path) -> set[str]:
    """Read explicit user-policy exclusions for this topology."""

    excluded_points: set[str] = set()
    exclusion_dir = input_dir / "excluded_points"
    central_policy = (
        REPOSITORY
        / "autonomous_allhad"
        / "configs"
        / "limit_point_exclusions.json"
    )
    policy_paths = sorted(
        exclusion_dir.glob("limit_point_exclusions_*.json")
    )
    if central_policy.is_file():
        policy_paths.append(central_policy)
    for path in policy_paths:
        payload = json.loads(path.read_text())
        for record in payload.get("exclusions", []):
            if record.get("status") not in {
                "excluded_by_user",
                "excluded_by_user_policy",
            }:
                continue
            if record.get("model") != input_dir.name:
                continue
            excluded_points.add(
                f"mStop{int(record['mStop_GeV'])}_mLSP{int(record['mLSP_GeV'])}"
            )
    return excluded_points


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_thermal_relic_contour(
    path: Path,
    level: float = 0.12,
) -> dict[str, Any]:
    """Load one audited relic-density contour from the shared machine data."""

    payload = read_json(path)
    matching = [
        entry
        for entry in payload.get("relic_contours", [])
        if math.isclose(
            float(entry.get("omega_h2", math.nan)),
            float(level),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"expected exactly one Omega*h^2={level:g} contour in {path}; "
            f"found {len(matching)}"
        )
    points = np.asarray(matching[0].get("points") or [], dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] < 2
        or points.shape[1] < 2
        or not np.all(np.isfinite(points[:, :2]))
    ):
        raise RuntimeError(f"invalid thermal-relic contour points in {path}")
    return {
        "omega_h2": float(level),
        "points": points[:, :2],
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": payload.get("model"),
        "sources": payload.get("sources"),
        "overabundance_definition": "Omega_LSP*h^2 > 0.12",
        "overabundance_side": (
            "a narrow one-sided hatched strip follows the published "
            "Omega*h^2=0.12 contour on the Omega*h^2 > 0.12 side"
        ),
    }


def thermal_overabundance_strips(
    contour_points: np.ndarray,
    width_GeV: float = 18.0,
) -> list[np.ndarray]:
    """Build thin normal-offset strips on the over-abundant contour side."""

    points = np.asarray(contour_points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        raise ValueError("thermal-relic contour must be an N-by-2 array")
    points = points[:, :2]
    turn = int(np.argmin(points[:, 0]))
    branches = (points[: turn + 1], points[turn:])
    strips = []
    for branch in branches:
        if len(branch) < 2:
            continue
        tangent = np.gradient(branch, axis=0)
        length = np.hypot(tangent[:, 0], tangent[:, 1])
        valid = length > 0.0
        if not np.all(valid):
            tangent = tangent[valid]
            branch = branch[valid]
            length = length[valid]
        if len(branch) < 2:
            continue
        # The source curve runs from the upper coannihilation branch into the
        # turn and then out along the lower asymptotic branch.  Its left-hand
        # normal points toward Omega*h^2 > 0.12 on both monotonic branches.
        left_normal = np.column_stack(
            (-tangent[:, 1] / length, tangent[:, 0] / length)
        )
        shifted = branch + float(width_GeV) * left_normal
        strips.append(np.vstack((branch, shifted[::-1])))
    return strips


def load_stop_pair_xsecs(path: Path) -> dict[int, float]:
    payload = read_json(path)
    if (
        payload.get("schema_version") != "stop_pair_xsec_13p6tev_v1"
        or payload.get("parsed") is not True
    ):
        raise RuntimeError(f"invalid stop-pair cross-section table: {path}")
    xsecs: dict[int, float] = {}
    for record in payload.get("records", []):
        mass = int(record["mStop"])
        value = float(record["xsec_pb"])
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"invalid stop-pair cross section at {mass} GeV")
        xsecs[mass] = value
    if not xsecs:
        raise RuntimeError(f"empty stop-pair cross-section table: {path}")
    return xsecs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_mass_key(key: str) -> tuple[int, int]:
    match = re.fullmatch(r"mStop(\d+)_mLSP(\d+)", key)
    if not match:
        raise ValueError(f"invalid mass key: {key}")
    return int(match.group(1)), int(match.group(2))


def parse_limit_file(path: Path) -> dict[str, float] | None:
    rows = []
    try:
        import uproot

        with uproot.open(path) as root_file:
            tree = root_file.get("limit")
            if tree is not None:
                quantiles = tree["quantileExpected"].array(library="np")
                limits = tree["limit"].array(library="np")
                rows = [
                    (float(quantile), float(value))
                    for quantile, value in zip(quantiles, limits)
                ]
    except Exception:
        rows = []
    if not rows:
        try:
            import ROOT

            root_file = ROOT.TFile.Open(str(path))
            tree = root_file.Get("limit") if root_file else None
            if not tree:
                return None
            for entry in tree:
                rows.append(
                    (float(entry.quantileExpected), float(entry.limit))
                )
            root_file.Close()
        except Exception:
            return None
    if not rows:
        return None
    labels = {
        0.025: "expected_m2",
        0.16: "expected_m1",
        0.5: "expected",
        0.84: "expected_p1",
        0.975: "expected_p2",
    }
    if len(rows) != len(labels):
        return None
    output = {}
    for expected_quantile, label in labels.items():
        matches = [
            value
            for quantile, value in rows
            if math.isclose(
                quantile,
                expected_quantile,
                rel_tol=0.0,
                abs_tol=1.0e-5,
            )
        ]
        if (
            len(matches) != 1
            or not math.isfinite(matches[0])
            or matches[0] < 0.0
        ):
            return None
        output[label] = matches[0]
    return output


def collect_limits(
    limit_dir: Path, mass_keys: list[str], output_json: Path
) -> dict[str, Any]:
    results = {}
    for mass_key in mass_keys:
        candidate_paths = (
            sorted(
                limit_dir.glob(
                    f"higgsCombine_{mass_key}.AsymptoticLimits*.root"
                )
            )
            + sorted(limit_dir.glob(f"higgsCombine_{mass_key}.root"))
            + sorted(limit_dir.glob(f"higgsCombine_{mass_key}*.root"))
        )
        # The broad legacy fallback must not allow a shorter LSP mass token to
        # consume a different point (for example mLSP100 matching mLSP1000).
        exact_mass_prefix = re.compile(
            rf"^higgsCombine_{re.escape(mass_key)}(?![0-9])"
        )
        candidates = []
        seen = set()
        for path in candidate_paths:
            if path in seen or not exact_mass_prefix.match(path.name):
                continue
            seen.add(path)
            candidates.append(path)
        parsed = None
        for path in candidates:
            parsed = parse_limit_file(path)
            if parsed:
                break
        if parsed:
            mstop, mlsp = parse_mass_key(mass_key)
            parsed.update({"mStop": mstop, "mLSP": mlsp})
            results[mass_key] = parsed
    status = (
        "complete"
        if len(results) == len(mass_keys)
        else "partial"
        if results
        else "no_combine_outputs"
    )
    payload = {
        "status": status,
        "points": results,
        "requested_point_count": len(mass_keys),
        "collected_point_count": len(results),
        "missing_points": [key for key in mass_keys if key not in results],
    }
    write_json(output_json, payload)
    return payload


def select_limit_range(
    payload: dict[str, Any], mass_keys: list[str]
) -> dict[str, Any]:
    """Return the audited subset used by the requested plot axes."""

    points = payload.get("points") or {}
    selected = {key: points[key] for key in mass_keys if key in points}
    missing = [key for key in mass_keys if key not in selected]
    return {
        "status": (
            "complete"
            if not missing
            else "partial"
            if selected
            else "no_combine_outputs"
        ),
        "points": selected,
        "requested_point_count": len(mass_keys),
        "collected_point_count": len(selected),
        "missing_points": missing,
    }


def plot_contour(
    limit_payload: dict[str, Any],
    output_png: Path,
    run2_contours: Path | None,
    luminosity_label: str,
    analysis_label: str | None,
    x_max: float,
    y_min: float,
    y_max: float,
    decay_label: str | None,
    color_field: str = "ratio",
    stop_pair_xsecs: dict[int, float] | None = None,
    mask_offshell: bool = False,
    run2_compressed_contours: Path | None = None,
    topology: str = "T2tt",
    interpolation_mode: str = "global",
    interpolation_diagnostics: dict[str, Any] | None = None,
    overlay_limit_payload: dict[str, Any] | None = None,
    primary_legend_label: str | None = None,
    overlay_legend_label: str | None = None,
    thermal_relic: dict[str, Any] | None = None,
    x_min: float = 500.0,
) -> bool:
    records = list((limit_payload.get("points") or {}).values())
    points = [
        record
        for record in records
        if "expected" in record and float(record["expected"]) > 0
    ]
    unique_points = {
        (float(record["mStop"]), float(record["mLSP"]))
        for record in points
    }
    if len(unique_points) < 4:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MultipleLocator
    from scipy.interpolate import griddata

    xmin, xmax = float(x_min), float(x_max)
    ymin, ymax = float(y_min), float(y_max)
    top_mass = TOP_MASS_GEV
    w_plus_b_mass = W_MASS_GEV + B_MASS_GEV
    final_offshell_boundary = (
        B_MASS_GEV if topology == "T2tt" else 2.0 * B_MASS_GEV
    )
    primary_offshell_boundary = (
        2.0 * W_MASS_GEV if topology == "T2bW" else TOP_MASS_GEV
    )
    xi = np.linspace(xmin, xmax, 260)
    yi = np.linspace(ymin, ymax, 260)
    xx, yy = np.meshgrid(xi, yi)
    offshell_mask = (
        yy > (xx - top_mass)
        if mask_offshell
        else np.zeros_like(xx, dtype=bool)
    )

    if interpolation_mode not in {"global", "regime", "topology-aware"}:
        raise ValueError(
            f"unsupported interpolation mode: {interpolation_mode}"
        )
    regime_boundaries = INTERPOLATION_REGIME_BOUNDARIES[topology]
    delta_grid = xx - yy
    lower_boundary_bridge_diagnostics: dict[str, Any] = {}

    def interpolation_values(
        quantity: str,
        scale_by_stop_xsec: bool = False,
        source_records: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = []
        for record in records if source_records is None else source_records:
            value = record.get(quantity)
            if value is None or float(value) <= 0:
                continue
            value = float(value)
            if scale_by_stop_xsec:
                mstop = int(record["mStop"])
                if stop_pair_xsecs is None or mstop not in stop_pair_xsecs:
                    raise RuntimeError(
                        f"missing stop-pair cross section at {mstop} GeV"
                    )
                value *= stop_pair_xsecs[mstop]
            values.append(
                (
                    float(record["mStop"]),
                    float(record["mLSP"]),
                    math.log10(value),
                )
            )
        if not values:
            return np.asarray([]), np.asarray([]), np.asarray([])
        xs = np.asarray([value[0] for value in values])
        ys = np.asarray([value[1] for value in values])
        zs = np.asarray([value[2] for value in values])
        return xs, ys, zs

    def global_log_grid(
        quantity: str,
        scale_by_stop_xsec: bool = False,
        source_records: list[dict[str, Any]] | None = None,
    ) -> np.ma.MaskedArray | None:
        xs, ys, zs = interpolation_values(
            quantity, scale_by_stop_xsec, source_records
        )
        if not xs.size:
            return None
        linear = griddata((xs, ys), zs, (xx, yy), method="linear")
        return np.ma.array(linear, mask=np.isnan(linear) | offshell_mask)

    def has_two_dimensional_support(
        first: np.ndarray,
        second: np.ndarray,
    ) -> bool:
        if first.size < 3:
            return False
        coordinates = np.column_stack((first, second))
        return bool(
            np.linalg.matrix_rank(
                coordinates - np.mean(coordinates, axis=0)
            ) >= 2
        )

    def regime_log_grids(
        quantity: str,
        scale_by_stop_xsec: bool = False,
        source_records: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ma.MaskedArray | None, list[np.ma.MaskedArray]]:
        xs, ys, zs = interpolation_values(
            quantity, scale_by_stop_xsec, source_records
        )
        if not xs.size:
            return None, []
        deltas = xs - ys
        point_regimes = np.searchsorted(
            regime_boundaries, deltas, side="right"
        )
        target_regimes = np.searchsorted(
            regime_boundaries, delta_grid, side="right"
        )
        combined = np.full_like(xx, np.nan, dtype=float)
        component_grids: list[np.ma.MaskedArray] = []
        for regime_index in range(len(regime_boundaries) + 1):
            select = point_regimes == regime_index
            if not has_two_dimensional_support(xs[select], deltas[select]):
                continue
            linear = griddata(
                (xs[select], deltas[select]),
                zs[select],
                (xx, delta_grid),
                method="linear",
            )
            valid = (
                np.isfinite(linear)
                & (target_regimes == regime_index)
                & ~offshell_mask
            )
            combined[valid] = linear[valid]
            component_grids.append(np.ma.array(linear, mask=~valid))
        return (
            np.ma.array(combined, mask=np.isnan(combined) | offshell_mask),
            component_grids,
        )

    def topology_interpolation_coordinates(
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        top_branch_weight: float = T2TB_TOP_BRANCH_WEIGHT,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        deltas = xs - ys
        phase_space = topology_phase_space_coordinate(
            topology,
            xs,
            ys,
            top_branch_weight=top_branch_weight,
        )
        valid = (
            np.isfinite(xs)
            & np.isfinite(phase_space)
            & (
                deltas
                > (
                    W_MASS_GEV + B_MASS_GEV
                    if topology == "T2tt"
                    else 2.0 * B_MASS_GEV
                )
            )
        )
        coordinate_name = (
            "signed_top_phase_space"
            if topology == "T2tt"
            else (
                "signed_W_phase_space"
                if topology == "T2bW"
                else "branch_weighted_signed_phase_space"
            )
        )
        return xs, phase_space, valid, coordinate_name

    def augmented_topology_inputs(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        *,
        top_branch_weight: float = T2TB_TOP_BRANCH_WEIGHT,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        first, second, valid, _ = topology_interpolation_coordinates(
            xs,
            ys,
            top_branch_weight=top_branch_weight,
        )
        interpolation_first = first[valid]
        interpolation_second = second[valid]
        interpolation_values_array = zs[valid]
        if not np.any(valid):
            return (
                interpolation_first,
                interpolation_second,
                interpolation_values_array,
                valid,
                0,
            )

        minimum_lsp = float(np.min(ys[valid]))
        lower_boundary = valid & np.isclose(ys, minimum_lsp)
        boundary_x = xs[lower_boundary]
        boundary_z = zs[lower_boundary]
        if boundary_x.size < 2:
            return (
                interpolation_first,
                interpolation_second,
                interpolation_values_array,
                valid,
                0,
            )
        order = np.argsort(boundary_x)
        boundary_x = boundary_x[order]
        boundary_z = boundary_z[order]
        unique_x, unique_indices = np.unique(boundary_x, return_index=True)
        boundary_x = unique_x
        boundary_z = boundary_z[unique_indices]
        dense_x = xi[
            (xi >= boundary_x[0])
            & (xi <= boundary_x[-1])
            & np.asarray(
                [
                    np.min(np.abs(boundary_x - candidate)) > 1.0e-9
                    for candidate in xi
                ]
            )
        ]
        if not dense_x.size:
            return (
                interpolation_first,
                interpolation_second,
                interpolation_values_array,
                valid,
                0,
            )
        dense_y = np.full_like(dense_x, minimum_lsp)
        dense_z = np.interp(dense_x, boundary_x, boundary_z)
        dense_first, dense_second, dense_valid, _ = (
            topology_interpolation_coordinates(
                dense_x,
                dense_y,
                top_branch_weight=top_branch_weight,
            )
        )
        interpolation_first = np.concatenate(
            (interpolation_first, dense_first[dense_valid])
        )
        interpolation_second = np.concatenate(
            (interpolation_second, dense_second[dense_valid])
        )
        interpolation_values_array = np.concatenate(
            (interpolation_values_array, dense_z[dense_valid])
        )
        return (
            interpolation_first,
            interpolation_second,
            interpolation_values_array,
            valid,
            int(np.count_nonzero(dense_valid)),
        )

    def topology_log_grids(
        quantity: str,
        scale_by_stop_xsec: bool = False,
        *,
        top_branch_weight: float = T2TB_TOP_BRANCH_WEIGHT,
        source_records: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ma.MaskedArray | None, list[np.ma.MaskedArray]]:
        xs, ys, zs = interpolation_values(
            quantity, scale_by_stop_xsec, source_records
        )
        if not xs.size:
            return None, []
        first, second, interpolation_zs, valid_input, boundary_anchor_count = (
            augmented_topology_inputs(
                xs,
                ys,
                zs,
                top_branch_weight=top_branch_weight,
            )
        )
        target_first, target_second, valid_target, _ = (
            topology_interpolation_coordinates(
                xx,
                yy,
                top_branch_weight=top_branch_weight,
            )
        )
        if not has_two_dimensional_support(
            first, second
        ):
            return None, []
        linear = griddata(
            (first, second),
            interpolation_zs,
            (target_first, target_second),
            method="linear",
            rescale=True,
        )

        # The dimensionless phase-space coordinate has a shallow fold very
        # close to mLSP=0 for the midpoint-chargino models.  Consequently the
        # real mLSP=1 boundary row can be finite while the immediately
        # adjacent rows fall just outside the Delaunay hull.  Bridge only that
        # bounded internal gap, in log-limit space, between the real boundary
        # value and the first supported row in the same mStop column.  This is
        # not extrapolation beyond the generated mass grid.
        bridged_cells = 0
        bridge_max_mlsp = None
        for column in range(linear.shape[1]):
            if not math.isfinite(float(linear[0, column])):
                continue
            row = 1
            while (
                row < linear.shape[0]
                and valid_target[row, column]
                and not math.isfinite(float(linear[row, column]))
            ):
                row += 1
            if row <= 1 or row >= linear.shape[0]:
                continue
            if not math.isfinite(float(linear[row, column])):
                continue
            missing_rows = np.arange(1, row, dtype=int)
            fractions = (
                (yi[missing_rows] - yi[0]) / (yi[row] - yi[0])
            )
            linear[missing_rows, column] = (
                linear[0, column]
                + fractions
                * (linear[row, column] - linear[0, column])
            )
            bridged_cells += int(missing_rows.size)
            column_max = float(yi[missing_rows[-1]])
            bridge_max_mlsp = (
                column_max
                if bridge_max_mlsp is None
                else max(bridge_max_mlsp, column_max)
            )

        if (
            quantity == "expected"
            and not scale_by_stop_xsec
            and top_branch_weight == T2TB_TOP_BRANCH_WEIGHT
            and source_records is None
        ):
            lower_boundary_bridge_diagnostics.update(
                {
                    "generated_boundary_mLSP_GeV": float(
                        np.min(ys[valid_input])
                    ),
                    "dense_boundary_anchor_count": boundary_anchor_count,
                    "bridged_grid_cell_count": bridged_cells,
                    "maximum_bridged_mLSP_GeV": bridge_max_mlsp,
                    "interpolation_space": "log10_expected_limit",
                    "policy": (
                        "linear bridge between the generated lower-boundary "
                        "row and the first Delaunay-supported row"
                    ),
                }
            )
        valid = np.isfinite(linear) & valid_target & ~offshell_mask
        grid = np.ma.array(linear, mask=~valid)
        return grid, [grid]

    def interpolated_log_grids(
        quantity: str,
        scale_by_stop_xsec: bool = False,
        source_records: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ma.MaskedArray | None, list[np.ma.MaskedArray]]:
        if interpolation_mode == "regime":
            return regime_log_grids(
                quantity, scale_by_stop_xsec, source_records
            )
        if interpolation_mode == "topology-aware":
            return topology_log_grids(
                quantity,
                scale_by_stop_xsec,
                source_records=source_records,
            )
        grid = global_log_grid(
            quantity, scale_by_stop_xsec, source_records
        )
        return grid, [] if grid is None else [grid]

    expected_ratio_grid, expected_ratio_components = interpolated_log_grids(
        "expected"
    )
    if expected_ratio_grid is None or expected_ratio_grid.count() == 0:
        return False
    minus1_grid, minus1_components = interpolated_log_grids("expected_m1")
    plus1_grid, plus1_components = interpolated_log_grids("expected_p1")
    overlay_expected_components: list[np.ma.MaskedArray] = []
    overlay_minus1_components: list[np.ma.MaskedArray] = []
    overlay_plus1_components: list[np.ma.MaskedArray] = []
    if overlay_limit_payload is not None:
        overlay_records = list(
            (overlay_limit_payload.get("points") or {}).values()
        )
        _, overlay_expected_components = interpolated_log_grids(
            "expected", source_records=overlay_records
        )
        _, overlay_minus1_components = interpolated_log_grids(
            "expected_m1", source_records=overlay_records
        )
        _, overlay_plus1_components = interpolated_log_grids(
            "expected_p1", source_records=overlay_records
        )
    color_grid, _ = (
        interpolated_log_grids("expected", scale_by_stop_xsec=True)
        if color_field == "xsec"
        else (expected_ratio_grid, expected_ratio_components)
    )
    if color_grid is None or color_grid.count() == 0:
        return False

    def compare_log_grids(
        reference_grid: np.ma.MaskedArray,
        candidate_grid: np.ma.MaskedArray,
    ) -> dict[str, Any] | None:
        overlap = (
            ~np.ma.getmaskarray(reference_grid)
            & ~np.ma.getmaskarray(candidate_grid)
        )
        if not np.any(overlap):
            return None
        reference_values = np.asarray(reference_grid)[overlap]
        candidate_values = np.asarray(candidate_grid)[overlap]
        return {
            "overlap_grid_cells": int(np.count_nonzero(overlap)),
            "exclusion_classification_disagreement_fraction": float(
                np.mean(
                    (reference_values <= 0.0)
                    != (candidate_values <= 0.0)
                )
            ),
            "median_absolute_log10_difference": float(
                np.median(np.abs(reference_values - candidate_values))
            ),
            "maximum_absolute_log10_difference": float(
                np.max(np.abs(reference_values - candidate_values))
            ),
        }

    def leave_one_out_diagnostics() -> dict[str, Any]:
        xs, ys, zs = interpolation_values("expected")
        first, second, valid, coordinate_name = (
            topology_interpolation_coordinates(xs, ys)
        )
        predictions = []
        for index in np.flatnonzero(valid):
            training = valid.copy()
            training[index] = False
            if not has_two_dimensional_support(
                first[training], second[training]
            ):
                continue
            try:
                prediction = griddata(
                    (first[training], second[training]),
                    zs[training],
                    (
                        np.asarray([first[index]]),
                        np.asarray([second[index]]),
                    ),
                    method="linear",
                    rescale=True,
                )
            except Exception:
                continue
            prediction_value = float(np.asarray(prediction).reshape(-1)[0])
            if not math.isfinite(prediction_value):
                continue
            predictions.append(
                {
                    "absolute_log10_error": abs(prediction_value - zs[index]),
                    "delta_m_GeV": float(xs[index] - ys[index]),
                }
            )

        def summarize(selected: list[dict[str, float]]) -> dict[str, Any]:
            errors = np.asarray(
                [entry["absolute_log10_error"] for entry in selected],
                dtype=float,
            )
            if not errors.size:
                return {"predicted_point_count": 0}
            return {
                "predicted_point_count": int(errors.size),
                "median_absolute_log10_error": float(np.median(errors)),
                "p90_absolute_log10_error": float(np.quantile(errors, 0.90)),
                "maximum_absolute_log10_error": float(np.max(errors)),
            }

        boundary_predictions = [
            entry
            for entry in predictions
            if min(
                abs(entry["delta_m_GeV"] - boundary)
                for boundary in regime_boundaries
            ) <= 50.0
        ]
        return {
            "coordinate": coordinate_name,
            "eligible_input_point_count": int(np.count_nonzero(valid)),
            "excluded_input_point_count": int(np.count_nonzero(~valid)),
            "all_predictable_points": summarize(predictions),
            "within_50_GeV_of_a_threshold": summarize(
                boundary_predictions
            ),
        }

    if interpolation_diagnostics is not None:
        expected_xs, expected_ys, _ = interpolation_values("expected")
        expected_deltas = expected_xs - expected_ys
        point_regimes = np.searchsorted(
            regime_boundaries, expected_deltas, side="right"
        )
        interval_edges = (-math.inf, *regime_boundaries, math.inf)
        regimes = []
        for index in range(len(interval_edges) - 1):
            lower, upper = interval_edges[index:index + 2]
            select = point_regimes == index
            regimes.append(
                {
                    "index": index,
                    "delta_m_min_GeV": None if math.isinf(lower) else lower,
                    "delta_m_max_GeV": None if math.isinf(upper) else upper,
                    "point_count": int(np.count_nonzero(select)),
                    "interpolated": has_two_dimensional_support(
                        expected_xs[select], expected_deltas[select]
                    ),
                }
            )
        coordinate_system = ["mStop", "mLSP"]
        topology_policy = None
        if interpolation_mode == "regime":
            coordinate_system = ["mStop", "delta_m"]
        elif interpolation_mode == "topology-aware":
            if topology == "T2tt":
                coordinate_system = ["mStop", "signed_top_phase_space"]
                topology_policy = {
                    "method": "continuous_signed_phase_space",
                    "threshold_GeV": TOP_MASS_GEV,
                    "threshold_stitching": "C0",
                    "input_requirement": "delta_m > mW + mb",
                    "final_offshell_boundary_GeV": final_offshell_boundary,
                }
            elif topology == "T2bW":
                coordinate_system = ["mStop", "signed_W_phase_space"]
                topology_policy = {
                    "method": "continuous_signed_phase_space",
                    "threshold_GeV": 2.0 * W_MASS_GEV,
                    "threshold_stitching": "C0",
                    "chargino_mass": "0.5 * (mStop + mLSP)",
                    "final_offshell_boundary_GeV": final_offshell_boundary,
                }
            else:
                coordinate_system = [
                    "mStop",
                    "branch_weighted_signed_phase_space",
                ]
                topology_policy = {
                    "method": "continuous_branch_weighted_signed_phase_space",
                    "top_branch_weight": T2TB_TOP_BRANCH_WEIGHT,
                    "W_branch_weight": 1.0 - T2TB_TOP_BRANCH_WEIGHT,
                    "thresholds_GeV": [
                        2.0 * W_MASS_GEV,
                        TOP_MASS_GEV,
                    ],
                    "threshold_stitching": "C0_at_each_branch_threshold",
                    "chargino_mass": "0.5 * (mStop + mLSP)",
                    "final_offshell_boundary_GeV": final_offshell_boundary,
                }
        interpolation_diagnostics.update(
            {
                "mode": interpolation_mode,
                "coordinate_system": coordinate_system,
                "delta_m_definition": "mStop - mLSP",
                "boundaries_GeV": list(regime_boundaries),
                "regimes": regimes,
                "topology_policy": topology_policy,
                "extrapolation": "none",
                "valid_interpolated_grid_cells": int(
                    expected_ratio_grid.count()
                ),
                "lower_boundary_bridge": (
                    lower_boundary_bridge_diagnostics
                    if interpolation_mode == "topology-aware"
                    else None
                ),
            }
        )
        if interpolation_mode in {"regime", "topology-aware"}:
            global_grid = global_log_grid("expected")
            if global_grid is not None:
                global_comparison = compare_log_grids(
                    global_grid, expected_ratio_grid
                )
                if global_comparison is not None:
                    interpolation_diagnostics["global_comparison"] = (
                        global_comparison
                    )
        if interpolation_mode == "topology-aware" and color_field == "ratio":
            interpolation_diagnostics["leave_one_out"] = (
                leave_one_out_diagnostics()
            )
            if topology == "T2tb":
                weight_variations = {}
                for top_branch_weight in (0.25, 0.75):
                    varied_grid, _ = topology_log_grids(
                        "expected",
                        top_branch_weight=top_branch_weight,
                    )
                    if varied_grid is None:
                        continue
                    variation = compare_log_grids(
                        expected_ratio_grid, varied_grid
                    )
                    if variation is not None:
                        weight_variations[
                            f"top_{top_branch_weight:.2f}_W_"
                            f"{1.0 - top_branch_weight:.2f}"
                        ] = variation
                interpolation_diagnostics["branch_weight_variations"] = (
                    weight_variations
                )

    hep.style.use("CMS")
    figure, axes = plt.subplots(figsize=(12.0, 10.0))
    figure.subplots_adjust(left=0.13, right=0.84, bottom=0.11, top=0.90)
    color_min, color_max = (
        (-4.0, -1.0) if color_field == "xsec" else (-1.5, 1.5)
    )
    limit_cmap = LinearSegmentedColormap.from_list(
        "cms_limit_reference",
        [
            "#5965f2",
            "#62a9ff",
            "#55d7f2",
            "#7ef0c9",
            "#d7fb80",
            "#fff176",
            "#ffb45e",
            "#ff6f6f",
        ],
        N=256,
    )
    plot_grid = np.ma.clip(color_grid, color_min, color_max)
    filled = axes.contourf(
        xx,
        yy,
        plot_grid,
        levels=np.linspace(color_min, color_max, 121),
        cmap=limit_cmap,
        extend="both",
    )
    colorbar = figure.colorbar(
        filled, ax=axes, pad=0.04, fraction=0.048, aspect=34
    )
    if color_field == "xsec":
        colorbar.set_label(
            "Expected 95% CL limit on cross section (pb)",
            fontsize=27,
            rotation=90,
            labelpad=22,
        )
        colorbar.set_ticks(np.arange(-4.0, -0.99, 1.0))
        colorbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: rf"$10^{{{int(value)}}}$")
        )
        colorbar.ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    else:
        colorbar.set_label(
            r"$\log_{10}$ (expected 95% CL limit on $\sigma/\sigma_{\mathrm{theory}}$)",
            fontsize=27,
            rotation=90,
            labelpad=22,
        )
        colorbar.set_ticks(np.arange(color_min, color_max + 0.001, 0.5))
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        colorbar.ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    colorbar.ax.tick_params(
        which="major", labelsize=23, direction="in", length=12, width=1.4
    )
    colorbar.ax.tick_params(
        which="minor", direction="in", length=7, width=1.1
    )
    colorbar.outline.set_linewidth(1.8)

    diagonal_x = np.linspace(xmin, xmax, 400)
    for mass_gap, linestyle in (
        (primary_offshell_boundary, ":"),
        (w_plus_b_mass, "--"),
        (final_offshell_boundary, "-."),
    ):
        diagonal_y = diagonal_x - mass_gap
        keep = (diagonal_y >= ymin) & (diagonal_y <= ymax)
        axes.plot(
            diagonal_x[keep],
            diagonal_y[keep],
            color="0.45",
            linestyle=linestyle,
            linewidth=1.1,
            zorder=4,
        )
    thermal_relic_handle = None
    if thermal_relic is not None:
        relic_points = np.asarray(thermal_relic["points"], dtype=float)
        for overabundance_strip in thermal_overabundance_strips(relic_points):
            axes.fill(
                overabundance_strip[:, 0],
                overabundance_strip[:, 1],
                facecolor="none",
                edgecolor="#666666",
                hatch="///",
                linewidth=0.0,
                zorder=3.0,
            )
        axes.plot(
            relic_points[:, 0],
            relic_points[:, 1],
            color="#666666",
            linestyle="-",
            linewidth=2.4,
            zorder=8.5,
        )
        thermal_relic_handle = Patch(
            facecolor="white",
            edgecolor="#666666",
            hatch="///",
            linewidth=2.0,
            label=(
                r"$\Omega_{\mathrm{LSP}}h^2\,"
                r"(\mu<0,\ \tan\beta=1.6)=0.12$"
            ),
        )
    forbidden_fraction = 0.08
    forbidden_x = xmin + forbidden_fraction * (xmax - xmin)
    forbidden_y = forbidden_x - final_offshell_boundary + 18.0
    axes.text(
        forbidden_x,
        forbidden_y,
        "Decay forbidden",
        color="0.35",
        fontsize=15,
        ha="left",
        va="bottom",
        rotation=45.0,
        rotation_mode="anchor",
        transform_rotates_text=True,
        zorder=4.5,
    )
    def suppress_tiny_contour_paths(
        contour_set: Any,
        max_span_GeV: float = 35.0,
    ) -> list[dict[str, float]]:
        """Remove marker-sized disconnected contour loops from rendering."""

        removed = []
        for collection in contour_set.collections:
            kept_paths = []
            for path in collection.get_paths():
                vertices = np.asarray(path.vertices, dtype=float)
                if len(vertices) < 2:
                    continue
                x_span = float(np.ptp(vertices[:, 0]))
                y_span = float(np.ptp(vertices[:, 1]))
                if x_span <= max_span_GeV and y_span <= max_span_GeV:
                    removed.append(
                        {
                            "x_center_GeV": float(np.mean(vertices[:, 0])),
                            "y_center_GeV": float(np.mean(vertices[:, 1])),
                            "x_span_GeV": x_span,
                            "y_span_GeV": y_span,
                        }
                    )
                    continue
                kept_paths.append(path)
            collection.set_paths(kept_paths)
        return removed

    removed_tiny_expected_paths = []
    for central_grid in expected_ratio_components:
        if central_grid.count() > 0:
            central_contour = axes.contour(
                xx,
                yy,
                central_grid,
                levels=[0.0],
                colors="red",
                linewidths=3.0,
                zorder=6,
            )
            if topology == "T2tb":
                removed_tiny_expected_paths.extend(
                    suppress_tiny_contour_paths(central_contour)
                )
    if removed_tiny_expected_paths and interpolation_diagnostics is not None:
        interpolation_diagnostics["rendering_artifact_suppression"] = {
            "policy": (
                "T2tb central expected contour only; remove disconnected "
                "paths whose x and y spans are both <= 35 GeV"
            ),
            "removed_paths": removed_tiny_expected_paths,
        }
    for band_components in (minus1_components, plus1_components):
        for band_grid in band_components:
            if band_grid.count() > 0:
                axes.contour(
                    xx,
                    yy,
                    band_grid,
                    levels=[0.0],
                    colors="red",
                    linewidths=1.7,
                    linestyles="--",
                    zorder=5,
                )
    for central_grid in overlay_expected_components:
        if central_grid.count() > 0:
            axes.contour(
                xx,
                yy,
                central_grid,
                levels=[0.0],
                colors="blue",
                linewidths=3.0,
                zorder=6.5,
            )
    for band_components in (
        overlay_minus1_components,
        overlay_plus1_components,
    ):
        for band_grid in band_components:
            if band_grid.count() > 0:
                axes.contour(
                    xx,
                    yy,
                    band_grid,
                    levels=[0.0],
                    colors="blue",
                    linewidths=1.7,
                    linestyles="--",
                    zorder=5.5,
                )

    run2_handles: list[Any] = []
    if run2_contours is not None and Path(run2_contours).exists():
        try:
            run2_payload = read_json(Path(run2_contours))
        except Exception:
            run2_payload = {}
        for key, style, label in (
            ("observed", "-", "SUS-19-010 obs."),
            ("expected", "--", "SUS-19-010 exp."),
        ):
            coordinates = np.asarray(run2_payload.get(key) or [], dtype=float)
            if coordinates.ndim == 2 and coordinates.shape[1] >= 2:
                axes.plot(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    color="black",
                    linestyle=style,
                    linewidth=2.4,
                    zorder=7,
                )
                run2_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        lw=2.4,
                        linestyle=style,
                        label=label,
                    )
                )

    if (
        run2_compressed_contours is not None
        and Path(run2_compressed_contours).exists()
    ):
        try:
            compressed_payload = read_json(Path(run2_compressed_contours))
        except Exception:
            compressed_payload = {}
        for key, style in (
            ("observed", "-"),
            ("expected", "--"),
        ):
            coordinates = np.asarray(
                compressed_payload.get(key) or [], dtype=float
            )
            if coordinates.ndim == 2 and coordinates.shape[1] >= 2:
                axes.plot(
                    coordinates[:, 0],
                    coordinates[:, 1],
                    color="black",
                    linestyle=style,
                    linewidth=2.4,
                    zorder=8,
                )

    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)
    axes.set_xlabel(r"$m_{\tilde{t}}$ (GeV)", fontsize=34, loc="right")
    axes.set_ylabel(r"$m_{\tilde{\chi}_1^0}$ (GeV)", fontsize=34)
    axes.xaxis.set_major_locator(MultipleLocator(200))
    axes.yaxis.set_major_locator(MultipleLocator(200))
    axes.xaxis.set_minor_locator(MultipleLocator(50))
    axes.yaxis.set_minor_locator(MultipleLocator(50))
    axes.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=24,
        length=9,
    )
    axes.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=5,
    )
    for spine in axes.spines.values():
        spine.set_linewidth(1.8)
    with plt.rc_context({"font.size": 22}):
        hep.cms.label(
            llabel="Work in progress",
            rlabel=luminosity_label,
            fontsize=27,
            ax=axes,
        )
    multiline_decay_label = bool(decay_label and "\n" in decay_label)
    if analysis_label:
        axes.text(
            0.14,
            0.905,
            analysis_label,
            transform=axes.transAxes,
            fontsize=16,
            va="top",
        )
    expected_handles = [
        Line2D(
            [0],
            [0],
            color="red",
            lw=3.0,
            label=(
                primary_legend_label
                or (
                    r"High-$\Delta m$ only exp. $\pm 1\sigma$"
                    if overlay_limit_payload is not None
                    else r"Expected $\pm 1\sigma$"
                )
            ),
        )
    ]
    if overlay_limit_payload is not None:
        expected_handles.append(
            Line2D(
                [0],
                [0],
                color="blue",
                lw=3.0,
                label=(
                    overlay_legend_label
                    or r"High-$\Delta m$ + Low-$\Delta m$ exp. $\pm 1\sigma$"
                ),
            )
        )
    legend = axes.legend(
        handles=[
            *expected_handles,
            *run2_handles,
            *([thermal_relic_handle] if thermal_relic_handle else []),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
        bbox_transform=axes.transAxes,
        mode="expand",
        ncol=1,
        borderaxespad=0.0,
        frameon=True,
        title=decay_label,
        title_fontsize=20 if multiline_decay_label else 23,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fancybox=False,
        borderpad=0.55 if thermal_relic_handle else 0.8,
        labelspacing=(
            0.95
            if multiline_decay_label
            else 0.32
            if thermal_relic_handle
            else 0.6
        ),
        fontsize=17 if thermal_relic_handle else 19,
        handlelength=2.4 if thermal_relic_handle else 2.8,
    )
    legend._legend_box.align = "left"
    legend.get_frame().set_linewidth(1.8)
    legend.set_zorder(30)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180)
    figure.savefig(output_png.with_suffix(".pdf"))
    plt.close(figure)
    return True


def comparison(
    merged: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    ratios = []
    absolute = []
    for mass_key, point in merged.get("points", {}).items():
        reference = baseline.get("points", {}).get(mass_key)
        if not reference:
            continue
        new_value = float(point["expected"])
        old_value = float(reference["expected"])
        if old_value <= 0.0:
            continue
        ratios.append(new_value / old_value)
        absolute.append(new_value - old_value)
    if not ratios:
        return {"matched_points": 0}
    ratio_array = np.asarray(ratios, dtype=float)
    absolute_array = np.asarray(absolute, dtype=float)
    return {
        "matched_points": int(ratio_array.size),
        "expected_limit_ratio": {
            "minimum": float(np.min(ratio_array)),
            "median": float(np.median(ratio_array)),
            "maximum": float(np.max(ratio_array)),
        },
        "expected_limit_difference": {
            "minimum": float(np.min(absolute_array)),
            "median": float(np.median(absolute_array)),
            "maximum": float(np.max(absolute_array)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--baseline-limits", type=Path)
    parser.add_argument(
        "--overlay-limits",
        type=Path,
        help=(
            "optional expected_limits.json whose expected contour is drawn "
            "in blue over the primary result"
        ),
    )
    parser.add_argument(
        "--primary-legend-label",
        help="optional legend label for the primary expected contour",
    )
    parser.add_argument(
        "--overlay-legend-label",
        help="optional legend label for the overlaid expected contour",
    )
    parser.add_argument(
        "--thermal-relic",
        type=Path,
        help=(
            "optional higgsino-stop relic-contour JSON; draws "
            "Omega_LSP*h^2=0.12 and hatches the over-abundant side"
        ),
    )
    parser.add_argument(
        "--campaign-year", choices=("2024", "2025", "2024_2025"), required=True
    )
    parser.add_argument(
        "--topology",
        choices=("T2tt", "T2bW", "T2tb"),
        required=True,
    )
    parser.add_argument("--min-mstop", type=float, default=500.0)
    parser.add_argument("--max-mstop", type=float, default=1800.0)
    parser.add_argument("--min-mlsp", type=float, default=1.0)
    parser.add_argument("--max-mlsp", type=float, default=1800.0)
    parser.add_argument(
        "--color-field",
        choices=("ratio", "xsec"),
        default="xsec",
        help="background color quantity (default: absolute cross-section limit)",
    )
    parser.add_argument(
        "--signal-xsec",
        type=Path,
        default=DEFAULT_STOP_XSEC,
        help=(
            "stop-pair cross-section table used by the default xsec view "
            f"(default: {DEFAULT_STOP_XSEC})"
        ),
    )
    parser.add_argument(
        "--interpolation-mode",
        choices=("global", "regime", "topology-aware"),
        default="global",
        help=(
            "global reproduces the historical interpolation; regime splits "
            "the grid at topology thresholds; topology-aware applies the "
            "model-specific analysis plotting policy"
        ),
    )
    parser.add_argument(
        "--reuse-collected-limits",
        action="store_true",
        help=(
            "reuse an audited expected_limits.json whose point accounting "
            "exactly matches the build manifest"
        ),
    )
    parser.add_argument(
        "--allow-noncanonical-runtime",
        action="store_true",
        help=(
            "allow local display-only rendering with the installed plotting "
            "stack; the actual versions are recorded in the output manifest"
        ),
    )
    args = parser.parse_args()
    if not (0.0 < args.min_mstop < args.max_mstop):
        parser.error("require 0 < --min-mstop < --max-mstop")
    runtime = (
        runtime_versions()
        if args.allow_noncanonical_runtime
        else validate_runtime()
    )

    if args.color_field == "xsec" and args.signal_xsec is None:
        parser.error("--signal-xsec is required with --color-field xsec")
    stop_pair_xsecs = (
        load_stop_pair_xsecs(args.signal_xsec)
        if args.color_field == "xsec"
        else None
    )

    manifest_candidates = (
        args.input_dir / "manifest.json",
        args.input_dir / "combine_input_manifest.json",
    )
    manifest_path = next(
        (path for path in manifest_candidates if path.is_file()),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(
            "missing limit-build manifest; expected one of: "
            + ", ".join(str(path) for path in manifest_candidates)
        )
    build_manifest = json.loads(manifest_path.read_text())
    excluded_points = load_excluded_points(args.input_dir)
    original_mass_keys = list(build_manifest["mass_points"])
    policy_mass_keys = [
        mass_key for mass_key in original_mass_keys
        if mass_key not in excluded_points
    ]
    mass_keys = []
    out_of_plot_range = []
    for mass_key in policy_mass_keys:
        mstop, mlsp = parse_mass_key(mass_key)
        if (
            args.min_mstop <= mstop <= args.max_mstop
            and args.min_mlsp <= mlsp <= args.max_mlsp
        ):
            mass_keys.append(mass_key)
        else:
            out_of_plot_range.append(mass_key)
    expected_limits_path = args.input_dir / "expected_limits.json"
    if args.reuse_collected_limits:
        if not expected_limits_path.is_file():
            raise FileNotFoundError(
                f"missing cached limits: {expected_limits_path}"
            )
        all_limits = read_json(expected_limits_path)
        cached_accounting = set(all_limits.get("points") or {}) | set(
            all_limits.get("missing_points") or []
        )
        # A cached collection belongs to the immutable build manifest.  User
        # exclusions are applied below when selecting the plotted grid; they
        # must not require rerunning or rewriting successful Combine outputs.
        if cached_accounting != set(original_mass_keys):
            raise RuntimeError(
                "cached expected-limit point accounting does not match the "
                "current build manifest"
            )
        limit_collection_source = "audited_expected_limits_cache"
    else:
        all_limits = collect_limits(
            args.input_dir / "limits",
            original_mass_keys,
            expected_limits_path,
        )
        limit_collection_source = "combine_root_collection"
    limits = select_limit_range(all_limits, mass_keys)
    overlay_limits = None
    if args.overlay_limits is not None:
        if not args.overlay_limits.is_file():
            raise FileNotFoundError(
                f"missing overlay limits: {args.overlay_limits}"
            )
        overlay_limits = select_limit_range(
            read_json(args.overlay_limits), mass_keys
        )
        if overlay_limits["status"] == "no_combine_outputs":
            raise RuntimeError(
                "overlay limits contain no points in the requested plot range"
            )
    thermal_relic = None
    if args.thermal_relic is not None:
        if not args.thermal_relic.is_file():
            raise FileNotFoundError(
                f"missing thermal-relic input: {args.thermal_relic}"
            )
        thermal_relic = load_thermal_relic_contour(args.thermal_relic)
    highdm_bins = int(build_manifest["model"]["highdm_bins"])
    lowdm_bins = int(build_manifest["model"]["lowdm_bins"])
    baseline = (
        json.loads(args.baseline_limits.read_text())
        if args.baseline_limits
        else {"points": {}}
    )
    color_suffix = "_xsec" if args.color_field == "xsec" else ""
    interpolation_suffix = {
        "global": "",
        "regime": "_regime",
        "topology-aware": "_topology",
    }[args.interpolation_mode]
    output_png = args.input_dir / (
        f"expected_limit_{args.topology.lower()}_"
        f"{args.campaign_year}_highdm{highdm_bins}_lowdm{lowdm_bins}"
        f"{color_suffix}{interpolation_suffix}"
        f"{'_overlay' if overlay_limits is not None else ''}.png"
    )
    if thermal_relic is not None:
        output_png = output_png.with_name(
            f"{output_png.stem}_thermal_relic{output_png.suffix}"
        )
    contour_complete = False
    interpolation_diagnostics: dict[str, Any] = {}
    if limits["status"] in {"complete", "partial"}:
        contour_complete = plot_contour(
            limits,
            output_png,
            run2_contours=RUN2_CONTOURS[args.topology],
            luminosity_label=LUMINOSITY_LABELS[args.campaign_year],
            analysis_label=None,
            x_max=args.max_mstop,
            x_min=args.min_mstop,
            y_min=args.min_mlsp,
            y_max=args.max_mlsp,
            decay_label=DECAY_LABELS[args.topology],
            color_field=args.color_field,
            stop_pair_xsecs=stop_pair_xsecs,
            run2_compressed_contours=RUN2_COMPRESSED_CONTOURS[args.topology],
            topology=args.topology,
            interpolation_mode=args.interpolation_mode,
            interpolation_diagnostics=interpolation_diagnostics,
            overlay_limit_payload=overlay_limits,
            primary_legend_label=args.primary_legend_label,
            overlay_legend_label=args.overlay_legend_label,
            thermal_relic=thermal_relic,
        )

    result = {
        "status": (
            "complete"
            if contour_complete and limits["status"] == "complete"
            else "partial"
            if contour_complete
            else "failed"
        ),
        "schema_version": (
            f"canonical_{args.campaign_year}_highdm{highdm_bins}_"
            f"lowdm{lowdm_bins}_limit_v1"
        ),
        "campaign_year": args.campaign_year,
        "topology": args.topology,
        "color_field": args.color_field,
        "interpolation_mode": args.interpolation_mode,
        "plot_range_GeV": {
            "mStop": [args.min_mstop, args.max_mstop],
            "mLSP": [args.min_mlsp, args.max_mlsp],
        },
        "interpolation": interpolation_diagnostics,
        "limit_collection_source": limit_collection_source,
        "highdm_bins": highdm_bins,
        "lowdm_bins": lowdm_bins,
        "original_mass_point_count": len(original_mass_keys),
        "mass_point_count": len(mass_keys),
        "out_of_plot_range_mass_point_count": len(out_of_plot_range),
        "out_of_plot_range_mass_points": sorted(out_of_plot_range),
        "excluded_point_count": len(excluded_points),
        "excluded_points": sorted(excluded_points),
        "limits": limits,
        "full_input_limit_accounting": {
            "status": all_limits["status"],
            "requested_point_count": all_limits["requested_point_count"],
            "collected_point_count": all_limits["collected_point_count"],
            "missing_points": all_limits["missing_points"],
        },
        "baseline_limits": (
            str(args.baseline_limits) if args.baseline_limits else None
        ),
        "baseline_comparison": (
            comparison(limits, baseline) if args.baseline_limits else None
        ),
        "overlay_limits_path": (
            str(args.overlay_limits) if args.overlay_limits else None
        ),
        "overlay_limits": overlay_limits,
        "thermal_relic": (
            {
                key: value
                for key, value in thermal_relic.items()
                if key != "points"
            }
            if thermal_relic is not None
            else None
        ),
        "contour_png": str(output_png) if contour_complete else None,
        "contour_pdf": (
            str(output_png.with_suffix(".pdf"))
            if contour_complete
            else None
        ),
        "run2_overlay": True,
        "run2_compressed_overlay": (
            str(RUN2_COMPRESSED_CONTOURS[args.topology])
            if RUN2_COMPRESSED_CONTOURS[args.topology]
            else None
        ),
        "data_mode": "asimov",
        "runtime": runtime,
    }
    if "topology_mapping_audit" in build_manifest:
        result["topology_mapping_audit"] = build_manifest["topology_mapping_audit"]
    manifest_stem = (
        "limit_manifest_xsec"
        if args.color_field == "xsec"
        else "limit_manifest"
    )
    manifest_name = (
        f"{manifest_stem}{interpolation_suffix}"
        f"{'_overlay' if overlay_limits is not None else ''}.json"
    )
    if thermal_relic is not None:
        manifest_name = manifest_name.replace(
            ".json", "_thermal_relic.json"
        )
    write_json(args.input_dir / manifest_name, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "topology": args.topology,
                "collected": limits["collected_point_count"],
                "missing": len(limits["missing_points"]),
                "comparison": result["baseline_comparison"],
                "contour_png": result["contour_png"],
            },
            sort_keys=True,
        )
    )
    return 0 if contour_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
