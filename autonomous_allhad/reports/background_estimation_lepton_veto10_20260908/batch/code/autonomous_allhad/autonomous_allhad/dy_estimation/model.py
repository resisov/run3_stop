"""Data model and on/off-Z profile-likelihood fit for DY normalization."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.optimize import minimize


CHANNELS = ("DY2E", "DY2M")
GROUPS = ("Nb1", "Nb2plus")
MASS_WINDOWS = ("on", "off")
MLL_EDGES = np.asarray(
    [50.0, 70.0, 81.0, 91.0, 101.0, 120.0, 160.0, 250.0, 500.0],
    dtype=float,
)


def empty_yield() -> dict[str, Any]:
    return {"sumw": 0.0, "sumw2": 0.0, "entries": 0}


def add_yield(target: dict[str, Any], weights: np.ndarray) -> None:
    selected = np.asarray(weights, dtype=float)
    selected = selected[np.isfinite(selected)]
    target["sumw"] = float(target["sumw"]) + float(np.sum(selected))
    target["sumw2"] = float(target["sumw2"]) + float(
        np.sum(selected * selected)
    )
    target["entries"] = int(target["entries"]) + int(len(selected))


def nested_yield(
    payload: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    target = payload
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    return target.setdefault(keys[-1], empty_yield())



def empty_histogram(edges: np.ndarray) -> dict[str, Any]:
    size = len(edges) - 1
    return {
        "edges": edges.tolist(),
        "sumw": [0.0] * size,
        "sumw2": [0.0] * size,
        "entries": [0] * size,
    }


def nested_histogram(
    payload: dict[str, Any],
    keys: tuple[str, ...],
    edges: np.ndarray,
) -> dict[str, Any]:
    target = payload
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    return target.setdefault(keys[-1], empty_histogram(edges))


def fill_histogram(
    target: dict[str, Any],
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
) -> None:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(values)
        & np.isfinite(weights)
        & (values >= edges[0])
    )
    if not np.any(valid):
        return
    indices = np.searchsorted(edges, values[valid], side="right") - 1
    indices = np.minimum(indices, len(edges) - 2)
    selected_weights = weights[valid]
    target["sumw"] = (
        np.asarray(target["sumw"], dtype=float)
        + np.bincount(
            indices,
            weights=selected_weights,
            minlength=len(edges) - 1,
        )
    ).tolist()
    target["sumw2"] = (
        np.asarray(target["sumw2"], dtype=float)
        + np.bincount(
            indices,
            weights=selected_weights * selected_weights,
            minlength=len(edges) - 1,
        )
    ).tolist()
    target["entries"] = (
        np.asarray(target["entries"], dtype=int)
        + np.bincount(indices, minlength=len(edges) - 1)
    ).tolist()


def merge_yield(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("sumw", "sumw2", "entries"):
        if isinstance(source[key], list):
            target[key] = (
                np.asarray(target[key]) + np.asarray(source[key])
            ).tolist()
        elif key == "entries":
            target[key] = int(target[key]) + int(source[key])
        else:
            target[key] = float(target[key]) + float(source[key])


def merge_tree(target: dict[str, Any], source: dict[str, Any]) -> None:
    if set(source) >= {"sumw", "sumw2", "entries"}:
        merge_yield(target, source)
        return
    for key, value in source.items():
        if isinstance(value, dict):
            if set(value) >= {"sumw", "sumw2", "entries"}:
                initial = (
                    empty_histogram(np.asarray(value["edges"], dtype=float))
                    if isinstance(value.get("sumw"), list)
                    and "edges" in value
                    else empty_yield()
                )
                merge_yield(target.setdefault(key, initial), value)
            else:
                merge_tree(target.setdefault(key, {}), value)


def solve_matrix(
    data_on: float,
    data_off: float,
    z_on: float,
    z_off: float,
    other_on: float,
    other_off: float,
    variances: list[float],
) -> dict[str, Any]:
    """Fit non-negative DY and non-DY normalizations in on/off-Z data.

    The two observed counts are described with Poisson terms.  The four
    weighted-MC template yields are profiled with Gaussian constraints whose
    variances are their sumw2 values.  This is the likelihood counterpart of
    the 2x2 matrix solution, but it remains physical when a low-count bin would
    otherwise return a negative component normalization.
    """

    matrix = np.asarray(
        [[z_on, other_on], [z_off, other_off]], dtype=float
    )
    data = np.asarray([data_on, data_off], dtype=float)
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or abs(determinant) < 1.0e-12:
        return {
            "status": "singular",
            "determinant": determinant,
            "RZ": None,
            "RT": None,
        }
    nominal = np.asarray([z_on, z_off, other_on, other_off], dtype=float)
    template_variance = np.maximum(
        np.asarray(variances[2:], dtype=float), 0.0
    )
    floating = np.flatnonzero(template_variance > 0.0)

    try:
        algebraic = np.linalg.solve(matrix, data)
    except np.linalg.LinAlgError:
        algebraic = np.asarray([1.0, 1.0], dtype=float)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = np.asarray(parameters[:2], dtype=float)
        templates = nominal.copy()
        templates[floating] = parameters[2:]
        return scale, templates

    def nll(parameters: np.ndarray) -> float:
        scale, templates = unpack(parameters)
        expectation = np.asarray(
            [
                scale[0] * templates[0] + scale[1] * templates[2],
                scale[0] * templates[1] + scale[1] * templates[3],
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(expectation)) or np.any(expectation <= 0.0):
            return 1.0e100
        poisson = float(np.sum(expectation - data * np.log(expectation)))
        if len(floating):
            delta = templates[floating] - nominal[floating]
            constraint = 0.5 * float(
                np.sum(delta * delta / template_variance[floating])
            )
        else:
            constraint = 0.0
        return poisson + constraint

    clipped = np.maximum(algebraic, 0.0)
    seeds = (
        clipped,
        np.asarray([1.0, 1.0], dtype=float),
        np.asarray(
            [
                max((data_on + data_off) / max(z_on + z_off, 1.0e-9), 0.0),
                0.0,
            ],
            dtype=float,
        ),
    )
    best = None
    bounds = [(0.0, None), (0.0, None)] + [
        (0.0, None) for _ in floating
    ]
    if np.all(algebraic >= 0.0):
        # With two observations and two positive scale factors, the algebraic
        # solution reproduces both Poisson means exactly while leaving every
        # constrained MC template at its nominal value.  It is therefore the
        # global interior likelihood maximum; no numerical minimizer is needed.
        parameters = np.concatenate(
            [algebraic, np.maximum(nominal[floating], 0.0)]
        )
        best = SimpleNamespace(
            x=parameters,
            fun=nll(parameters),
            success=True,
            message="analytic interior maximum",
        )
    else:
        for seed in seeds:
            initial = np.concatenate(
                [seed, np.maximum(nominal[floating], 0.0)]
            )
            candidate = minimize(
                nll,
                initial,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1.0e-12, "gtol": 1.0e-8, "maxiter": 5000},
            )
            if np.isfinite(candidate.fun) and (
                best is None or candidate.fun < best.fun
            ):
                best = candidate
    if best is None:
        return {
            "status": "fit_failed",
            "determinant": determinant,
            "message": "no finite likelihood candidate",
            "RZ": None,
            "RT": None,
        }

    scale, templates = unpack(best.x)
    expectation = np.asarray(
        [
            scale[0] * templates[0] + scale[1] * templates[2],
            scale[0] * templates[1] + scale[1] * templates[3],
        ],
        dtype=float,
    )

    # Expected Fisher information for the fitted scale factors and profiled MC
    # template yields.  Its inverse gives the local covariance after profiling.
    dimension = 2 + len(floating)
    information = np.zeros((dimension, dimension), dtype=float)
    template_to_parameter = {
        int(template_index): 2 + position
        for position, template_index in enumerate(floating)
    }
    for window_index in range(2):
        if window_index == 0:
            gradient = np.asarray([templates[0], templates[2]], dtype=float)
            template_indices = (0, 2)
        else:
            gradient = np.asarray([templates[1], templates[3]], dtype=float)
            template_indices = (1, 3)
        full_gradient = np.zeros(dimension, dtype=float)
        full_gradient[:2] = gradient
        for component_index, template_index in enumerate(template_indices):
            parameter_index = template_to_parameter.get(template_index)
            if parameter_index is not None:
                full_gradient[parameter_index] = scale[component_index]
        information += np.outer(full_gradient, full_gradient) / max(
            expectation[window_index], 1.0e-12
        )
    for position, template_index in enumerate(floating):
        information[2 + position, 2 + position] += (
            1.0 / template_variance[template_index]
        )
    covariance_full = np.linalg.pinv(information, hermitian=True)
    covariance = covariance_full[:2, :2]
    return {
        "status": "complete",
        "determinant": determinant,
        "solver": "nonnegative_profile_likelihood_poisson_data_gaussian_mcstat",
        "fit_converged": bool(best.success),
        "fit_message": str(best.message),
        "fit_nll": float(best.fun),
        "fit_expectation": expectation.tolist(),
        "profiled_templates": templates.tolist(),
        "boundary": {
            "RZ": bool(scale[0] <= 1.0e-8),
            "RT": bool(scale[1] <= 1.0e-8),
        },
        "RZ": float(scale[0]),
        "RT": float(scale[1]),
        "RZ_stat": float(math.sqrt(max(covariance[0, 0], 0.0))),
        "RT_stat": float(math.sqrt(max(covariance[1, 1], 0.0))),
        "correlation": float(
            covariance[0, 1]
            / math.sqrt(
                max(covariance[0, 0], 1.0e-300)
                * max(covariance[1, 1], 1.0e-300)
            )
        ),
        "covariance": covariance.tolist(),
    }


def finalize_rz(payload: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"channels": {}, "combined": {}}
    for channel in CHANNELS:
        output["channels"][channel] = {}
        for group in GROUPS:
            source = ((payload.get(channel) or {}).get(group) or {})

            def leaf(window: str, component: str) -> dict[str, Any]:
                return (
                    (source.get(window) or {}).get(component)
                    or empty_yield()
                )

            data_on = leaf("on", "data")
            data_off = leaf("off", "data")
            z_on = leaf("on", "zll")
            z_off = leaf("off", "zll")
            other_on = leaf("on", "other")
            other_off = leaf("off", "other")
            solution = solve_matrix(
                float(data_on["sumw"]),
                float(data_off["sumw"]),
                float(z_on["sumw"]),
                float(z_off["sumw"]),
                float(other_on["sumw"]),
                float(other_off["sumw"]),
                [
                    max(float(data_on["sumw"]), 0.0),
                    max(float(data_off["sumw"]), 0.0),
                    float(z_on["sumw2"]),
                    float(z_off["sumw2"]),
                    float(other_on["sumw2"]),
                    float(other_off["sumw2"]),
                ],
            )
            solution["inputs"] = {
                "data_on": data_on,
                "data_off": data_off,
                "zll_on": z_on,
                "zll_off": z_off,
                "other_on": other_on,
                "other_off": other_off,
            }
            output["channels"][channel][group] = solution
    for group in GROUPS:
        measurements = [
            output["channels"][channel][group]
            for channel in CHANNELS
            if output["channels"][channel][group].get("status")
            == "complete"
            and output["channels"][channel][group].get("RZ_stat", 0.0) > 0.0
        ]
        if not measurements:
            output["combined"][group] = {"status": "unavailable"}
            continue
        weights = np.asarray(
            [1.0 / item["RZ_stat"] ** 2 for item in measurements],
            dtype=float,
        )
        values = np.asarray(
            [item["RZ"] for item in measurements], dtype=float
        )
        output["combined"][group] = {
            "status": "complete",
            "RZ": float(np.sum(weights * values) / np.sum(weights)),
            "RZ_stat": float(math.sqrt(1.0 / np.sum(weights))),
            "channels": [
                channel
                for channel in CHANNELS
                if output["channels"][channel][group].get("status")
                == "complete"
            ],
        }
    return output
