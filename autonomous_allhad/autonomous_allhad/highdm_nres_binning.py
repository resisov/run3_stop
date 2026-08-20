from __future__ import annotations

from typing import Sequence

import numpy as np

from .highdm_resolved_categories import COARSE_NRES_TOPOLOGIES


HIGH_MERGE_PAIRS = ((22, 23), (34, 35), (40, 41), (52, 53), (58, 59))
RECOIL_LABELS = ("250to300", "300to350", "350to400", "400to500", "500to800", "800to1500")


def adopted55_mapping() -> tuple[tuple[int, ...], ...]:
    """Return the adopted mapping from the 60-bin precursor to 55 bins."""
    merge = dict(HIGH_MERGE_PAIRS)
    seconds = {second for _first, second in HIGH_MERGE_PAIRS}
    result: list[tuple[int, ...]] = []
    for index in range(60):
        if index in seconds:
            continue
        result.append((index, merge[index]) if index in merge else (index,))
    if len(result) != 55 or sorted(sum((list(group) for group in result), [])) != list(range(60)):
        raise AssertionError("invalid adopted High-dM 55-bin mapping")
    return tuple(result)


def source60_to_adopted55() -> np.ndarray:
    result = np.full(60, -1, dtype=np.int16)
    for target, sources in enumerate(adopted55_mapping()):
        result[list(sources)] = target
    if np.any(result < 0):
        raise AssertionError("the adopted 55-bin map does not cover all 60 source bins")
    return result


def map60_indices_to_adopted55(indices: Sequence[int] | np.ndarray) -> np.ndarray:
    source = np.asarray(indices, dtype=int)
    output = np.full(source.shape, -1, dtype=np.int16)
    valid = (source >= 0) & (source < 60)
    output[valid] = source60_to_adopted55()[source[valid]]
    return output


def adopted55_labels(source60_labels: Sequence[str]) -> list[str]:
    if len(source60_labels) != 60:
        raise ValueError(f"expected 60 source labels, found {len(source60_labels)}")
    labels = []
    for sources in adopted55_mapping():
        if len(sources) == 1:
            labels.append(str(source60_labels[sources[0]]))
        else:
            labels.append(
                f"{source60_labels[sources[0]]}__plus__{source60_labels[sources[1]]}"
            )
    return labels


def topology_positions(nt: np.ndarray, nw: np.ndarray, nres: np.ndarray) -> np.ndarray:
    """Vectorized exclusive topology position for events with Nres>0."""
    top = np.asarray(nt, dtype=int)
    w = np.asarray(nw, dtype=int)
    resolved = np.asarray(nres, dtype=int)
    if not (top.shape == w.shape == resolved.shape):
        raise ValueError("Nt, NW, and Nres arrays have different shapes")
    if np.any(top < 0) or np.any(w < 0) or np.any(resolved < 0):
        raise ValueError("topology multiplicities must be non-negative")
    output = np.full(top.shape, -1, dtype=np.int8)
    positive = resolved > 0
    output[positive & (top == 0) & (w == 0) & (resolved == 1)] = 0
    output[positive & (top == 0) & (w == 0) & (resolved >= 2)] = 1
    output[positive & (top > 0) & (w == 0)] = 2
    output[positive & (top == 0) & (w > 0)] = 3
    output[positive & (top > 0) & (w > 0)] = 4
    if np.any(positive & (output < 0)):
        raise AssertionError("uncovered Nres>0 topology")
    return output


def exclusive85_indices(
    baseline55: np.ndarray,
    recoil_index: np.ndarray,
    nt: np.ndarray,
    nw: np.ndarray,
    nres: np.ndarray,
) -> np.ndarray:
    """Split the adopted 55 bins into Nres=0 plus five 6-bin blocks.

    The input population is defined by ``baseline55 >= 0``.  Every such event
    maps to exactly one output bin: Nres=0 stays in its original 55-bin slot,
    while Nres>0 moves to a coarse Run-2-inspired topology crossed with the
    six recoil bins.
    """
    baseline = np.asarray(baseline55, dtype=int)
    recoil = np.asarray(recoil_index, dtype=int)
    resolved = np.asarray(nres, dtype=int)
    if not (baseline.shape == recoil.shape == np.asarray(nt).shape == np.asarray(nw).shape == resolved.shape):
        raise ValueError("exclusive binning arrays have different shapes")
    output = np.full(baseline.shape, -1, dtype=np.int16)
    population = baseline >= 0
    invalid_recoil = population & ((recoil < 0) | (recoil >= len(RECOIL_LABELS)))
    if np.any(invalid_recoil):
        raise ValueError("baseline event has an invalid recoil index")
    output[population & (resolved == 0)] = baseline[population & (resolved == 0)]
    positions = topology_positions(np.asarray(nt), np.asarray(nw), resolved)
    moved = population & (resolved > 0)
    output[moved] = 55 + positions[moved] * len(RECOIL_LABELS) + recoil[moved]
    if np.any(population & ((output < 0) | (output >= 85))):
        raise AssertionError("baseline event was not assigned to exactly one 85-bin slot")
    return output


def extended85_to_tailmerged80() -> tuple[tuple[int, ...], ...]:
    """Keep the 55-bin baseline and merge the last two recoil bins per new block."""
    mapping: list[tuple[int, ...]] = [(index,) for index in range(55)]
    for topology in range(len(COARSE_NRES_TOPOLOGIES)):
        start = 55 + topology * len(RECOIL_LABELS)
        mapping.extend((start + recoil,) for recoil in range(4))
        mapping.append((start + 4, start + 5))
    flat = sorted(sum((list(group) for group in mapping), []))
    if len(mapping) != 80 or flat != list(range(85)):
        raise AssertionError("invalid 85-to-80 tail-merge map")
    return tuple(mapping)


def map85_indices_to_tailmerged80(indices: Sequence[int] | np.ndarray) -> np.ndarray:
    source = np.asarray(indices, dtype=int)
    lookup = np.full(85, -1, dtype=np.int16)
    for target, sources in enumerate(extended85_to_tailmerged80()):
        lookup[list(sources)] = target
    output = np.full(source.shape, -1, dtype=np.int16)
    valid = (source >= 0) & (source < 85)
    output[valid] = lookup[source[valid]]
    return output


def exclusive85_labels(source60_labels: Sequence[str]) -> list[str]:
    labels = [f"{label}__Nres0" for label in adopted55_labels(source60_labels)]
    labels.extend(
        f"{topology}__recoil_{recoil}"
        for topology in COARSE_NRES_TOPOLOGIES
        for recoil in RECOIL_LABELS
    )
    if len(labels) != 85:
        raise AssertionError("wrong exclusive 85-bin label count")
    return labels


def tailmerged80_labels(source60_labels: Sequence[str]) -> list[str]:
    source = exclusive85_labels(source60_labels)
    result = []
    for group in extended85_to_tailmerged80():
        if len(group) == 1:
            result.append(source[group[0]])
        else:
            topology = source[group[0]].split("__recoil_", 1)[0]
            result.append(f"{topology}__recoil_500to1500")
    return result
