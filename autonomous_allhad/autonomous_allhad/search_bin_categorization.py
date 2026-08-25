from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .highdm_resolved_categories import COARSE_NRES_TOPOLOGIES


HIGH_MERGE_PAIRS = ((22, 23), (34, 35), (40, 41), (52, 53), (58, 59))
RECOIL_LABELS = ("250to300", "300to350", "350to400", "400to500", "500to800", "800plus")
ADDITIONAL_HIGHDM_CATEGORIES = ("nb2_nt0_nw2_nres0",)
BASELINE_SOURCE_BIN_COUNT = 55
NRES_SOURCE_OFFSET = BASELINE_SOURCE_BIN_COUNT
ADDITIONAL_CATEGORY_SOURCE_OFFSET = (
    NRES_SOURCE_OFFSET + len(COARSE_NRES_TOPOLOGIES) * len(RECOIL_LABELS)
)
EXCLUSIVE_SOURCE_BIN_COUNT = (
    ADDITIONAL_CATEGORY_SOURCE_OFFSET
    + len(ADDITIONAL_HIGHDM_CATEGORIES) * len(RECOIL_LABELS)
)


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


def exclusive_category_source_indices(
    baseline55: np.ndarray,
    recoil_index: np.ndarray,
    nb: np.ndarray,
    nt: np.ndarray,
    nw: np.ndarray,
    nres: np.ndarray,
    population_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Assign the exclusive High-dM category-source bins.

    The input population is defined by ``baseline55 >= 0``.  Every such event
    maps to exactly one output bin: Nres=0 stays in its original 55-bin slot,
    except for the explicit Nb=2, Nt=0, NW=2 category, while Nres>0 moves to a
    coarse Run-2-inspired topology.  Every added category is crossed with the
    same six open-ended recoil bins.
    """
    baseline = np.asarray(baseline55, dtype=int)
    recoil = np.asarray(recoil_index, dtype=int)
    bottom = np.asarray(nb, dtype=int)
    top = np.asarray(nt, dtype=int)
    w = np.asarray(nw, dtype=int)
    resolved = np.asarray(nres, dtype=int)
    if not (
        baseline.shape
        == recoil.shape
        == bottom.shape
        == top.shape
        == w.shape
        == resolved.shape
    ):
        raise ValueError("exclusive binning arrays have different shapes")
    output = np.full(baseline.shape, -1, dtype=np.int16)
    population = (
        baseline >= 0
        if population_mask is None
        else np.asarray(population_mask, dtype=bool)
    )
    if population.shape != baseline.shape:
        raise ValueError("High-dM population mask has a different shape")
    if np.any(population & (resolved == 0) & (baseline < 0)):
        raise ValueError("Nres=0 High-dM event has no baseline source bin")
    invalid_recoil = population & ((recoil < 0) | (recoil >= len(RECOIL_LABELS)))
    if np.any(invalid_recoil):
        raise ValueError("baseline event has an invalid recoil index")
    output[population & (resolved == 0)] = baseline[population & (resolved == 0)]
    positions = topology_positions(top, w, resolved)
    moved = population & (resolved > 0)
    output[moved] = (
        NRES_SOURCE_OFFSET
        + positions[moved] * len(RECOIL_LABELS)
        + recoil[moved]
    )
    nb2_w2 = (
        population
        & (bottom == 2)
        & (top == 0)
        & (w == 2)
        & (resolved == 0)
    )
    output[nb2_w2] = ADDITIONAL_CATEGORY_SOURCE_OFFSET + recoil[nb2_w2]
    if np.any(
        population & ((output < 0) | (output >= EXCLUSIVE_SOURCE_BIN_COUNT))
    ):
        raise AssertionError(
            "High-dM event was not assigned to exactly one category-source bin"
        )
    return output


def exclusive_category_source_labels(source60_labels: Sequence[str]) -> list[str]:
    labels = [f"{label}__Nres0" for label in adopted55_labels(source60_labels)]
    labels.extend(
        f"{topology}__recoil_{recoil}"
        for topology in COARSE_NRES_TOPOLOGIES
        for recoil in RECOIL_LABELS
    )
    labels.extend(
        f"{category}__recoil_{recoil}"
        for category in ADDITIONAL_HIGHDM_CATEGORIES
        for recoil in RECOIL_LABELS
    )
    if len(labels) != EXCLUSIVE_SOURCE_BIN_COUNT:
        raise AssertionError("wrong exclusive High-dM source-label count")
    return labels


def configured_exclusive_mapping(
    configuration: dict[str, Any],
) -> tuple[int, ...]:
    """Return the exclusive category-source order requested by a scheme config.

    The configuration controls the visible block order and which Nres>0
    topology blocks are retained.  The resulting bin count is derived from
    that layout; callers must not assume a fixed bin count.
    """
    category_sizes = tuple(
        int(value) for value in configuration.get("baseline_category_sizes", ())
    )
    if not category_sizes or any(value <= 0 for value in category_sizes):
        raise ValueError("baseline_category_sizes must contain positive integers")
    if sum(category_sizes) != 55:
        raise ValueError(
            "baseline_category_sizes do not cover the adopted 55-bin baseline"
        )

    baseline_blocks: list[tuple[int, ...]] = []
    offset = 0
    for size in category_sizes:
        baseline_blocks.append(tuple(range(offset, offset + size)))
        offset += size
    topology_blocks = {
        topology: tuple(
            range(
                NRES_SOURCE_OFFSET + position * len(RECOIL_LABELS),
                NRES_SOURCE_OFFSET + (position + 1) * len(RECOIL_LABELS),
            )
        )
        for position, topology in enumerate(COARSE_NRES_TOPOLOGIES)
    }
    category_blocks = {
        category: tuple(
            range(
                ADDITIONAL_CATEGORY_SOURCE_OFFSET
                + position * len(RECOIL_LABELS),
                ADDITIONAL_CATEGORY_SOURCE_OFFSET
                + (position + 1) * len(RECOIL_LABELS),
            )
        )
        for position, category in enumerate(ADDITIONAL_HIGHDM_CATEGORIES)
    }

    mapping: list[int] = []
    seen_baseline: set[int] = set()
    seen_topologies: set[str] = set()
    seen_categories: set[str] = set()
    for block in configuration.get("layout", ()):
        if not isinstance(block, dict):
            raise ValueError("every configured layout block must be an object")
        kind = str(block.get("kind") or "")
        if kind == "baseline":
            index = int(block.get("index", -1))
            if index < 0 or index >= len(baseline_blocks):
                raise ValueError(f"invalid baseline block index {index}")
            if index in seen_baseline:
                raise ValueError(f"duplicate baseline block index {index}")
            seen_baseline.add(index)
            mapping.extend(baseline_blocks[index])
        elif kind == "topology":
            topology = str(block.get("name") or "")
            if topology not in topology_blocks:
                raise ValueError(f"unknown Nres topology {topology!r}")
            if topology in seen_topologies:
                raise ValueError(f"duplicate Nres topology {topology!r}")
            seen_topologies.add(topology)
            mapping.extend(topology_blocks[topology])
        elif kind == "category":
            category = str(block.get("name") or "")
            if category not in category_blocks:
                raise ValueError(f"unknown High-dM category {category!r}")
            if category in seen_categories:
                raise ValueError(f"duplicate High-dM category {category!r}")
            seen_categories.add(category)
            mapping.extend(category_blocks[category])
        else:
            raise ValueError(f"unknown configured layout block kind {kind!r}")

    expected_baseline = set(range(len(baseline_blocks)))
    if seen_baseline != expected_baseline:
        missing = sorted(expected_baseline - seen_baseline)
        raise ValueError(f"configured layout omits baseline blocks {missing}")
    omitted = set(str(value) for value in configuration.get("omitted_topologies", ()))
    if seen_topologies & omitted:
        raise ValueError("a topology cannot be both retained and omitted")
    if seen_topologies | omitted != set(COARSE_NRES_TOPOLOGIES):
        missing = sorted(set(COARSE_NRES_TOPOLOGIES) - seen_topologies - omitted)
        raise ValueError(
            "every Nres topology must be retained or explicitly omitted; "
            f"unaccounted={missing}"
        )
    omitted_categories = set(
        str(value) for value in configuration.get("omitted_categories", ())
    )
    if seen_categories & omitted_categories:
        raise ValueError("a category cannot be both retained and omitted")
    if seen_categories | omitted_categories != set(ADDITIONAL_HIGHDM_CATEGORIES):
        missing = sorted(
            set(ADDITIONAL_HIGHDM_CATEGORIES)
            - seen_categories
            - omitted_categories
        )
        raise ValueError(
            "every additional High-dM category must be retained or explicitly "
            f"omitted; unaccounted={missing}"
        )
    if len(mapping) != len(set(mapping)):
        raise AssertionError("configured exclusive layout reuses a source bin")
    return tuple(mapping)


def configured_bin_position_groups(
    configuration: dict[str, Any],
) -> tuple[tuple[int, ...], ...]:
    """Return final-bin groups in the pre-merge configured ordering.

    ``bin_merges_1based`` is deliberately expressed in the visible,
    pre-merge search-bin numbering.  This keeps review requests such as
    "merge bin 18 into bin 17" auditable without exposing internal source-bin
    identifiers.  Every unmentioned bin remains a singleton.
    """
    source_count = len(configured_exclusive_mapping(configuration))
    raw_merges = configuration.get("bin_merges_1based", ())
    if not isinstance(raw_merges, (list, tuple)):
        raise ValueError("bin_merges_1based must be a list of bin groups")

    merge_by_anchor: dict[int, tuple[int, ...]] = {}
    merged_positions: set[int] = set()
    for raw_group in raw_merges:
        if not isinstance(raw_group, (list, tuple)) or len(raw_group) < 2:
            raise ValueError("every configured bin merge must contain at least two bins")
        positions = tuple(int(value) - 1 for value in raw_group)
        if len(positions) != len(set(positions)):
            raise ValueError(f"configured bin merge repeats a bin: {raw_group}")
        if any(position < 0 or position >= source_count for position in positions):
            raise ValueError(
                f"configured bin merge is outside 1..{source_count}: {raw_group}"
            )
        if any(position in merged_positions for position in positions):
            raise ValueError(f"configured bin merges overlap at {raw_group}")
        ordered = tuple(sorted(positions))
        merge_by_anchor[ordered[0]] = ordered
        merged_positions.update(ordered)

    groups: list[tuple[int, ...]] = []
    for position in range(source_count):
        if position in merge_by_anchor:
            groups.append(merge_by_anchor[position])
        elif position not in merged_positions:
            groups.append((position,))
    covered = sorted(position for group in groups for position in group)
    if covered != list(range(source_count)):
        raise AssertionError("configured bin merges do not cover the source ordering once")
    return tuple(groups)


def configured_exclusive_groups(
    configuration: dict[str, Any],
) -> tuple[tuple[int, ...], ...]:
    """Return source-bin groups for every final configured search bin."""
    mapping = configured_exclusive_mapping(configuration)
    return tuple(
        tuple(mapping[position] for position in positions)
        for positions in configured_bin_position_groups(configuration)
    )


def configured_exclusive_bin_count(configuration: dict[str, Any]) -> int:
    """Return the final configured bin count after explicit bin merges."""
    return len(configured_bin_position_groups(configuration))


def map_category_sources_to_configured(
    indices: Sequence[int] | np.ndarray,
    configuration: dict[str, Any],
) -> np.ndarray:
    """Map exclusive category sources into the configured retained-bin ordering."""
    source = np.asarray(indices, dtype=int)
    lookup = np.full(EXCLUSIVE_SOURCE_BIN_COUNT, -1, dtype=np.int16)
    for target, sources in enumerate(configured_exclusive_groups(configuration)):
        lookup[list(sources)] = target
    output = np.full(source.shape, -1, dtype=np.int16)
    valid = (source >= 0) & (source < EXCLUSIVE_SOURCE_BIN_COUNT)
    output[valid] = lookup[source[valid]]
    return output


def configured_exclusive_labels(
    source60_labels: Sequence[str],
    configuration: dict[str, Any],
) -> list[str]:
    """Return labels in the exact order defined by the scheme config."""
    source_labels = exclusive_category_source_labels(source60_labels)
    return [
        "__plus__".join(source_labels[index] for index in sources)
        for sources in configured_exclusive_groups(configuration)
    ]
