from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


SUBJET_MATCH_DR = 0.4
FATJET_FALLBACK_DR = 0.8


# Run-3 coarse projection of the Run-2 Table-16 (Nt, NW, Nres) topology
# lattice.  The adopted 55-bin categories keep only Nres=0 events; every
# baseline event with Nres>0 moves to exactly one of these five labels.
COARSE_NRES_TOPOLOGIES = (
    "resolved1_only",
    "resolved2plus_only",
    "top_resolved",
    "w_resolved",
    "top_w_resolved",
)


def _combined_event_key(file_id: np.ndarray, entry: np.ndarray) -> np.ndarray:
    file_values = np.asarray(file_id, dtype=np.uint64)
    entry_values = np.asarray(entry, dtype=np.uint64)
    if np.any(file_values >= (1 << 31)) or np.any(entry_values >= (1 << 32)):
        raise RuntimeError("file_id/entry exceeds the validated 31/32-bit join key")
    return (file_values << np.uint64(32)) | entry_values


def map_candidates_to_events(
    event_file_id: np.ndarray,
    event_entry: np.ndarray,
    candidate_file_id: np.ndarray,
    candidate_entry: np.ndarray,
) -> np.ndarray:
    """Map sparse candidates to Events using the primary file/entry identity."""
    event_keys = _combined_event_key(event_file_id, event_entry)
    candidate_keys = _combined_event_key(candidate_file_id, candidate_entry)
    if np.unique(event_keys).size != event_keys.size:
        raise RuntimeError("(file_id, entry) is not unique in Events")
    order = np.argsort(event_keys, kind="stable")
    sorted_keys = event_keys[order]
    positions = np.searchsorted(sorted_keys, candidate_keys)
    if np.any(positions >= sorted_keys.size):
        raise RuntimeError("TROTA candidate does not map to an Events entry")
    mapped = order[positions]
    if not np.array_equal(event_keys[mapped], candidate_keys):
        raise RuntimeError("TROTA candidate identity differs from Events identity")
    return np.asarray(mapped, dtype=np.int64)


def map_candidates_to_events_rle(
    event_run: np.ndarray,
    event_lumi: np.ndarray,
    event_number: np.ndarray,
    candidate_run: np.ndarray,
    candidate_lumi: np.ndarray,
    candidate_number: np.ndarray,
) -> np.ndarray:
    """Fallback sparse-candidate join using an exact run/lumi/event tuple."""
    event_keys = list(
        zip(
            np.asarray(event_run).tolist(),
            np.asarray(event_lumi).tolist(),
            np.asarray(event_number).tolist(),
        )
    )
    lookup = {key: index for index, key in enumerate(event_keys)}
    if len(lookup) != len(event_keys):
        raise RuntimeError("(run, luminosityBlock, event) is not unique in Events")
    candidate_keys = zip(
        np.asarray(candidate_run).tolist(),
        np.asarray(candidate_lumi).tolist(),
        np.asarray(candidate_number).tolist(),
    )
    mapped: list[int] = []
    for key in candidate_keys:
        if key not in lookup:
            raise RuntimeError("TROTA run/lumi/event identity does not map to Events")
        mapped.append(lookup[key])
    return np.asarray(mapped, dtype=np.int64)


def delta_phi(phi1: float, phi2: float) -> float:
    """Return the wrapped azimuthal separation in [-pi, pi)."""
    return (float(phi1) - float(phi2) + math.pi) % (2.0 * math.pi) - math.pi


def delta_r2(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    deta = float(eta1) - float(eta2)
    dphi = delta_phi(phi1, phi2)
    return deta * deta + dphi * dphi


def _valid_subjet_indices(
    index1: int,
    index2: int,
    subjet_count: int,
) -> list[int]:
    indices: list[int] = []
    for value in (int(index1), int(index2)):
        if 0 <= value < subjet_count and value not in indices:
            indices.append(value)
    return indices


def boosted_overlap_vetoed_ak4_indices(
    *,
    jet_source_indices: Sequence[int],
    jet_eta: Sequence[float],
    jet_phi: Sequence[float],
    fatjet_eta: Sequence[float],
    fatjet_phi: Sequence[float],
    fatjet_subjet_index1: Sequence[int],
    fatjet_subjet_index2: Sequence[int],
    fatjet_top_pass: Sequence[int | bool],
    fatjet_w_pass: Sequence[int | bool],
    subjet_eta: Sequence[float],
    subjet_phi: Sequence[float],
) -> frozenset[int]:
    """Return AK4 source indices removed by selected boosted top/W objects.

    This is the Run-2 prescription: use dR < 0.4 around each of two valid
    subjets, or dR < 0.8 around the AK8 axis when fewer than two valid subjets
    are available.  Only FatJets passing the stored top or W selection enter.
    """
    jet_lengths = {len(jet_source_indices), len(jet_eta), len(jet_phi)}
    if len(jet_lengths) != 1:
        raise ValueError("AK4 identity and kinematic arrays have different lengths")
    fatjet_lengths = {
        len(fatjet_eta),
        len(fatjet_phi),
        len(fatjet_subjet_index1),
        len(fatjet_subjet_index2),
        len(fatjet_top_pass),
        len(fatjet_w_pass),
    }
    if len(fatjet_lengths) != 1:
        raise ValueError("FatJet identity, selection, and kinematic arrays differ in length")
    if len(subjet_eta) != len(subjet_phi):
        raise ValueError("SubJet eta and phi arrays have different lengths")

    vetoed: set[int] = set()
    for fatjet_index in range(len(fatjet_eta)):
        if not (
            bool(fatjet_top_pass[fatjet_index])
            or bool(fatjet_w_pass[fatjet_index])
        ):
            continue
        subjet_indices = _valid_subjet_indices(
            int(fatjet_subjet_index1[fatjet_index]),
            int(fatjet_subjet_index2[fatjet_index]),
            len(subjet_eta),
        )
        if len(subjet_indices) >= 2:
            axes = [
                (float(subjet_eta[index]), float(subjet_phi[index]), SUBJET_MATCH_DR)
                for index in subjet_indices
            ]
        else:
            axes = [
                (
                    float(fatjet_eta[fatjet_index]),
                    float(fatjet_phi[fatjet_index]),
                    FATJET_FALLBACK_DR,
                )
            ]

        for source_index, eta, phi in zip(jet_source_indices, jet_eta, jet_phi):
            if any(
                delta_r2(float(eta), float(phi), axis_eta, axis_phi) < radius * radius
                for axis_eta, axis_phi, radius in axes
            ):
                vetoed.add(int(source_index))
    return frozenset(vetoed)


@dataclass(frozen=True)
class ResolvedSelection:
    selected_candidate_indices: tuple[int, ...]
    rejected_by_boosted_overlap: tuple[int, ...]
    rejected_by_resolved_overlap: tuple[int, ...]

    @property
    def nres(self) -> int:
        return len(self.selected_candidate_indices)


def select_exclusive_resolved_candidates(
    *,
    candidate_indices: Sequence[int],
    candidate_scores: Sequence[float],
    candidate_source_jets: Sequence[Sequence[int]],
    boosted_vetoed_ak4_indices: Iterable[int] = (),
) -> ResolvedSelection:
    """Select deterministic, jet-disjoint resolved candidates.

    Candidates touching a boosted-object AK4 veto are removed first.  The
    remainder are considered by descending discriminator, with candidateIndex
    as the deterministic tie-breaker.  Arbitrarily large sourceJetIdx values
    are supported; no fixed-width bit mask is used.
    """
    lengths = {
        len(candidate_indices),
        len(candidate_scores),
        len(candidate_source_jets),
    }
    if len(lengths) != 1:
        raise ValueError("resolved candidate arrays have different lengths")

    boosted_veto = {int(value) for value in boosted_vetoed_ak4_indices}
    candidates: list[tuple[int, float, frozenset[int]]] = []
    rejected_boosted: list[int] = []
    for candidate_index, score, source_jets in zip(
        candidate_indices, candidate_scores, candidate_source_jets,
    ):
        index = int(candidate_index)
        jets = frozenset(int(value) for value in source_jets if int(value) >= 0)
        if len(jets) != 3:
            raise ValueError(
                f"candidate {index} does not contain three distinct valid AK4 source indices"
            )
        if jets & boosted_veto:
            rejected_boosted.append(index)
            continue
        candidates.append((index, float(score), jets))

    candidates.sort(key=lambda item: (-item[1], item[0]))
    used_jets: set[int] = set()
    selected: list[int] = []
    rejected_resolved: list[int] = []
    for candidate_index, _score, source_jets in candidates:
        if used_jets & source_jets:
            rejected_resolved.append(candidate_index)
            continue
        selected.append(candidate_index)
        used_jets.update(source_jets)

    return ResolvedSelection(
        selected_candidate_indices=tuple(selected),
        rejected_by_boosted_overlap=tuple(sorted(rejected_boosted)),
        rejected_by_resolved_overlap=tuple(rejected_resolved),
    )


def high_mtb_topology_label(
    *,
    nb: int,
    nt: int,
    nw: int,
    nres: int,
    high_mtb: bool,
) -> str | None:
    """Assign an exclusive Run-2-style High-dM topology label.

    The Nb=1 categories use presence/absence, as in Run-2.  Nb>=2 categories
    retain exact one- and two-object compositions and collect sums >=3 in a
    dedicated overflow.  A triple-object Nb=1 overflow is explicit instead of
    being silently discarded.
    """
    nb = int(nb)
    nt = int(nt)
    nw = int(nw)
    nres = int(nres)
    if min(nb, nt, nw, nres) < 0:
        raise ValueError("category multiplicities must be non-negative")
    if not high_mtb or nb < 1:
        return None

    nb_label = "nb1" if nb == 1 else ("nb2" if nb == 2 else "nb3plus")
    if nb == 1:
        presence = (int(nt > 0), int(nw > 0), int(nres > 0))
        labels = {
            (0, 0, 0): "none",
            (1, 0, 0): "top",
            (0, 1, 0): "w",
            (0, 0, 1): "resolved",
            (1, 1, 0): "top_w",
            (1, 0, 1): "top_resolved",
            (0, 1, 1): "w_resolved",
            (1, 1, 1): "sum3plus",
        }
        return f"{nb_label}_{labels[presence]}"

    total = nt + nw + nres
    if total >= 3:
        return f"{nb_label}_sum3plus"
    exact_labels = {
        (0, 0, 0): "none",
        (1, 0, 0): "top1",
        (0, 1, 0): "w1",
        (0, 0, 1): "resolved1",
        (1, 1, 0): "top1_w1",
        (1, 0, 1): "top1_resolved1",
        (0, 1, 1): "w1_resolved1",
        (2, 0, 0): "top2",
        (0, 2, 0): "w2",
        (0, 0, 2): "resolved2",
    }
    label = exact_labels.get((nt, nw, nres))
    if label is None:
        raise RuntimeError(
            f"uncovered High-dM topology: Nb={nb}, Nt={nt}, NW={nw}, Nres={nres}"
        )
    return f"{nb_label}_{label}"


def coarse_nres_topology_label(
    *,
    nt: int,
    nw: int,
    nres: int,
    high_mtb: bool,
) -> str | None:
    """Return the exclusive coarse Nres>0 topology used by the Run-3 study.

    This is a deliberately smaller projection of the Run-2 Table-16 category
    lattice.  It preserves the physics distinctions explicitly requested for
    the 55-bin extension: one versus at least two resolved-only candidates,
    coexistence with a boosted top, coexistence with a boosted W, and the
    three-object overflow.  Nres=0 remains in the corresponding adopted
    55-bin category and therefore has no additional label here.
    """
    nt = int(nt)
    nw = int(nw)
    nres = int(nres)
    if min(nt, nw, nres) < 0:
        raise ValueError("topology multiplicities must be non-negative")
    if not high_mtb or nres == 0:
        return None
    if nt == 0 and nw == 0:
        return "resolved1_only" if nres == 1 else "resolved2plus_only"
    if nt > 0 and nw == 0:
        return "top_resolved"
    if nt == 0 and nw > 0:
        return "w_resolved"
    return "top_w_resolved"
