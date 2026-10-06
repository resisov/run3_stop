"""Pure in-memory TROTA candidate builders, migrated without physics changes.

Pinned source hashes and function list: models/TROTA/provenance.json.
No ROOT writer, external inference cache or external analysis import.
"""
from __future__ import annotations
import math
import awkward as ak
import numpy as np
from numba import njit

RESOLVED_JETS = ("jet_nanoaod_pt", "jet_eta_all", "jet_phi_all", "jet_nanoaod_mass",
                 "jet_area", "jet_btag_upart_all", "jet_id_all")
JETS = RESOLVED_JETS + ("jet_source_index_all",)
FATJETS = ("fatjet_nanoaod_pt", "fatjet_eta_all", "fatjet_phi_all", "fatjet_nanoaod_mass",
           "fatjet_area", "fatjet_globalpart3_xbb_all", "fatjet_globalpart3_qcd_all",
           "fatjet_top_score_all", "fatjet_w_score_all", "fatjet_id_all", "fatjet_source_index_all")

@njit(cache=False)
def build_resolved_candidates(
    offsets: np.ndarray,
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    mass: np.ndarray,
    area: np.ndarray,
    btag: np.ndarray,
    jet_id: np.ndarray,
) -> tuple[np.ndarray, ...]:
    number_of_events = offsets.size - 1
    counts = np.zeros(number_of_events, dtype=np.int32)
    total_candidates = 0
    for event_index in range(number_of_events):
        number_good = 0
        for flat_index in range(offsets[event_index], offsets[event_index + 1]):
            if jet_id[flat_index] and pt[flat_index] > 25.0 and abs(eta[flat_index]) < 2.5:
                number_good += 1
        if number_good >= 3:
            count = number_good * (number_good - 1) * (number_good - 2) // 6
            counts[event_index] = count
            total_candidates += count

    features = np.empty((total_candidates, 3, 8), dtype=np.float32)
    candidate_event_index = np.empty(total_candidates, dtype=np.int32)
    candidate_index = np.empty(total_candidates, dtype=np.int32)
    good_idx0 = np.empty(total_candidates, dtype=np.int32)
    good_idx1 = np.empty(total_candidates, dtype=np.int32)
    good_idx2 = np.empty(total_candidates, dtype=np.int32)
    source_idx0 = np.empty(total_candidates, dtype=np.int32)
    source_idx1 = np.empty(total_candidates, dtype=np.int32)
    source_idx2 = np.empty(total_candidates, dtype=np.int32)
    candidate_pt = np.empty(total_candidates, dtype=np.float32)
    candidate_eta = np.empty(total_candidates, dtype=np.float32)
    candidate_phi = np.empty(total_candidates, dtype=np.float32)
    candidate_mass = np.empty(total_candidates, dtype=np.float32)

    output_index = 0
    for event_index in range(number_of_events):
        event_start = offsets[event_index]
        event_stop = offsets[event_index + 1]
        good_sources = np.empty(event_stop - event_start, dtype=np.int32)
        number_good = 0
        for flat_index in range(event_start, event_stop):
            if jet_id[flat_index] and pt[flat_index] > 25.0 and abs(eta[flat_index]) < 2.5:
                good_sources[number_good] = flat_index
                number_good += 1

        local_candidate_index = 0
        for idx0 in range(number_good):
            for idx1 in range(idx0):
                for idx2 in range(idx1):
                    source0 = good_sources[idx0]
                    source1 = good_sources[idx1]
                    source2 = good_sources[idx2]

                    sum_px = 0.0
                    sum_py = 0.0
                    sum_pz = 0.0
                    sum_energy = 0.0
                    sources = (source0, source1, source2)
                    for source in sources:
                        jet_pt = float(pt[source])
                        jet_eta = float(eta[source])
                        jet_phi = float(phi[source])
                        jet_mass = float(mass[source])
                        px = jet_pt * math.cos(jet_phi)
                        py = jet_pt * math.sin(jet_phi)
                        pz = jet_pt * math.sinh(jet_eta)
                        energy2 = jet_mass * jet_mass + px * px + py * py + pz * pz
                        sum_px += px
                        sum_py += py
                        sum_pz += pz
                        sum_energy += math.sqrt(max(energy2, 0.0))

                    top_pt = math.hypot(sum_px, sum_py)
                    top_phi = math.atan2(sum_py, sum_px)
                    top_eta = math.asinh(sum_pz / top_pt) if top_pt > 0.0 else 0.0
                    top_mass2 = (
                        sum_energy * sum_energy
                        - sum_px * sum_px
                        - sum_py * sum_py
                        - sum_pz * sum_pz
                    )
                    top_mass = math.sqrt(max(top_mass2, 0.0))

                    for leg_index in range(3):
                        source = sources[leg_index]
                        jet_minus_top_phi = float(phi[source]) - top_phi
                        while jet_minus_top_phi > math.pi:
                            jet_minus_top_phi -= 2.0 * math.pi
                        while jet_minus_top_phi < -math.pi:
                            jet_minus_top_phi += 2.0 * math.pi
                        jet_phi = float(phi[source])
                        while jet_phi > math.pi:
                            jet_phi -= 2.0 * math.pi
                        while jet_phi < -math.pi:
                            jet_phi += 2.0 * math.pi
                        features[output_index, leg_index, 0] = np.float32(area[source])
                        features[output_index, leg_index, 1] = np.float32(btag[source])
                        features[output_index, leg_index, 2] = np.float32(
                            float(eta[source]) - top_eta
                        )
                        features[output_index, leg_index, 3] = np.float32(mass[source])
                        features[output_index, leg_index, 4] = np.float32(
                            jet_minus_top_phi
                        )
                        features[output_index, leg_index, 5] = np.float32(pt[source])
                        features[output_index, leg_index, 6] = np.float32(jet_phi)
                        features[output_index, leg_index, 7] = np.float32(eta[source])

                    candidate_event_index[output_index] = event_index
                    candidate_index[output_index] = local_candidate_index
                    good_idx0[output_index] = idx0
                    good_idx1[output_index] = idx1
                    good_idx2[output_index] = idx2
                    source_idx0[output_index] = source0 - event_start
                    source_idx1[output_index] = source1 - event_start
                    source_idx2[output_index] = source2 - event_start
                    candidate_pt[output_index] = np.float32(top_pt)
                    candidate_eta[output_index] = np.float32(top_eta)
                    candidate_phi[output_index] = np.float32(top_phi)
                    candidate_mass[output_index] = np.float32(top_mass)
                    output_index += 1
                    local_candidate_index += 1

    return (
        counts,
        features,
        candidate_event_index,
        candidate_index,
        good_idx0,
        good_idx1,
        good_idx2,
        source_idx0,
        source_idx1,
        source_idx2,
        candidate_pt,
        candidate_eta,
        candidate_phi,
        candidate_mass,
    )


def _flatten_jets(arrays: ak.Array, jet_branches=RESOLVED_JETS) -> tuple[np.ndarray, ...]:
    reference_counts = np.asarray(ak.to_numpy(ak.num(arrays[jet_branches[0]])), dtype=np.int64)
    for branch in jet_branches[1:]:
        counts = np.asarray(ak.to_numpy(ak.num(arrays[branch])), dtype=np.int64)
        if not np.array_equal(counts, reference_counts):
            raise ValueError(f"unaligned intermediate jet vector: {branch}")
    offsets = np.empty(reference_counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(reference_counts, out=offsets[1:])

    flattened = []
    dtypes = (
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.bool_,
    )
    for branch, dtype in zip(jet_branches, dtypes):
        values = ak.to_numpy(ak.flatten(arrays[branch], axis=1))
        flattened.append(np.asarray(values, dtype=dtype))
    return (offsets, *flattened)


@njit(cache=False)
def dphi(a, b):
    x = a - b
    while x > math.pi:
        x -= 2 * math.pi
    while x < -math.pi:
        x += 2 * math.pi
    return x


@njit(cache=False)
def p4(row):
    pt, eta, phi, mass = row[:4]
    px, py, pz = pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)
    return np.array((px, py, pz, math.sqrt(px*px + py*py + pz*pz + mass*mass)))


@njit(cache=False)
def kin(v):
    px, py, pz, energy = v
    pt = math.sqrt(px*px + py*py)
    # ROOT TVector3::PseudoRapidity convention at the zero-pT boundary.
    eta = math.asinh(pz/pt) if pt > 0 else (1e11 if pz > 0 else (-1e11 if pz < 0 else 0.))
    m2 = energy*energy - (px*px + py*py + pz*pz)
    mass = math.sqrt(m2) if m2 >= 0 else -math.sqrt(-m2)
    return pt, eta, math.atan2(py, px), mass


@njit(cache=False)
def build_mixed_candidates(jo, jets, fo, fatjets):
    """Float64 four-vectors, float32 inputs, official leg/enumeration order."""
    n = len(jo)-1
    capacity = 0
    # Candidate, eligible-AK8 and event counts, each split into zero/negative.
    invalid = np.zeros(6, np.int64)
    for ev in range(n):
        nj, nf = 0, 0
        zero, negative = 0, 0
        for i in range(jo[ev], jo[ev+1]):
            if jets[i, 6] != 0 and jets[i, 0] > 25 and abs(jets[i, 1]) < 2.5:
                nj += 1
        for i in range(fo[ev], fo[ev+1]):
            if fatjets[i, 9] != 0 and fatjets[i, 0] > 200 and abs(fatjets[i, 1]) < 2.5:
                nf += 1
                denom = fatjets[i, 5]+fatjets[i, 6]
                zero += int(denom==0)
                negative += int(denom<0)
        if nj>=2:
            invalid[2] += zero
            invalid[3] += negative
            invalid[4] += int(zero>0)
            invalid[5] += int(negative>0)
        capacity += nf*(nj*(nj-1)//2 + nj*(nj-1)*(nj-2)//6)
    jf = np.zeros((capacity, 3, 8), np.float32)
    ff = np.empty((capacity, 9), np.float32)
    tf = np.empty((capacity, 3), np.float32)
    ints = np.empty((capacity, 10), np.int32)
    kinematics = np.empty((capacity, 4), np.float32)
    events = np.empty(capacity, np.int64)
    out = 0
    for ev in range(n):
        js = [i for i in range(jo[ev], jo[ev+1])
              if jets[i, 6] != 0 and jets[i, 0] > 25 and abs(jets[i, 1]) < 2.5]
        fs = [i for i in range(fo[ev], fo[ev+1])
              if fatjets[i, 9] != 0 and fatjets[i, 0] > 200 and abs(fatjets[i, 1]) < 2.5]
        ci = 0
        for j0 in range(len(js)):
            for j1 in range(j0):
                # -1 is the pair; then 0..j1-1 are the triplets.
                for j2 in range(-1, j1):
                    for fi in range(len(fs)):
                        legids = (j0, j1, j2)
                        nleg = 2 if j2 == -1 else 3
                        fj = fatjets[fs[fi]]
                        denom = fj[5] + fj[6]
                        if denom <= 0:
                            # User-adopted policy: only this candidate is
                            # untagged; never invent a model feature or discard
                            # the event. Keep AK8 indices and other candidates.
                            invalid[0 if denom==0 else 1] += 1
                            ci += 1
                            continue
                        s = p4(jets[js[j0]]) + p4(jets[js[j1]])
                        if nleg == 3:
                            s += p4(jets[js[j2]])
                        sk = kin(s)
                        overlap = np.zeros(3, np.bool_)
                        for leg in range(nleg):
                            j = jets[js[legids[leg]]]
                            overlap[leg] = math.hypot(j[1]-fj[1], dphi(j[2], fj[2])) < .8
                        # Upstream top3j1fj without any overlap deliberately uses
                        # only the three AK4 jets; preserve that nontrivial rule.
                        if nleg == 3 and not np.any(overlap):
                            top = s.copy()
                        else:
                            top = p4(fj)
                            for leg in range(nleg):
                                if not overlap[leg]:
                                    top += p4(jets[js[legids[leg]]])
                        tk = kin(top)
                        if tk[0] <= 0:
                            continue
                        ints[out, 0] = ci
                        ints[out, 1] = 2 if nleg == 2 else 0
                        ints[out, 2] = fi
                        ints[out, 6] = int(fj[10])
                        for leg in range(3):
                            ints[out, 3+leg] = legids[leg]
                            ints[out, 7+leg] = int(jets[js[legids[leg]], 7]) if leg < nleg else -1
                        for leg in range(nleg):
                            j = jets[js[legids[leg]]]
                            jf[out, leg, 0] = j[4]
                            jf[out, leg, 1] = j[5]
                            jf[out, leg, 2] = j[1]-sk[1]
                            jf[out, leg, 3] = j[3]
                            jf[out, leg, 4] = dphi(j[2], sk[2])
                            jf[out, leg, 5] = j[0]
                            jf[out, leg, 6] = dphi(j[2], fj[2])
                            jf[out, leg, 7] = j[1]-fj[1]
                        ff[out] = np.array((fj[4], fj[5]/denom, fj[6], fj[7], fj[8],
                                            fj[1], fj[3], fj[2], fj[0]), np.float32)
                        tf[out] = np.array((sk[3], tk[3], tk[0]), np.float32)
                        kinematics[out] = np.array(tk, np.float32)
                        events[out] = ev
                        out += 1
                        ci += 1
    return jf[:out], ff[:out], tf[:out], ints[:out], kinematics[:out], events[:out], invalid


def pack(a, names):
    sizes = np.asarray(ak.num(a[names[0]]), dtype=np.int64)
    for name in names:
        if not np.array_equal(sizes, np.asarray(ak.num(a[name]))):
            raise ValueError("Inconsistent object array lengths: " + name)
    values = np.column_stack([np.asarray(ak.flatten(a[k]), dtype=np.float64) for k in names])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite Mixed inputs")
    return np.r_[0, np.cumsum(sizes)].astype(np.int64), values


def candidate_cost(a):
    j = ((a[JETS[6]] != 0) & (a[JETS[0]] > 25) & (abs(a[JETS[1]]) < 2.5))
    f = ((a[FATJETS[9]] != 0) & (a[FATJETS[0]] > 200) & (abs(a[FATJETS[1]]) < 2.5))
    nj, nf = np.asarray(ak.sum(j, axis=1), dtype=np.int64), np.asarray(ak.sum(f, axis=1), dtype=np.int64)
    return nf*(nj*(nj-1)//2 + nj*(nj-1)*(nj-2)//6)
