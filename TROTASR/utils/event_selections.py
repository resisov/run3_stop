"""All analysis event selections, independent of histogram and plotting code.

2026-09-23: user-authorized TEST of B > M > R > W constituent arbitration.
The only new topology cuts are the Mixed high-dM union and low-dM veto.
Other high-dM and diagonal-GNN low-dM cuts retain the adopted definitions.
"""
from __future__ import annotations
import numpy as np
import awkward as ak
from .region_kinematics import build_region_blocks, dycr_lepton_mask, object_masks, clean_by_delta_r
from .ids import delta_phi
from .constituents import boosted_overlap_vetoed_ak4_indices

REGIONS = ("SR", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M")
MIXED_WP = 0.9027690887451172
CATEGORY_ORDER = ("Nb", "Nbst", "Nmix", "Nres", "Nw")
MIXED_FIELDS = ("candidateIndex", "sourceFatJetIdx", "sourceJetIdx0", "sourceJetIdx1",
                "sourceJetIdx2", "QCDDiscriminant")
OVERLAP_FIELDS = ("jet_source_index_all", "jet_eta_all", "jet_phi_all", "fatjet_source_index_all",
                  "fatjet_eta_all", "fatjet_phi_all", "fatjet_subjet_index1_all", "fatjet_subjet_index2_all",
                  "fatjet_boosted_top_pass_all", "fatjet_boosted_w_pass_all", "subjet_eta_all", "subjet_phi_all")


def arbitrate(boosted, mixed, resolved, w_candidates, footprints):
    """Deterministic disjoint constituents, independent W retained.

    AK8 footprint means the already-adopted two-subjet dR<0.4, otherwise
    AK8-axis dR<0.8 AK4 association, not a newly invented cone size.
    """
    chosen_b = sorted(set(boosted))
    occupied_fat = set(chosen_b)
    occupied_jets = set().union(*(footprints[f] for f in chosen_b)) if chosen_b else set()
    chosen_m, chosen_r, rejected = [], [], {"mixed": 0, "resolved": 0, "w": 0}
    for c in sorted(mixed, key=lambda x: (-x["score"], x["index"])):
        constituents = set(c["jets"]) | footprints[c["fat"]]
        if c["fat"] in occupied_fat or constituents & occupied_jets:
            rejected["mixed"] += 1
            continue
        chosen_m.append(c)
        occupied_fat.add(c["fat"])
        occupied_jets.update(constituents)
    for c in sorted(resolved, key=lambda x: (-x["score"], x["index"])):
        if set(c["jets"]) & occupied_jets:
            rejected["resolved"] += 1
            continue
        chosen_r.append(c)
        occupied_jets.update(c["jets"])
    chosen_w = []
    for f in sorted(set(w_candidates)):
        if f in occupied_fat or footprints[f] & occupied_jets:
            rejected["w"] += 1
            continue
        chosen_w.append(f)
        # Existing W multiplicity counts independent AK8s; do not invent W-W
        # overlap removal in a change authorized for top/W exclusivity.
    return dict(B=chosen_b, M=chosen_m, R=chosen_r, W=chosen_w, rejected=rejected)


def _references(a, region):
    if region in ("SR", "LLCR", "QCDCR"):
        empty = a["jet_eta_all"][:, :0]
        return empty, empty
    flavor = {"GCR": "photon", "DY2E": "electron", "DY2M": "muon"}[region]
    selected = object_masks(a)[flavor + "_medium"]
    return a[flavor + "_eta_all"][selected], a[flavor + "_phi_all"][selected]


def topology_buffers(a):
    """Chunk-local buffers shared by regions; no decisions cached across shifts."""
    fields = set(ak.fields(a))
    needed = set(OVERLAP_FIELDS) | {"TopMixed1pct_" + n for n in MIXED_FIELDS} | {"nTopMixed1pct"}
    if needed - fields:
        raise ValueError("Missing topology input: " + ",".join(sorted(needed - fields)))
    columns = {k: ak.to_list(a[k]) for k in
               (*OVERLAP_FIELDS, *("TopMixed1pct_" + x for x in MIXED_FIELDS))}
    scalar = {k: np.asarray(a[k]) for k in ('nTopMixed1pct', 'file_id', 'entry')}
    footprints = []
    for i in range(len(a)):
        event = {k: columns[k][i] for k in OVERLAP_FIELDS}
        fj, jj = event['fatjet_source_index_all'], event['jet_source_index_all']
        if len(set(fj)) != len(fj) or len(set(jj)) != len(jj):
            raise ValueError("Duplicate original object identity")
        associations = {}
        for k, source in enumerate(fj):
            associations[source] = set(boosted_overlap_vetoed_ak4_indices(
                jet_source_indices=jj, jet_eta=event['jet_eta_all'], jet_phi=event['jet_phi_all'],
                fatjet_eta=event['fatjet_eta_all'], fatjet_phi=event['fatjet_phi_all'],
                fatjet_subjet_index1=event['fatjet_subjet_index1_all'],
                fatjet_subjet_index2=event['fatjet_subjet_index2_all'],
                fatjet_top_pass=[j == k for j in range(len(fj))], fatjet_w_pass=[False]*len(fj),
                subjet_eta=event['subjet_eta_all'], subjet_phi=event['subjet_phi_all']))
        footprints.append(associations)
    return columns, scalar, footprints


def topology_counts(a, resolved_by_identity, region, prepared=None):
    columns_all, scalar, footprints_all = topology_buffers(a) if prepared is None else prepared
    eta, phi = _references(a, region)
    clean_fat = clean_by_delta_r(a["fatjet_eta_all"], a["fatjet_phi_all"], eta, phi, .4)
    clean_jet = clean_by_delta_r(a["jet_eta_all"], a["jet_phi_all"], eta, phi, .2)
    n = len(a)
    counts = np.zeros((n, 4), dtype=np.int32)
    baseline = np.zeros((n, 4), dtype=np.int32)
    sf_clean, audits = [], {"mixed_overlap": 0, "resolved_overlap": 0, "w_overlap": 0, "mixed_and_independent_w": 0}
    fat_keep_all, jet_keep_all = ak.to_list(clean_fat), ak.to_list(clean_jet)
    for i in range(n):
        event = {k: columns_all[k][i] for k in OVERLAP_FIELDS}
        fj = event["fatjet_source_index_all"]
        jj = event["jet_source_index_all"]
        fat_keep, jet_keep = fat_keep_all[i], jet_keep_all[i]
        good_fat = {j for j, good in zip(fj, fat_keep) if good}
        good_jet = {j for j, good in zip(jj, jet_keep) if good}
        footprints = footprints_all[i]
        boosted = [j for j, passed in zip(fj, event["fatjet_boosted_top_pass_all"]) if passed and j in good_fat]
        ws = [j for j, passed in zip(fj, event["fatjet_boosted_w_pass_all"]) if passed and j in good_fat]
        columns = {k: columns_all["TopMixed1pct_" + k][i] for k in MIXED_FIELDS}
        if any(len(v) != int(scalar["nTopMixed1pct"][i]) for v in columns.values()):
            raise ValueError("Mixed candidate branch lengths differ")
        mixed = []
        for k, index in enumerate(columns["candidateIndex"]):
            fat = columns["sourceFatJetIdx"][k]
            jets = [columns["sourceJetIdx" + str(j)][k] for j in range(3)]
            jets = [j for j in jets if j >= 0]
            score = columns["QCDDiscriminant"][k]
            if fat not in footprints or not set(jets) <= set(jj) or len(jets) != len(set(jets)):
                raise ValueError("Invalid Mixed original constituent index")
            if not np.isfinite(score) or score < MIXED_WP:
                raise ValueError("Invalid stored Mixed discriminant")
            if fat in good_fat and set(jets) <= good_jet:
                mixed.append(dict(index=index, fat=fat, jets=jets, score=score))
        key = (int(scalar["file_id"][i]), int(scalar["entry"][i]))
        resolved = []
        for c in resolved_by_identity.get(key, []):
            if not set(c["jets"]) <= set(jj):
                raise ValueError("Invalid Resolved original constituent index")
            if set(c["jets"]) <= good_jet and abs(c["eta"]) < 2. and 100. <= c["mass"] <= 250.:
                resolved.append(c)
        selection = arbitrate(boosted, mixed, resolved, ws, footprints)
        counts[i] = [len(selection[k]) for k in ("B", "M", "R", "W")]
        old = arbitrate(boosted + ws, [], resolved, [], footprints)
        baseline[i] = [len(boosted), 0, len(old['R']), len(ws)]
        used_m_fat = {c["fat"] for c in selection["M"]}
        used_mr_jets = set().union(*(set(c["jets"]) for c in selection["M"] + selection["R"]))
        used_mr_jets.update(set().union(*(footprints[f] for f in used_m_fat)))
        sf_clean.append([good and (f in boosted or (f not in used_m_fat and not (footprints[f] & used_mr_jets)))
                         for f, good in zip(fj, fat_keep)])
        for src, dst in (("mixed", "mixed_overlap"), ("resolved", "resolved_overlap"), ("w", "w_overlap")):
            audits[dst] += selection["rejected"][src]
        audits["mixed_and_independent_w"] += bool(selection["M"] and selection["W"])
    return counts, ak.Array(sf_clean), audits, baseline


def legacy_vr_mask(chunk, region, flag, n):
    """Adopted high-dM VR non-topology mask, used also for trigger SF support."""
    get = lambda key: np.asarray(chunk[key])
    d = [get("j%d_met_dphi" % k) for k in range(1, 5)]
    nb = get("nb_medium")
    group = {"HighDMVR_Nb1": nb == 1, "HighDMVR_Nb2": nb == 2, "HighDMVR_Nb3plus": nb >= 3}[region]
    return (get("pass_base_common").astype(bool) & get("pass_signal_trigger").astype(bool)
            & get("pass_zero_tau").astype(bool) & get("pass_no_veto_leptons").astype(bool)
            & (get("njet") >= 5) & group & get("pass_met_250").astype(bool) & get("pass_ht_300").astype(bool)
            & (d[0] > .5) & (d[1] > .15) & (d[2] > .15) & ((d[1] < .5) | (d[2] < .5) | (d[3] < .5)))


def highdm_core(a, blocks, region, dy_mass_window=(71., 111.)):
    get = lambda k: np.asarray(a[k])
    base = get("pass_base_common").astype(bool) & get("pass_zero_tau").astype(bool)
    if region in ("SR", "LLCR", "QCDCR"):
        core = base & get("pass_signal_trigger").astype(bool) & (get("njet") >= 4) & (get("nb_medium") >= 1)
        core &= get("pass_met_250").astype(bool) & get("pass_ht_300").astype(bool)
        if region == "LLCR":
            return core & get("pass_one_veto_lepton").astype(bool) & get("pass_mt_100").astype(bool) & get("pass_open_high").astype(bool)
        if region == "QCDCR":
            return core & get("pass_no_veto_leptons").astype(bool) & get("pass_qcd_open").astype(bool) & get("pass_dphi123_0p1").astype(bool)
        return core & get("pass_no_veto_leptons").astype(bool) & get("pass_open_high").astype(bool)
    if region == "GCR":
        # The intermediate schema does not store pass_gcr_open_high. Rebuild
        # the same four-jet angular predicate from its retained AK4 columns.
        eta, phi = _references(a, region)
        good = ((a['jet_corrected_pt'] > 30.) & (abs(a['jet_eta_all']) < 2.4)
                & ak.values_astype(a['jet_id_all'], np.bool_)
                & clean_by_delta_r(a['jet_eta_all'], a['jet_phi_all'], eta, phi, .2))
        angles = delta_phi(a['jet_phi_all'][good][:, :4], get('recoil_gcr_phi')[:, None])
        opened = np.asarray(ak.all(angles > .5, axis=1))
        b = blocks[region]
        return b.core & (b.njet >= 4) & (b.nb >= 1) & opened
    return (dycr_lepton_mask(a, region, mass_window=dy_mass_window)
            & get("pass_" + region.lower() + "_open_high").astype(bool)
            & (get("njet_lepton_clean") >= 4) & (get("nb_lepton_clean") >= 1)
            & (get("ht_lepton_clean") > 300.) & (get("recoil_" + region.lower()) > 250.))


class FixedJetTopologyCache:
    """Chunk-local reuse only when every jet/candidate input is exactly fixed.

    Non-JME endpoints keep these columns unchanged. Region cleaning is reused
    only for identical selected reference directions; threshold crossings miss
    the cache and run the original arbitration. No event selections are cached.
    """
    def __init__(self, a, resolved):
        self.resolved = resolved
        self.prepared = topology_buffers(a)
        names = (*OVERLAP_FIELDS, *("TopMixed1pct_" + x for x in MIXED_FIELDS),
                 'nTopMixed1pct', 'file_id', 'entry')
        self.inputs = {name: self.signature(a[name]) for name in names}
        self.regions = {}

    @staticmethod
    def signature(array):
        counts = np.asarray(ak.num(array, axis=1)) if array.layout.purelist_depth > 1 else None
        return counts, np.asarray(ak.flatten(array, axis=None))

    def validate(self, a, resolved):
        if resolved is not self.resolved:
            raise ValueError('Fixed-jet cache received different resolved candidates')
        for name, before in self.inputs.items():
            after = self.signature(a[name])
            if any(not np.array_equal(x, y) for x, y in zip(before, after)):
                raise ValueError('Fixed-jet topology input changed: ' + name)

    def get(self, a, region):
        eta, phi = _references(a, region)
        reference = (ak.to_list(eta), ak.to_list(phi))
        old = self.regions.get(region)
        if old is None or old[0] != reference:
            value = topology_counts(a, self.resolved, region, self.prepared)
            self.regions[region] = reference, value
        return self.regions[region][1]


def select(a, resolved_by_identity, dy_mass_window=(71., 111.), topology_cache=None):
    blocks, audit = build_region_blocks(a, dy_mass_window=dy_mass_window)
    if topology_cache is None:
        prepared = topology_buffers(a)
    else:
        topology_cache.validate(a, resolved_by_identity)
    output = {}
    for region in REGIONS:
        if region in ('LLCR', 'QCDCR'):
            counts, sf_clean, caudit, baseline = shared_topology
        elif topology_cache is not None:
            counts, sf_clean, caudit, baseline = topology_cache.get(a, region)
        else:
            counts, sf_clean, caudit, baseline = topology_counts(a, resolved_by_identity, region, prepared)
        if region == 'SR':
            shared_topology = counts, sf_clean, caudit, baseline
        b = blocks[region]
        core = highdm_core(a, blocks, region, dy_mass_window)
        high = core & (counts.sum(axis=1) >= 1)
        low = b.core & (b.nb >= 1) & (counts.sum(axis=1) == 0)
        old_high = core & (baseline.sum(axis=1) >= 1)
        old_low = b.core & (b.nb >= 1) & (baseline.sum(axis=1) == 0)
        # mTb>=175 is the adopted high-dM SR *search-bin* cut, not a CR cut.
        search = high & ((np.isfinite(b.mtb) & (b.mtb >= 175.)) if region == 'SR' else True)
        old_search = old_high & ((np.isfinite(b.mtb) & (b.mtb >= 175.)) if region == 'SR' else True)
        if np.any(high & low):
            raise ValueError("High/low overlap: " + region)
        output[region] = dict(high=high, high_search=search, low=low, counts=counts, block=b, sf_clean=sf_clean,
                              baseline_counts=baseline, baseline_high=old_high, baseline_search=old_search,
                              baseline_low=old_low)
        audit[region] = dict(high=int(high.sum()), high_search=int(search.sum()), low=int(low.sum()), intersection=0,
                            baseline_high=int(old_high.sum()), baseline_low=int(old_low.sum()),
                            high_gained=int((high & ~old_high).sum()), high_lost=int((old_high & ~high).sum()),
                            low_gained=int((low & ~old_low).sum()), low_lost=int((old_low & ~low).sum()), **caudit)
    return output, audit
