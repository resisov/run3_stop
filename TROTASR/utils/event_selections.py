"""All analysis event selections, independent of histogram and plotting code.

Constituent arbitration prioritizes boosted, mixed and resolved top candidates,
then selects independent W candidates. Topology cuts include the Mixed high-dM
union and the low-dM veto.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import awkward as ak
from .analysis_regions import highdm_llcr_inference_support, select, select_highdm_llcr
from .region_kinematics import build_region_blocks, dycr_lepton_mask, object_masks, clean_by_delta_r
from .ids import delta_phi
from .constituents import boosted_overlap_vetoed_ak4_indices
from .paths import EARLY_ERAS, tagging_definition, analysis_era

REGIONS = ("SR", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M")
HIGHDM_LLCR_JME_EARLY_COLUMNS = (
    "is_data", "pass_base_common", "pass_signal_trigger",
    "electron_pt_all", "electron_eta_all", "electron_cutbased_all", "electron_mini_iso_all",
    "muon_pt_all", "muon_eta_all", "muon_mini_iso_all", "muon_loose_id_all", "muon_medium_id_all",
    "photon_pt_all", "photon_eta_all", "photon_cutbased_all", "photon_electron_veto_all",
)
HIGHDM_LLCR_NONJME_INPUT_COLUMNS = (
    "is_data", "pass_base_common", "pass_signal_trigger", "electron_pt_all", "muon_pt_all",
)
MIXED_WP = 0.9027690887451172
CATEGORY_ORDER = ("Nb", "Nbst", "Nmix", "Nres", "Nw")
MIXED_FIELDS = ("candidateIndex", "sourceFatJetIdx", "sourceJetIdx0", "sourceJetIdx1",
                "sourceJetIdx2", "QCDDiscriminant")
OVERLAP_FIELDS = ("jet_source_index_all", "jet_eta_all", "jet_phi_all", "fatjet_source_index_all",
                  "fatjet_eta_all", "fatjet_phi_all", "fatjet_subjet_index1_all", "fatjet_subjet_index2_all",
                  "fatjet_boosted_top_pass_all", "fatjet_boosted_w_pass_all", "subjet_eta_all", "subjet_phi_all")


def arbitrate(boosted, mixed, resolved, w_candidates, footprints):
    """Deterministic disjoint constituents, independent W retained.

    AK8 footprints use the two-subjet dR<0.4 association where available,
    otherwise the dR<0.8 association around the AK8 axis.
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
        # W multiplicity counts independent AK8s; overlap removal applies to
        # top/W pairs, not W/W pairs.
    return dict(B=chosen_b, M=chosen_m, R=chosen_r, W=chosen_w, rejected=rejected)


def _references(a, region, year=2024):
    if region in ("SR", "LLCR", "QCDCR"):
        empty = a["jet_eta_all"][:, :0]
        return empty, empty
    flavor = {"GCR": "photon", "DY2E": "electron", "DY2M": "muon"}[region]
    selected = object_masks(a, year)[flavor + "_medium"]
    return a[flavor + "_eta_all"][selected], a[flavor + "_phi_all"][selected]


@dataclass(frozen=True)
class _JaggedRows:
    offsets: np.ndarray
    values: np.ndarray

    def __getitem__(self, i):
        return self.values[self.offsets[i]:self.offsets[i + 1]]


def _jagged_rows(a, offset_pool=()):
    counts = ak.to_numpy(ak.num(a, axis=1), allow_missing=False)
    offsets = np.empty(len(counts) + 1, dtype=np.intp)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    offsets = next((other for other in offset_pool if np.array_equal(other, offsets)), offsets)
    return _JaggedRows(offsets, ak.to_numpy(ak.flatten(a, axis=1), allow_missing=False))


def topology_buffers(a):
    """Keep jagged columns as flat buffers shared across consumers of this view."""
    fields = set(ak.fields(a))
    needed = set(OVERLAP_FIELDS) | {"TopMixed1pct_" + n for n in MIXED_FIELDS} | {"nTopMixed1pct"}
    if needed - fields:
        raise ValueError("Missing topology input: " + ",".join(sorted(needed - fields)))
    columns, offset_pool = {}, []
    for k in (*OVERLAP_FIELDS, *("TopMixed1pct_" + x for x in MIXED_FIELDS)):
        rows = columns[k] = _jagged_rows(a[k], offset_pool)
        if not any(rows.offsets is other for other in offset_pool):
            offset_pool.append(rows.offsets)
    scalar = {k: np.asarray(a[k]) for k in ('nTopMixed1pct', 'file_id', 'entry')}
    for name in ("fatjet_source_index_all", "jet_source_index_all"):
        identities = ak.sort(a[name], axis=1)
        if ak.any(identities[:, 1:] == identities[:, :-1]):
            raise ValueError("Duplicate original object identity")
    # Historically the footprint builder checked these lengths whenever the
    # event had an AK8 jet, even if none of its candidates survived cleaning.
    has_fatjet = np.diff(columns["fatjet_source_index_all"].offsets) > 0
    for names, message in (
        (("jet_source_index_all", "jet_eta_all", "jet_phi_all"),
         "AK4 identity and kinematic arrays have different lengths"),
        (("fatjet_source_index_all", "fatjet_eta_all", "fatjet_phi_all",
          "fatjet_subjet_index1_all", "fatjet_subjet_index2_all"),
         "FatJet identity, selection, and kinematic arrays differ in length"),
        (("subjet_eta_all", "subjet_phi_all"),
         "SubJet eta and phi arrays have different lengths"),
    ):
        ref = columns[names[0]].offsets
        if any(columns[k].offsets is not ref and
               np.any((np.diff(columns[k].offsets) != np.diff(ref)) & has_fatjet)
               for k in names[1:]):
            raise ValueError(message)
    return columns, scalar


class _EventFootprints:
    """Build the canonical footprint only when this event needs that fatjet."""

    def __init__(self, event):
        self.event = event
        self.positions = {source: i for i, source in enumerate(event["fatjet_source_index_all"])}
        self.associations = {}

    def __contains__(self, source):
        return source in self.positions

    def __getitem__(self, source):
        if source not in self.associations:
            i = self.positions[source]
            sl = slice(i, i + 1)
            event = self.event
            self.associations[source] = boosted_overlap_vetoed_ak4_indices(
                jet_source_indices=event["jet_source_index_all"], jet_eta=event["jet_eta_all"],
                jet_phi=event["jet_phi_all"], fatjet_eta=event["fatjet_eta_all"][sl],
                fatjet_phi=event["fatjet_phi_all"][sl],
                fatjet_subjet_index1=event["fatjet_subjet_index1_all"][sl],
                fatjet_subjet_index2=event["fatjet_subjet_index2_all"][sl],
                fatjet_top_pass=(True,), fatjet_w_pass=(False,),
                subjet_eta=event["subjet_eta_all"], subjet_phi=event["subjet_phi_all"],
            )
        return self.associations[source]


def topology_counts(a, resolved_by_identity, region, prepared=None, year=2024):
    mixed_wp = tagging_definition(year)['mixed_wp']
    columns_all, scalar = topology_buffers(a) if prepared is None else prepared
    eta, phi = _references(a, region, year)
    fat_keep_all = _jagged_rows(clean_by_delta_r(a["fatjet_eta_all"], a["fatjet_phi_all"], eta, phi, .4))
    jet_keep_all = _jagged_rows(clean_by_delta_r(a["jet_eta_all"], a["jet_phi_all"], eta, phi, .2))
    n = len(a)
    counts = np.zeros((n, 4), dtype=np.int32)
    baseline = np.zeros((n, 4), dtype=np.int32)
    fat_offsets = columns_all["fatjet_source_index_all"].offsets
    sf_clean = np.empty(fat_offsets[-1], dtype=bool)
    audits = {"mixed_overlap": 0, "resolved_overlap": 0, "w_overlap": 0, "mixed_and_independent_w": 0}
    aligned = (hasattr(resolved_by_identity, 'matches_rows') and
               resolved_by_identity.matches_rows(scalar['file_id'], scalar['entry']))
    for i in range(n):
        event = {k: columns_all[k][i] for k in OVERLAP_FIELDS}
        fj = event["fatjet_source_index_all"]
        jj = event["jet_source_index_all"]
        fat_keep, jet_keep = fat_keep_all[i], jet_keep_all[i]
        good_fat = {j for j, good in zip(fj, fat_keep) if good}
        good_jet = {j for j, good in zip(jj, jet_keep) if good}
        footprints = _EventFootprints(event)
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
            score = float(columns["QCDDiscriminant"][k])
            if fat not in footprints or not set(jets) <= set(jj) or len(jets) != len(set(jets)):
                raise ValueError("Invalid Mixed original constituent index")
            if not np.isfinite(score) or score < mixed_wp:
                raise ValueError("Invalid stored Mixed discriminant")
            if fat in good_fat and set(jets) <= good_jet:
                mixed.append(dict(index=index, fat=fat, jets=jets, score=score))
        if aligned:
            candidates = resolved_by_identity.row(i)
        else:
            key = (int(scalar["file_id"][i]), int(scalar["entry"][i]))
            candidates = resolved_by_identity.get(key, ())
        resolved = []
        for c in candidates:
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
        sf_clean[fat_offsets[i]:fat_offsets[i + 1]] = [
            good and (f in boosted or (f not in used_m_fat and not (footprints[f] & used_mr_jets)))
            for f, good in zip(fj, fat_keep)
        ]
        for src, dst in (("mixed", "mixed_overlap"), ("resolved", "resolved_overlap"), ("w", "w_overlap")):
            audits[dst] += selection["rejected"][src]
        audits["mixed_and_independent_w"] += bool(selection["M"] and selection["W"])
    return counts, ak.unflatten(sf_clean, np.diff(fat_offsets)), audits, baseline


def trigger_highdm_vr_mask(chunk, region):
    """High-dM VR support of the trigger scale factor, before topology cuts."""
    get = lambda key: np.asarray(chunk[key])
    d = [get("j%d_met_dphi" % k) for k in range(1, 5)]
    nb = get("nb_medium")
    group = {"HighDMVR_Nb1": nb == 1, "HighDMVR_Nb2": nb == 2, "HighDMVR_Nb3plus": nb >= 3}[region]
    return (get("pass_base_common").astype(bool) & get("pass_signal_trigger").astype(bool)
            & get("pass_zero_tau").astype(bool) & get("pass_no_veto_leptons").astype(bool)
            & (get("njet") >= 5) & group & get("pass_met_250").astype(bool) & get("pass_ht_300").astype(bool)
            & (d[0] > .5) & (d[1] > .15) & (d[2] > .15) & ((d[1] < .5) | (d[2] < .5) | (d[3] < .5)))


def highdm_core(a, blocks, region, dy_mass_window=(71., 111.), year=2024):
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
        eta, phi = _references(a, region, year)
        good = ((a['jet_corrected_pt'] > 30.) & (abs(a['jet_eta_all']) < 2.4)
                & ak.values_astype(a['jet_id_all'], np.bool_)
                & clean_by_delta_r(a['jet_eta_all'], a['jet_phi_all'], eta, phi, .2))
        angles = delta_phi(a['jet_phi_all'][good][:, :4], get('recoil_gcr_phi')[:, None])
        opened = np.asarray(ak.all(angles > .5, axis=1))
        b = blocks[region]
        return b.core & (b.njet >= 4) & (b.nb >= 1) & opened
    return (dycr_lepton_mask(a, region, mass_window=dy_mass_window, year=year)
            & get("pass_" + region.lower() + "_open_high").astype(bool)
            & (get("njet_lepton_clean") >= 4) & (get("nb_lepton_clean") >= 1)
            & (get("ht_lepton_clean") > 300.) & (get("recoil_" + region.lower()) > 250.))


class FixedJetTopologyCache:
    """Chunk-local reuse only when every jet/candidate input is exactly fixed.

    Non-JME endpoints keep these columns unchanged. Region cleaning is reused
    only for identical selected reference directions; threshold crossings miss
    the cache and run the original arbitration. No event selections are cached.
    """
    def __init__(self, a, resolved, year=2024):
        self.year = analysis_era(year)
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
        eta, phi = _references(a, region, self.year)
        reference = (self.signature(eta), self.signature(phi))
        old = self.regions.get(region)
        if old is None or any(not np.array_equal(before, after)
                              for left, right in zip(old[0], reference)
                              for before, after in zip(left, right)):
            value = topology_counts(a, self.resolved, region, self.prepared, self.year)
            self.regions[region] = reference, value
        return self.regions[region][1]


def _highdm_llcr_jet_kinematics(a, year):
    from .region_kinematics import as_ak_bool, as_float, jet_kinematics

    pt, eta, phi = (a[k] for k in ("jet_corrected_pt", "jet_eta_all", "jet_phi_all"))
    tag = a[tagging_definition(year)["btag_branch"]]
    recoil, recoil_phi = as_float(a, "met"), as_float(a, "met_phi")
    good = (pt > 30.) & (abs(eta) < 2.4) & as_ak_bool(a["jet_id_all"])
    return recoil, recoil_phi, jet_kinematics(pt, eta, phi, tag, good, recoil, recoil_phi, year)


def _highdm_llcr_common_mc(a):
    if np.any(np.asarray(a["is_data"], dtype=bool)):
        raise ValueError("Early LLCR support is MC-only")
    return (np.asarray(a["pass_base_common"], dtype=bool)
            & np.asarray(a["pass_signal_trigger"], dtype=bool))


def highdm_llcr_jme_early_support(central, year=2024):
    """Return prerequisites unchanged by JES/JER, before building any view.

    Use canonical central lepton columns (after any era-dependent central
    muon repair), not a stored one-lepton flag. The adopted JME views preserve
    the base-quality and trigger flags and do not change electron/muon IDs.
    MET, tau veto, lepton mT and jet cuts still belong to each endpoint.
    This gate is only for MC LLCR and cannot be shared with lepton shapes.
    """
    common = _highdm_llcr_common_mc(central)
    masks = object_masks(central, year)
    one_lepton = np.asarray(ak.sum(masks["electron_veto"], axis=1)
                           + ak.sum(masks["muon_loose"], axis=1)) == 1
    return common & one_lepton


def highdm_llcr_input_support(source, year=2024, *, shape_mode):
    """Conservative LLCR MC prerequisites on a light, uncorrected read.

    The early eras' central lepton momenta are already the adopted ones.
    For 2024/25, a stored muon can cross the loose threshold in the central
    repair; for non-JME shapes, both lepton populations can migrate. Never
    cut on a central MET, tau veto, mT, angular or jet quantity here.
    """
    if shape_mode == "cms_trota_jme":
        if analysis_era(year) in EARLY_ERAS:
            return highdm_llcr_jme_early_support(source, year)
        common = _highdm_llcr_common_mc(source)
        e_veto = np.asarray(ak.sum(object_masks(source, year)["electron_veto"], axis=1))
        muons = ak.to_numpy(ak.num(source["muon_pt_all"], axis=1), allow_missing=False)
        return common & (e_veto <= 1) & ((e_veto == 1) | (muons > 0))
    if shape_mode == "stored_trota_nonjme":
        common = _highdm_llcr_common_mc(source)
        electrons = ak.to_numpy(ak.num(source["electron_pt_all"], axis=1), allow_missing=False)
        muons = ak.to_numpy(ak.num(source["muon_pt_all"], axis=1), allow_missing=False)
        return common & ((electrons > 0) | (muons > 0))
    raise ValueError("Early LLCR input support needs a declared systematic view")
