"""Canonical final SR/CR masks, shared by full and region-only histogramming.

The event_selections module supplies the common kinematics and topology
primitives and re-exports these entry points. Resolve those primitives at call
time so importing either module first is supported.
"""
import numpy as np
from .paths import analysis_era


def select(a, resolved_by_identity, dy_mass_window=(71., 111.), topology_cache=None, year=2024):
    from .event_selections import REGIONS, build_region_blocks, highdm_core, topology_buffers, topology_counts

    blocks, audit = build_region_blocks(a, dy_mass_window=dy_mass_window, year=year)
    if topology_cache is None:
        prepared = topology_buffers(a)
    else:
        if topology_cache.year != analysis_era(year):
            raise ValueError('Topology cache belongs to another analysis era')
        topology_cache.validate(a, resolved_by_identity)
    output = {}
    for region in REGIONS:
        if region in ('LLCR', 'QCDCR'):
            counts, sf_clean, caudit, baseline = shared_topology
        elif topology_cache is not None:
            counts, sf_clean, caudit, baseline = topology_cache.get(a, region)
        else:
            counts, sf_clean, caudit, baseline = topology_counts(a, resolved_by_identity, region, prepared, year)
        if region == 'SR':
            shared_topology = counts, sf_clean, caudit, baseline
        b = blocks[region]
        core = highdm_core(a, blocks, region, dy_mass_window, year)
        high = core & (counts.sum(axis=1) >= 1)
        low = b.core & (b.nb >= 1) & (counts.sum(axis=1) == 0)
        old_high = core & (baseline.sum(axis=1) >= 1)
        old_low = b.core & (b.nb >= 1) & (baseline.sum(axis=1) == 0)
        # Preserve high-dM SR preselection for overlap and cutflow audits.
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


def highdm_llcr_inference_support(view, year=2024):
    """Necessary LLCR mask from this endpoint, before testing any topology."""
    from .event_selections import highdm_core
    from .region_kinematics import update_veto_leptons

    block = view[:]  # Awkward field assignments must not mutate the caller's view.
    update_veto_leptons(block, year=year)
    return highdm_core(block, {}, "LLCR", year=year)


def select_highdm_llcr(a, resolved_by_identity, topology_cache=None, year=2024):
    """Project the approved LLCR branch without evaluating any other region."""
    from .event_selections import _highdm_llcr_jet_kinematics, highdm_core, topology_counts
    from .region_kinematics import RegionBlock, fatjet_kinematics, update_veto_leptons

    audit = update_veto_leptons(a, year=year)
    recoil, recoil_phi, jets = _highdm_llcr_jet_kinematics(a, year)
    empty = a["photon_pt_all"][:, :0]
    nisr, nt, nw = fatjet_kinematics(a, empty, empty)
    core = highdm_core(a, {}, "LLCR", year=year)
    b = RegionBlock(core=core, recoil=recoil, recoil_phi=recoil_phi, ht=jets["ht"], njet=jets["njet"],
        nb=jets["nb"], nt=nt, nw=nw, nisr=nisr, met_sqrt_ht=jets["met_sqrt_ht"], mtb=jets["mtb"], ptb=jets["ptb"])
    if topology_cache is None:
        counts, sf_clean, caudit, baseline = topology_counts(a, resolved_by_identity, "LLCR", year=year)
    else:
        if topology_cache.year != analysis_era(year):
            raise ValueError("Topology cache belongs to another analysis era")
        topology_cache.validate(a, resolved_by_identity)
        counts, sf_clean, caudit, baseline = topology_cache.get(a, "LLCR")
    high, old_high = core & (counts.sum(axis=1) >= 1), core & (baseline.sum(axis=1) >= 1)
    low = np.zeros(len(a), dtype=bool)
    audit.update(events=len(a), LLCR=dict(high=int(high.sum()), high_search=int(high.sum()),
        low=0, intersection=0, baseline_high=int(old_high.sum()), baseline_low=0,
        high_gained=int((high & ~old_high).sum()), high_lost=int((old_high & ~high).sum()),
        low_gained=0, low_lost=0, **caudit))
    return dict(LLCR=dict(high=high, high_search=high, low=low, counts=counts, block=b,
        sf_clean=sf_clean, baseline_counts=baseline, baseline_high=old_high,
        baseline_search=old_high, baseline_low=low)), audit
