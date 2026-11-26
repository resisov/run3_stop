"""Full-source signal tag efficiencies with the adopted per-era algorithms.

Read each NanoAOD file once for all six target eras. Counts use stored NanoAOD
momenta, canonical jet ID, no event skim and no lepton cleaning. The producer
owns bounded execution, retries, collection and installation of these counts.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import contextlib
import os
import pickle

import awkward as ak
import correctionlib
import lz4.frame
import numpy as np
import uproot

from .paths import ROOT, ERAS, EARLY_ERAS, tagging_definition
from .io import load_efficiency
from .reader import project_candidate_entries

DIRECTORY = ROOT / 'production' / 'efficiencies'
DEFINITION = dict(
    population='all original NanoAODv15 events; no event skim or lepton cleaning',
    momenta='stored NanoAOD pT and msoftdrop; no extra JEC/JER',
    jet_id='canonical NanoAODv15 TightLeptonVeto',
    btag='AK4 pT>30 GeV, abs(eta)<2.4; era loose/medium/tight score working points',
    top='AK8 pT>400 GeV, abs(eta)<2, msoftdrop>105 GeV',
    w='AK8 pT>200 GeV, abs(eta)<2, 60<msoftdrop<105 GeV',
    score_only='AK8 pT>200 GeV, abs(eta)<2; no mass cut',
    counts='unit counts; additional signed genWeight Top/W counts and sumw2',
    topw_truth='canonical direct hadronic W/top containment classifier',
    late_top_score='FatJet_globalParT3_withMassTopvsQCD',
    late_w_score='FatJet_globalParT3_withMassWvsQCD',
)


def save_counts(path, data):
    """Atomically write trusted analysis-owned Coffea-compatible histograms."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with lz4.frame.open(temporary, 'wb') as stream:
        pickle.dump(data, stream, protocol=4)
    os.replace(temporary, path)


def read_counts(path):
    from .io import decode_efficiency
    return decode_efficiency(Path(path).read_bytes())


@lru_cache(maxsize=1)
def prototypes():
    result = {}
    for era in ERAS:
        result[era] = {}
        for kind in ('btag', 'topwtag'):
            current = load_efficiency(ROOT / 'estimations' / f'{kind}eff{era}.merged')
            result[era][kind] = {}
            for tagger, datasets in current.items():
                prototype = next(iter(datasets.values())).copy()
                prototype.reset()
                result[era][kind][tagger] = prototype
    return result


@lru_cache(maxsize=6)
def working_points(era):
    definition = tagging_definition(era)
    corrections = correctionlib.CorrectionSet.from_file(
        str(ROOT / 'scales' / 'BTV' / era / 'btagging.json.gz'))
    values = corrections[definition['btag'] + '_wp_values']
    return dict(loose=definition['loose'], medium=definition['medium'],
                tight=float(values.evaluate('T')))


_JET_ID_FIELDS = ('eta', 'chHEF', 'neHEF', 'chEmEF', 'neEmEF', 'muEF',
                  'chMultiplicity', 'neMultiplicity')


def columns():
    names = ['genWeight', 'Jet_pt', 'Jet_btagPNetB', 'Jet_btagUParTAK4B',
             'Jet_hadronFlavour', 'FatJet_pt', 'FatJet_phi', 'FatJet_msoftdrop',
             'FatJet_particleNetWithMass_TvsQCD', 'FatJet_particleNetWithMass_WvsQCD',
             'FatJet_globalParT3_withMassTopvsQCD', 'FatJet_globalParT3_withMassWvsQCD',
             'GenPart_pdgId', 'GenPart_genPartIdxMother', 'GenPart_eta', 'GenPart_phi']
    for prefix in ('Jet', 'FatJet'):
        names.extend(prefix + '_' + field for field in _JET_ID_FIELDS)
    return list(dict.fromkeys(names))


def _eligible_jet_id(raw, prefix, eligible):
    """Run the canonical ID only on geometrically eligible jets; retain source alignment."""
    from .nanoaod_calibration import jet_id
    take = np.asarray(ak.flatten(eligible), dtype=bool)
    if not take.any():
        return eligible
    if take.all():
        return jet_id(raw, prefix)
    fields = (prefix + '_' + name for name in _JET_ID_FIELDS)
    inputs = {field: raw[field][eligible] for field in fields}
    accepted = np.zeros_like(take)
    accepted[take] = np.asarray(ak.flatten(jet_id(inputs, prefix)), dtype=bool)
    return ak.unflatten(accepted, ak.num(eligible))


def _truth_for_eligible(tree, start, raw, ak8, njet):
    from .nanoaod_events import decay_flavor, TOPW_GEN_FIELDS
    take = np.flatnonzero(njet)
    if not len(take):
        return np.empty(0, dtype=np.int64)
    gen = project_candidate_entries(tree, ['GenPart_' + field for field in TOPW_GEN_FIELDS], start + take)
    gen_counts = ak.num(gen['GenPart_pdgId'])
    mother = gen['GenPart_genPartIdxMother']
    if bool(ak.any((mother < -1) | (mother >= gen_counts))):
        raise ValueError('Invalid GenPart mother index')
    flat = lambda x: np.asarray(ak.to_numpy(ak.flatten(x)))
    return decay_flavor(njet[take], flat(raw['FatJet_eta'][ak8][take]), flat(raw['FatJet_phi'][ak8][take]),
                        np.asarray(gen_counts, dtype=np.int64),
                        *(flat(gen['GenPart_' + field]) for field in TOPW_GEN_FIELDS))


def measure(source):
    from .nanoaod_production import cache_source

    name = Path(source['name']).stem
    output = DIRECTORY / 'files' / (name + '.pkl.lz4')
    if output.exists():
        result = read_counts(output)
        if result['source'] != source['name'] or result['events'] != source['nevents']:
            raise ValueError('Completed efficiency counts do not match the input file')
        return dict(file=name, events=result['events'], reused=True)
    cache = DIRECTORY / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache_source(source, cache)
    counts = {era: {kind: {tag: h.copy() for tag, h in tags.items()}
                    for kind, tags in kinds.items()} for era, kinds in prototypes().items()}
    flat = lambda x: np.asarray(ak.to_numpy(ak.flatten(x)))
    processed, gen_sumw, gen_sumw2 = 0, 0., 0.
    with uproot.open(cached) as root:
        tree = root['Events']
        if tree.num_entries != source['nevents']:
            raise ValueError('Full-source event count differs from the production input')
        for raw in tree.iterate([c for c in columns() if not c.startswith('GenPart_')],
                                step_size=10000, library='ak'):
            start = processed
            weights = np.asarray(raw['genWeight'], dtype=float)
            if not np.isfinite(weights).all():
                raise ValueError('Nonfinite generator weight')
            processed += len(raw)
            gen_sumw += float(weights.sum()); gen_sumw2 += float(np.square(weights).sum())
            ak4 = _eligible_jet_id(raw, 'Jet', (raw['Jet_pt'] > 30) & (abs(raw['Jet_eta']) < 2.4))
            bpt, beta = flat(raw['Jet_pt'][ak4]), flat(abs(raw['Jet_eta'][ak4]))
            flavor = flat(raw['Jet_hadronFlavour'][ak4]).astype(int)
            scores = {True: flat(raw['Jet_btagPNetB'][ak4]),
                      False: flat(raw['Jet_btagUParTAK4B'][ak4])}
            ak8 = _eligible_jet_id(raw, 'FatJet', (raw['FatJet_pt'] > 200) & (abs(raw['FatJet_eta']) < 2))
            pt, eta = flat(raw['FatJet_pt'][ak8]), flat(abs(raw['FatJet_eta'][ak8]))
            mass = flat(raw['FatJet_msoftdrop'][ak8])
            njet = np.asarray(ak.num(raw['FatJet_pt'][ak8]), dtype=np.int64)
            truth = _truth_for_eligible(tree, start, raw, ak8, njet)
            if len(truth) != len(pt) or np.any((truth < 0) | (truth > 4)):
                raise ValueError('Invalid aligned Top/W truth categories')
            jet_weights = np.repeat(weights, njet)
            for values in (bpt, beta, pt, eta, mass):
                if not np.isfinite(values).all(): raise ValueError('Nonfinite jet kinematics')
            topw_scores = {
                True: dict(top=flat(raw['FatJet_particleNetWithMass_TvsQCD'][ak8]),
                           w=flat(raw['FatJet_particleNetWithMass_WvsQCD'][ak8])),
                False: dict(top=flat(raw['FatJet_globalParT3_withMassTopvsQCD'][ak8]),
                            w=flat(raw['FatJet_globalParT3_withMassWvsQCD'][ak8])),
            }
            for era in ERAS:
                definition = tagging_definition(era); early = era in EARLY_ERAS
                btag = counts[era]['btag'][definition['btag']]
                for wp, threshold in working_points(era).items():
                    passed = (scores[early] > threshold) & (scores[early] <= 1.)
                    btag.fill(wp=wp, btag=np.where(passed, 'pass', 'fail'),
                              flavor=flavor, pt=bpt, abseta=beta)
                for tag in ('top', 'w'):
                    score = topw_scores[early][tag]
                    passed = (score > definition[tag]) & (score <= 1.)
                    eligible = ((pt > 400) & (mass > 105) if tag == 'top'
                                else (mass > 60) & (mass < 105))
                    for selection, mask in [('score_only', np.ones(len(pt), dtype=bool)),
                                             ('analysis', eligible)]:
                        for suffix, weight in [('', np.ones(len(pt))), ('_genweight', jet_weights)]:
                            hist = counts[era]['topwtag'][definition['topw'] + '_' + tag.capitalize() + suffix]
                            hist.fill(selection=selection, tag=np.where(passed[mask], 'pass', 'fail'),
                                      flavor=truth[mask], pt=pt[mask], abseta=eta[mask], weight=weight[mask])
            print('EFFICIENCY', name, processed, '/', source['nevents'], flush=True)
    if processed != source['nevents']:
        raise ValueError('Incomplete full-source traversal')
    save_counts(output, dict(source=source['name'], dataset=source['dataset'], events=processed,
                             sumw=gen_sumw, sumw2=gen_sumw2, counts=counts))
    cached.unlink()
    return dict(file=name, events=processed, reused=False)


def measure_logged(source):
    """Keep per-file transfer and counting output in the stage's own log."""
    logs = DIRECTORY / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / (Path(source['name']).stem + '.log')
    with path.open('a', buffering=1) as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        return measure(source)


def dataset_key(dataset, era, kind):
    if kind == 'btag' and era not in EARLY_ERAS:
        parts = dataset.strip('/').split('/')
        return parts[0] + '-' + parts[1]
    return dataset


def assemble(sources):
    """Keep existing entries intact and install exact new dataset counts."""
    products = {era: {} for era in ERAS}
    for era in ERAS:
        for kind in ('btag', 'topwtag'):
            products[era][kind] = load_efficiency(ROOT / 'estimations' / f'{kind}eff{era}.merged')
    new_keys = {era: {kind: set() for kind in ('btag', 'topwtag')} for era in ERAS}
    for source in sources:
        result = read_counts(DIRECTORY / 'files' / (Path(source['name']).stem + '.pkl.lz4'))
        if result['events'] != source['nevents'] or result['dataset'] != source['dataset']:
            raise ValueError('Efficiency file coverage mismatch')
        for era in ERAS:
            for kind in ('btag', 'topwtag'):
                key = dataset_key(source['dataset'], era, kind)
                for tag, hist in result['counts'][era][kind].items():
                    if key not in new_keys[era][kind]:
                        products[era][kind][tag][key] = hist.copy()
                    else:
                        products[era][kind][tag][key] += hist
                new_keys[era][kind].add(key)
    for era in ERAS:
        for kind in ('btag', 'topwtag'):
            filename = f'{kind}eff{era}.merged'
            save_counts(DIRECTORY / filename, products[era][kind])
            save_counts(ROOT / 'estimations' / filename, products[era][kind])
    return {era: {kind: len(keys) for kind, keys in kinds.items()} for era, kinds in new_keys.items()}
