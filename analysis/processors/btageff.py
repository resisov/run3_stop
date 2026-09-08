import os
import numpy
import json
from coffea import processor, util
import hist
from coffea.util import save, load
from optparse import OptionParser
import numpy as np
import awkward as ak
from coffea.lookup_tools.dense_lookup import dense_lookup
import gzip
import correctionlib
from numba import njit


@njit
def decay_flavor(jet_counts, jet_eta, jet_phi, gen_counts, pdg, mother, eta, phi):
    """Classify AK8 containment of direct hadronic W/top decay partons."""
    result = np.zeros(len(jet_eta), dtype=np.int64)
    jo, go = 0, 0
    for event in range(len(jet_counts)):
        ng = gen_counts[event]
        for iw in range(ng):
            if abs(pdg[go + iw]) != 24:
                continue
            quarks = []
            for iq in range(ng):
                if mother[go + iq] == iw and 1 <= abs(pdg[go + iq]) <= 5:
                    quarks.append(iq)
            if len(quarks) != 2:
                continue
            parent = mother[go + iw]
            for _ in range(ng):
                if parent < 0 or abs(pdg[go + parent]) != 24:
                    break
                parent = mother[go + parent]
            top = parent if parent >= 0 and abs(pdg[go + parent]) == 6 else -1
            # Identify the same top across generator copy chains.
            if top >= 0:
                for _ in range(ng):
                    p = mother[go + top]
                    if p < 0 or pdg[go + p] != pdg[go + top]:
                        break
                    top = p
            bs = []
            if top >= 0:
                for ib in range(ng):
                    if abs(pdg[go + ib]) != 5:
                        continue
                    p = mother[go + ib]
                    if p < 0 or abs(pdg[go + p]) != 6:
                        continue
                    for _ in range(ng):
                        pp = mother[go + p]
                        if pp < 0 or pdg[go + pp] != pdg[go + p]:
                            break
                        p = pp
                    if p == top:
                        bs.append(ib)
            for ij in range(jet_counts[event]):
                j = jo + ij
                nq, nb = 0, 0
                for iq in quarks:
                    dp = (jet_phi[j] - phi[go + iq] + np.pi) % (2 * np.pi) - np.pi
                    nq += (jet_eta[j] - eta[go + iq]) ** 2 + dp ** 2 < 0.64
                for ib in bs:
                    dp = (jet_phi[j] - phi[go + ib] + np.pi) % (2 * np.pi) - np.pi
                    nb += (jet_eta[j] - eta[go + ib]) ** 2 + dp ** 2 < 0.64
                # Priority: merged top > merged W > partial top > partial W > other.
                category = 4 if nq == 2 and nb else 3 if nq == 2 else 2 if top >= 0 and (nq or nb) else 1 if nq else 0
                result[j] = max(result[j], category)
        jo += jet_counts[event]
        go += ng
    return result


class TopWTagEfficiency(processor.ProcessorABC):
    jet_id_fields = ('chHEF', 'neHEF', 'chEmEF', 'neEmEF', 'muEF', 'chMultiplicity', 'neMultiplicity')
    branches = tuple(
        ['run', 'Rho_fixedGridRhoFastjetAll', 'genWeight']
        + ['FatJet_' + f for f in ('pt', 'eta', 'phi', 'mass', 'area', 'msoftdrop',
           'globalParT3_withMassTopvsQCD', 'globalParT3_withMassWvsQCD')]
        + ['FatJet_' + f for f in jet_id_fields]
        + ['GenPart_' + f for f in ('pdgId', 'genPartIdxMother', 'eta', 'phi')]
    )

    def __init__(self, year, fjec, jet_id_json):
        self._year = year
        self._fjec = fjec
        self._jet_id = correctionlib.CorrectionSet.from_string(jet_id_json)

    def make_output(self):
        def histogram():
            return hist.Hist(
                hist.axis.StrCategory(['analysis', 'score_only'], name='selection'),
                hist.axis.StrCategory(['pass', 'fail'], name='tag'),
                hist.axis.IntCategory([0, 1, 2, 3, 4], name='flavor'),
                hist.axis.Variable([200, 300, 400, 480, 500, 600, 800, 1000, 1200, 1500, 2000, 3000, np.inf], name='pt'),
                hist.axis.Variable([0, 0.8, 1.5, 2.0], name='abseta'),
                storage=hist.storage.Weight(),
            )
        return {name + suffix: histogram() for name in ('GlobalParT3_Top', 'GlobalParT3_W') for suffix in ('', '_genweight')}

    def process(self, arrays):
        pt, eta, phi = (arrays['FatJet_' + f] for f in ('pt', 'eta', 'phi'))
        counts = ak.num(pt)
        output = self.make_output()
        if int(ak.sum(counts)) == 0:
            return output
        corr = self._fjec(self._year, pt, eta, phi, arrays['Rho_fixedGridRhoFastjetAll'],
                          arrays['FatJet_area'], arrays['run'], False)
        pt = pt * corr
        args = [ak.flatten(eta)] + [ak.flatten(arrays['FatJet_' + f]) for f in self.jet_id_fields]
        args.append(args[-1] + args[-2])
        jet_id = ak.unflatten(self._jet_id['AK8PUPPI_TightLeptonVeto'].evaluate(*args), counts) == 1
        flat = lambda x: np.asarray(ak.to_numpy(ak.flatten(x)))
        gen_counts = ak.num(arrays['GenPart_pdgId'])
        mother = arrays['GenPart_genPartIdxMother']
        if bool(ak.any((mother < -1) | (mother >= gen_counts))):
            raise ValueError('Invalid GenPart mother index')
        flavor = ak.unflatten(decay_flavor(
            np.asarray(counts), flat(eta), flat(phi), np.asarray(gen_counts),
            *(flat(arrays['GenPart_' + f]) for f in ('pdgId', 'genPartIdxMother', 'eta', 'phi')),
        ), counts)
        mass = arrays['FatJet_msoftdrop']
        weight = ak.broadcast_arrays(arrays['genWeight'], pt)[0]
        for value in (pt, eta, phi, mass, weight):
            if not bool(ak.all(np.isfinite(value))):
                raise ValueError('Non-finite jet kinematics or generator weights')
        base = jet_id & (pt > 200) & (abs(eta) < 2.0)
        for tagger, score_name, wp, analysis in (
            ('GlobalParT3_Top', 'Top', 0.5078, (pt > 400) & (mass > 105)),
            ('GlobalParT3_W', 'W', 0.9385, (mass > 60) & (mass < 105)),
        ):
            score = arrays['FatJet_globalParT3_withMass' + score_name + 'vsQCD']
            if not bool(ak.all(np.isfinite(score))):
                raise ValueError('Non-finite ' + score_name + ' tag score')
            for selection, eligible in (('analysis', base & analysis),
                                         ('score_only', base & (mass > 50) & (mass < 220))):
                for decision, tagged in (('pass', score > wp), ('fail', score <= wp)):
                    mask = eligible & tagged
                    values = dict(selection=selection, tag=decision, flavor=flat(flavor[mask]),
                                  pt=flat(pt[mask]), abseta=flat(abs(eta[mask])))
                    output[tagger].fill(**values)
                    output[tagger + '_genweight'].fill(**values, weight=flat(weight[mask]))
        return output

    def postprocess(self, output):
        return output

class BTagEfficiency(processor.ProcessorABC):

    def __init__(self, year, wp, ids):
        self._year = year
        self._btagWPs = wp
        self._ids = ids
        self.make_output = lambda: {
            'UParTAK4' :
            hist.Hist(
                hist.axis.StrCategory([], growth=True, name="wp", label="Working point"),
                hist.axis.StrCategory([], growth=True, name="btag", label="BTag WP pass/fail"),
                hist.axis.IntCategory([0, 4, 5, 6], name="flavor", label="Jet hadronFlavour"),
                hist.axis.Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name="pt", label=r'Jet $p_{T}$ [GeV]'),
                hist.axis.Variable([0, 1.4, 2.0, 2.5], name="abseta", label=r'Jet |$\eta$|'),
            ),
        }

    def process(self, events):
        
        dataset = events.metadata['dataset']
        isGoodJet = self._ids['isGoodJet']

        j = events.Jet
        j['isgood'] = isGoodJet(j, self._year)
        j_good = j[j.isgood]

        name = {}
        name['UParTAK4']= 'btagUParTAK4B'

        output = self.make_output()
        for wp in ['loose','medium','tight']:
            for tagger in ['UParTAK4']:
                passbtag = j_good[name[tagger]] > self._btagWPs[tagger][self._year][wp]
                output[tagger].fill(
                    wp=wp,
                    btag='pass',
                    flavor=ak.flatten(j_good[passbtag].hadronFlavour),
                    pt=ak.flatten(j_good[passbtag].pt),
                    abseta=ak.flatten(abs(j_good[passbtag].eta)),
                )
                output[tagger].fill(
                    wp=wp,
                    btag='fail',
                    flavor=ak.flatten(j_good[~passbtag].hadronFlavour),
                    pt=ak.flatten(j_good[~passbtag].pt),
                    abseta=ak.flatten(abs(j_good[~passbtag].eta)),
                )
        return output

    def postprocess(self, a):
        return a

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    parser.add_option('-m', '--metadata', help='metadata', dest='metadata')
    parser.add_option('-n', '--name', help='name', dest='name')
    parser.add_option('--tagger', type='choice', choices=['b', 'topw'], default='b')
    (options, args) = parser.parse_args()


    if options.tagger == 'topw':
        if options.year not in ('2024', '2025'):
            parser.error('topw supports 2024 and 2025')
        with gzip.open('data/JMESF/2024/jetid.json.gz', 'rt') as stream:
            jet_id_json = stream.read()
        processor_instance = TopWTagEfficiency(options.year, load('data/corrections.coffea')['get_fjec_correction'], jet_id_json)
        prefix = 'topwtageff'
    else:
        common = load('data/common.coffea')
        ids = load('data/ids.coffea')
        processor_instance = BTagEfficiency(year=options.year, wp=common['btagWPs'], ids=ids)
        prefix = 'btageff'
    save(processor_instance, 'data/' + prefix + (options.name or options.year) + '.processor')
