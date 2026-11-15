"""Apply the current target-era object corrections before the retained skim."""
from functools import lru_cache
import awkward as ak
import numpy as np
import correctionlib
from .paths import ROOT, analysis_era, EARLY_ERAS
from .jme_calibration import calibrated_endpoints
from .corrections import nonjme_calibrate


@lru_cache(maxsize=24)
def correction_set(relative):
    return correctionlib.CorrectionSet.from_file(str(ROOT / relative))


def jet_id(raw, prefix):
    """NanoAODv15 ID, including v15 used for early target eras."""
    corr = correction_set('scales/JME/2024/jetid.json.gz')
    radius = 'AK4' if prefix == 'Jet' else 'AK8'
    names = ('eta', 'chHEF', 'neHEF', 'chEmEF', 'neEmEF', 'muEF',
             'chMultiplicity', 'neMultiplicity')
    values = [raw[prefix + '_' + name] for name in names]
    values.append(values[-2] + values[-1])
    result = corr[radius + 'PUPPI_TightLeptonVeto'].evaluate(
        *(np.asarray(ak.flatten(value)) for value in values))
    return ak.unflatten(result, ak.num(values[0])) == 1


def jet_veto(pt, eta, phi, era):
    era = analysis_era(era)
    names = {'2022': 'Summer22_23Sep2023_RunCD_V1',
             '2022EE': 'Summer22EE_23Sep2023_RunEFG_V1',
             '2023': 'Summer23Prompt23_RunC_V1',
             '2023BPix': 'Summer23BPixPrompt23_RunD_V1',
             '2024': 'Summer24Prompt24_RunBCDEFGHI_V1',
             '2025': 'Summer24Prompt25_RunCDEFG_V1'}
    corr = correction_set(f'scales/JME/{era}/jetvetomaps.json.gz')[names[era]]
    bad = corr.evaluate('jetvetomap', np.asarray(ak.flatten(eta)),
                        np.asarray(ak.flatten(phi))) != 0
    return (pt > 30) & ak.unflatten(bad, ak.num(eta))


def jme_inputs(raw):
    out = {key: raw[key] for key in ('run', 'luminosityBlock', 'event')}
    out['is_data'] = np.zeros(len(raw), dtype=bool)
    out['rho'] = raw['Rho_fixedGridRhoFastjetAll']
    mapping = {'pt': 'nanoaod_pt', 'mass': 'nanoaod_mass', 'eta': 'eta_all',
               'phi': 'phi_all', 'rawFactor': 'raw_factor', 'area': 'area'}
    for prefix, lower in (('Jet', 'jet'), ('FatJet', 'fatjet')):
        for name, target in mapping.items():
            out[lower + '_' + target] = raw[prefix + '_' + name]
    out['jet_genjet_index_all'] = raw['Jet_genJetIdx']
    out['fatjet_genjetak8_index_all'] = raw['FatJet_genJetAK8Idx']
    for prefix, lower in (('GenJet', 'genjet'), ('GenJetAK8', 'genjetak8')):
        for name in ('pt', 'eta', 'phi'):
            out[lower + '_' + name + '_all'] = raw[prefix + '_' + name]
    return ak.Array(out)


def calibrate(raw, era):
    """Reuse canonical JEC/JER, non-JME central values and type-1 propagation."""
    era = analysis_era(era)
    momenta = calibrated_endpoints(jme_inputs(raw), era, nominal_only=True)['nominal']
    updates = {'Jet_pt': momenta['jet_corrected_pt'],
               'Jet_mass': momenta['jet_corrected_mass'],
               'FatJet_pt': momenta['fatjet_corrected_pt'],
               'FatJet_mass': momenta['fatjet_corrected_mass']}
    audit = {}
    for flavor in ('electron', 'muon', 'photon', 'tau'):
        pt, mass, audit[flavor] = nonjme_calibrate(raw, era, flavor, 'nominal')
        prefix = flavor.capitalize()
        updates[prefix + '_pt'] = pt
        if prefix != 'Photon':
            updates[prefix + '_mass'] = mass
    x = np.asarray(raw['PuppiMET_pt']) * np.cos(raw['PuppiMET_phi'])
    y = np.asarray(raw['PuppiMET_pt']) * np.sin(raw['PuppiMET_phi'])
    type1 = ((updates['Jet_pt'] > 15) & (abs(raw['Jet_eta']) < 5.2)
             & ((raw['Jet_chEmEF'] + raw['Jet_neEmEF']) < .9))
    delta = (raw['Jet_pt'] - updates['Jet_pt']) * (1 - raw['Jet_muonSubtrFactor'])
    for values, phi in [(ak.where(type1, delta, 0), raw['Jet_phi'])] + [
            (raw[p + '_pt'] - updates[p + '_pt'], raw[p + '_phi'])
            for p in ('Electron', 'Muon', 'Photon', 'Tau')]:
        x = x + np.asarray(ak.sum(values * np.cos(phi), axis=1))
        y = y + np.asarray(ak.sum(values * np.sin(phi), axis=1))
    pt, phi = np.hypot(x, y), np.arctan2(y, x)
    if era in EARLY_ERAS:
        epoch = {'2022': '2022', '2022EE': '2022EE',
                 '2023': '2023', '2023BPix': '2023BPix'}[era]
        cset = correction_set(f'scales/JME/{era}/met_xyCorrections_{era[:4]}_{epoch}.json.gz')
        args = ('PuppiMET', epoch, 'MC', 'nom', pt, phi, np.asarray(raw['PV_npvsGood']))
        corr = cset['met_xy_corrections']
        updates['AllHadMETXY_pre_pt'], updates['AllHadMETXY_pre_phi'] = pt, phi
        pt, phi = corr.evaluate('pt', *args), corr.evaluate('phi', *args)
        updates['AllHadMETXY_delta_x'] = pt * np.cos(phi) - x
        updates['AllHadMETXY_delta_y'] = pt * np.sin(phi) - y
        updates['AllHadMETXY_applied'] = np.ones(len(raw), dtype=bool)
    updates.update(PuppiMET_pt=pt, PuppiMET_phi=phi)
    corrected = raw
    for name, value in updates.items():
        corrected = ak.with_field(corrected, value, name)
    return corrected, audit
