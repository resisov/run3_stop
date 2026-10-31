"""Prediction/renderer contracts, not substitute physics fits or production plots."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json,sha256
from TROTASR.utils.statistics_inputs import build_inputs
from TROTASR.utils.statistical_templates import assemble,card_text,json_ready
from TROTASR.tests.test_statistics_inputs import fixture
from TROTASR.workflows.prediction_plot_inputs import (
    parse_card,channel_prediction,combine,flat_blocks,expected_labels,read_th1,
    uncertainty_band,renderer,LOW_SIGNALS,ll_display_projection)
from TROTASR.workflows.plotting_gnn_controls import prepare,ll_inclusive,POOLED,current_card_controls
from TROTASR.workflows.plotting_card_predictions import prepare as prepare_cards


class Histogram:
    classname='TH1D'
    def __init__(self,values,variance,labels):
        self.value,self.variance,self.label=np.asarray(values),np.asarray(variance),labels
    def values(self):return self.value
    def variances(self):return self.variance
    def axis(self):return self
    def labels(self):return self.label


class Directory(dict):
    def __init__(self,*args,sr=False,**kwargs):
        super().__init__(*args,**kwargs);self.sr=sr
    def __getitem__(self,key):
        if self.sr and key=='data_obs':
            raise AssertionError('SR data_obs must never be read')
        return super().__getitem__(key)


def model_records(year=2024):
    inputs=build_inputs(*fixture(year),'T2tt',1200,500)
    channels,values,_,issues=assemble([inputs])
    assert not issues
    card=parse_card(card_text(channels,values));output={}
    for mode,regions in (('highdm',('SR',)),('lowdm',('SR','LLCR','QCDCR','GCR'))):
        for region in regions:
            name=f'{region}_{mode}_c1_{year}'
            labels=[b['name'] for b in channels[name]]
            assert labels==expected_labels(name)
            directory=Directory(sr=region=='SR')
            for (c,process),record in values.items():
                if c!=name:continue
                directory[process]=Histogram(record['nominal'],record['sumw2'],labels)
                for nuisance,pair in record['variations'].items():
                    for direction,suffix in (('up','Up'),('down','Down')):
                        directory[process+'_'+nuisance+suffix]=Histogram(pair[direction],record['sumw2'],labels)
            if region!='SR':
                obs=[b['observation'] for b in channels[name]]
                directory['data_obs']=Histogram(obs,obs,labels)
            record=channel_prediction(card,name,directory,labels)
            if region=='SR':
                masses=[s['key'] for s in renderer.SIGNAL_OVERLAYS] if mode=='highdm' else LOW_SIGNALS
                record['signals']={k:np.ones(len(labels)) for k in masses}
            output[str(year)+'/'+region+'/'+mode]=record
    return output


def raw_fixture():
    payload=fixture()[0]
    payload['contract'].update(sr_binning_sha256=sha256(ROOT/'highdm_sr_binning.json'),
        configuration_sha256=sha256(ROOT/'gnn4lowdm/config.json'))
    source=payload['histograms']['lowdm']['nominal']
    for region in ('DY2E','DY2M'):
        source[region]=deepcopy(source['LLCR'])
        for samples in source[region].values():
            samples['DY']=deepcopy(samples['ST'])
    rz=dict(status='complete',scope=payload['scope'],provenance=dict(campaign_year='2024'),
            rz_low=dict(channels={c:{g:dict(status='complete',fit_converged=True,RZ=v)
            for g,v in (('Nb1',2.),('Nb2plus',3.))} for c in ('DY2E','DY2M')}))
    return payload,rz


class PredictionTests(unittest.TestCase):
    def test_six_era_sr_display_is_explicit_blinded_and_yield_preserving(self):
        from TROTASR.utils.paths import ERAS
        from TROTASR.gnn4lowdm.configuration import configuration_path
        rows = model_records()
        channels = {y+k[4:]:deepcopy(v) for y in ERAS for k,v in rows.items()}
        payload = dict(schema='trotasr_card_prediction_plot_v1',status='complete',
            scope='full_run3_hybrid_production',prediction_stage='prefit',
            sr_observations_included=False,rate_initials_applied=True,channels=channels,
            sr_binning_sha256=sha256(ROOT/'jsons/highdm_sr_binning.json'),
            gnn_sha256=sha256(configuration_path('2022')))
        with patch('TROTASR.workflows.diagnostic_common.require_revised_record',return_value=True), \
             patch('TROTASR.workflows.plotting_card_predictions.prediction_years',return_value=ERAS):
            with self.assertRaisesRegex(ValueError,'explicit'):prepare_cards(payload)
            result=prepare_cards(payload,combine_era_sr=True)
            self.assertEqual(set(result),{('highdm','SR')})
            np.testing.assert_allclose(result['highdm','SR']['background'],
                6*np.array(rows['2024/SR/highdm']['background']))
            for years in (ERAS[:4], ERAS[4:]):
                result=prepare_cards(payload,combine_era_sr=True,display_years=years)
                np.testing.assert_allclose(result['lowdm','SR']['background'],
                    len(years)*np.array(rows['2024/SR/lowdm']['background']))
                self.assertNotIn('data',result['lowdm','SR'])
            channels['2022/SR/lowdm']['data']=[0.]*30
            with self.assertRaisesRegex(ValueError,'SR observation'):
                prepare_cards(payload,combine_era_sr=True,display_years=ERAS[:4])

    def test_no_unrequested_footer_or_renderer_interception(self):
        source=(ROOT/'workflows/plotting_card_predictions.py').read_text()
        self.assertNotIn('figure.text(',source)
        self.assertNotIn('Figure.savefig',source)
        self.assertNotIn('combined_sr_display',source)

    def test_current_card_matches_direct_initial_scaled_yields(self):
        records=model_records()
        r=records['2024/SR/lowdm']
        # GNN x native-recoil projection: 4*(3+4+5+4*6)=144, not score-index factors.
        np.testing.assert_allclose(r['groups']['ZJet'],np.full(30,144.))
        np.testing.assert_allclose(r['group_variances']['ZJet'],np.full(30,16*(9+16+25+4*36)))
        self.assertNotIn('data',r)
        self.assertTrue(r['nuisance_deltas'])
        self.assertFalse(any('sgamma_shape' in k for k in r['nuisance_deltas']))

    def test_full_sr162_and_cr10_even_for_empty_categories(self):
        records=model_records()
        high=combine([records['2024/SR/highdm']]);blocks=flat_blocks(high,'highdm','SR')
        self.assertEqual(len(blocks),28)
        self.assertEqual(sum(b['nbin'] for b in blocks),162)
        self.assertTrue(any(not b['background'].any() for b in blocks))
        self.assertTrue(all(b['blind_data'] and not b['data'].any() for b in blocks))
        self.assertTrue(all('N_{mix}' in b['label'] and 'N_{W}' in b['label'] for b in blocks))
        self.assertTrue(all(b['label_fontsize']==9.5 and b['label_box_pad']==.12 for b in blocks))
        for region in ('LLCR','QCDCR','GCR'):
            b=flat_blocks(combine([records['2024/'+region+'/lowdm']]),'lowdm',region)
            self.assertEqual([x['nbin'] for x in b],[5] if region=='LLCR' else [5,5])
            self.assertTrue(all(not x['signals'] and not x['blind_data'] for x in b))
            if region!='LLCR':
                self.assertTrue(all(x['label_fontsize']==15 and 'label_box_pad' not in x for x in b))

    def test_correlated_cancellation_happens_before_envelope(self):
        r=model_records()['2024/LLCR/lowdm'];n=len(r['background'])
        r['nuisance_deltas']={'shared':dict(up=np.ones(n)*3,down=-np.ones(n)*2)}
        s=deepcopy(r);s['nuisance_deltas']={'shared':dict(up=-np.ones(n)*3,down=np.ones(n)*2)}
        combined=combine([r,s])
        np.testing.assert_allclose(combined['uncertainty'],np.sqrt(r['variance']*2))
        s['nuisance_deltas']['independent']=s['nuisance_deltas'].pop('shared')
        np.testing.assert_allclose(combine([r,s])['uncertainty'],np.sqrt(r['variance']*2+18))

    def test_ll_display_projects_shared_nuisances_before_envelope(self):
        r=combine([model_records()['2024/LLCR/lowdm']])
        r['nuisance_deltas']={'same':dict(up=np.r_[np.ones(5)*3,-np.ones(5)*3],
                                        down=np.r_[-np.ones(5)*2,np.ones(5)*2])}
        before=deepcopy(r)
        p=ll_display_projection(r)
        np.testing.assert_allclose(p['uncertainty'],np.sqrt(r['variance'][:5]+r['variance'][5:]))
        for field in ('data','background','variance','data_variance'):
            np.testing.assert_allclose(p[field],r[field][:5]+r[field][5:])
            np.testing.assert_array_equal(r[field],before[field])
        blocks=flat_blocks(r,'lowdm','LLCR')
        self.assertEqual(len(blocks),1)
        self.assertEqual(blocks[0]['label'],'')
        self.assertFalse(blocks[0]['label_box'])
        self.assertEqual(blocks[0]['annotation'],r'Low-$\Delta m$ LLCR')
        self.assertEqual(len(r['background']),10)

    def test_yearly_controls_bind_exact_current_card_inputs_no_unit_area(self):
        payload=dict(status='complete',scope='full_nominal_production',prediction_stage='prefit',
            rate_initials_applied=True,sr_observations_included=False,channels=model_records(),
            source_inputs={'2024':dict(hists=dict(sha256='hist'),rz_low=dict(sha256='rz'))})
        result=current_card_controls(payload,2024,'hist','rz')
        self.assertEqual(set(result),{'LLCR','QCDCR','GCR'})
        for region,r in result.items():
            np.testing.assert_allclose(r['background'],payload['channels']['2024/'+region+'/lowdm']['background'])
            self.assertTrue(all(not b['unit_area'] for b in flat_blocks(r,'lowdm',region)))
        with self.assertRaisesRegex(ValueError,'differ'):current_card_controls(payload,2024,'stale','rz')
        payload['prediction_stage']='postfit'
        with self.assertRaisesRegex(ValueError,'prefit'):current_card_controls(payload,2024,'hist','rz')

    def test_real_legacy_llcr_reference_reproduced_not_used_for_new_fit(self):
        folder=ROOT/'validation/gnn_prefit_binding_20260923/reference'
        payload=read_json(folder/'legacy_card_predictions.json')
        manifest=read_json(folder/'legacy_llcr_manifest.json')
        self.assertEqual(sha256(folder/'legacy_card_predictions.json'),manifest['source_sha256'])
        for expected in manifest['plots']:
            sources=[payload['channels'][name] for name in expected['source_channels']]
            record=dict(sr_data_blinded=False,signals={},groups={},nuisance_deltas={})
            for field in ('data','background','variance'):
                record[field]=sum(np.asarray(r[field]) for r in sources)
            record['data_variance']=record['data'].copy()
            for r in sources:
                for k,v in r['groups'].items():
                    record['groups'].setdefault(k,np.zeros(10))[:]+=v
                for k,pair in r['nuisance_deltas'].items():
                    dst=record['nuisance_deltas'].setdefault(k,dict(up=np.zeros(10),down=np.zeros(10)))
                    for d in ('up','down'):dst[d]+=pair[d]
            result=ll_display_projection(record)
            for field,target in (('data','data'),('background','background'),
                                  ('variance','mc_stat_variance'),('uncertainty','total_uncertainty')):
                np.testing.assert_allclose(result[field],expected[target],rtol=1e-12,atol=1e-8)

    def test_five_final_card_plots_exact_years_and_signal_choices(self):
        payload=dict(schema='trotasr_card_prediction_plot_v1',status='complete',
            scope='full_nominal_production',prediction_stage='prefit',sr_observations_included=False,
            sr_binning_sha256=sha256(ROOT/'highdm_sr_binning.json'),gnn_sha256=sha256(ROOT/'gnn4lowdm/config.json'),
            rate_initials_applied=True,channels=json_ready({**model_records(),**model_records(2025)}))
        result=prepare_cards(payload)
        self.assertEqual(len(result),5)
        self.assertEqual(set(result['lowdm','SR']['signals']),set(LOW_SIGNALS))
        changed=deepcopy(payload);changed['channels']['2024/SR/highdm']['labels'][0]='old_bin'
        with self.assertRaisesRegex(ValueError,'full native order'):prepare_cards(changed)
        del payload['channels']['2025/SR/highdm']
        with self.assertRaisesRegex(ValueError,'year or plot channel'):prepare_cards(payload)

    def test_observation_or_label_mismatch_rejected(self):
        r=model_records()['2024/SR/highdm'];r['data']=np.zeros(162)
        with self.assertRaisesRegex(ValueError,'SR observation'):combine([r])
        r.pop('data');s=deepcopy(r);s['labels'][0]='different'
        with self.assertRaisesRegex(ValueError,'Mixed layout'):combine([r,s])

    def test_lnN_asymmetry_and_rate_scaling_exactly_once(self):
        c='LLCR_lowdm_c1_2024';labels=expected_labels(c)
        card=dict(columns=[(c,'PhotonJet')],nuisances=[['foo','lnN','.8/1.3']],
                  rateparams=[['q','rateParam',c,'PhotonJet','2','[0,10]']])
        directory={'PhotonJet':Histogram([3.]*10,[5.]*10,labels),
                   'data_obs':Histogram([10.]*10,[10.]*10,labels)}
        r=channel_prediction(card,c,directory,labels)
        np.testing.assert_allclose(r['background'],6.)
        np.testing.assert_allclose(r['variance'],20.)
        np.testing.assert_allclose(r['nuisance_deltas']['foo']['up'],1.8)
        np.testing.assert_allclose(r['nuisance_deltas']['foo']['down'],-1.2)

    def test_unsupported_card_directives_never_ignored(self):
        inp=build_inputs(*fixture(),'T2tt',1200,500)
        channels,data,_,_=assemble([inp]);text=card_text(channels,data)
        for extra in ('\nx param 0 1\n','\nx flatParam\n'):
            with self.assertRaisesRegex(ValueError,'Unsupported card'):parse_card(text+extra)
        with self.assertRaisesRegex(ValueError,'MC statistics'):
            parse_card(text.replace('autoMCStats 10 1 1','autoMCStats 0 1 1'))

    def test_changed_or_malformed_th1_fails(self):
        h=Histogram([1.],[1.],['wrong'])
        with self.assertRaisesRegex(ValueError,'TH1'):read_th1({'p':h},'p',['expected'])
        h.label=['expected'];h.value=np.asarray([float('nan')])
        with self.assertRaisesRegex(ValueError,'prediction array'):read_th1({'p':h},'p',['expected'])
        h.value=np.asarray([-1.])
        with self.assertRaisesRegex(ValueError,'prediction array'):read_th1({'p':h},'p',['expected'])

    def test_band_reproduces_source_and_statistically_independent_sum(self):
        r=combine([model_records()['2024/SR/lowdm']])
        b=uncertainty_band(r)
        np.testing.assert_allclose(renderer.validated_plot_uncertainty(r['background'],r['variance'],b),r['uncertainty'])


class GNNControlTests(unittest.TestCase):
    def test_rz_channel_nb_yields_and_variance_no_double_application(self):
        p,rz=raw_fixture();before=deepcopy(p)
        out=prepare(p,rz,True)
        self.assertEqual(p,before)
        for region in ('DY2E','DY2M'):
            for category,factor in zip(POOLED,(2,3)):
                raw=out['histograms']['nominal'][region][category]
                scaled=out['histograms']['nominal_rz'][region][category]
                for key,power in (('sumw',1),('sumw2',2)):
                    np.testing.assert_allclose(scaled['DY']['gnn_score'][key],np.asarray(raw['DY']['gnn_score'][key])*factor**power)
                self.assertEqual(scaled['ST'],raw['ST'])

    def test_ll_nb_pooling_display_only_preserves_every_bin_and_variance(self):
        p=prepare(*raw_fixture(),True);out=ll_inclusive(p)
        self.assertEqual(len(p['histograms']['nominal']['LLCR']),2)
        self.assertEqual(out['histograms']['nominal']['GCR'],p['histograms']['nominal']['GCR'])
        original=p['histograms']['nominal']['LLCR']
        pooled=out['histograms']['nominal']['LLCR']['Nb1plus_NISRinclusive']
        for sample,axes in pooled.items():
            for f in ('sumw','sumw2','entries'):
                np.testing.assert_allclose(axes['gnn_score'][f],sum(np.asarray(original[c][sample]['gnn_score'][f]) for c in POOLED))

    def test_no_failed_fit_unity_or_nearest_binning_fallback(self):
        p,rz=raw_fixture();rz['rz_low']['channels']['DY2E']['Nb1']['fit_converged']=False
        with self.assertRaisesRegex(ValueError,'RZ measurement'):prepare(p,rz,True)
        p,rz=raw_fixture();p['histograms']['lowdm']['nominal']['LLCR']['Nb1']['ST']['gnn_score']['edges'][1]=.2
        with self.assertRaisesRegex(ValueError,'histogram edges'):prepare(p,rz,True)

    def test_valid_boundary_rz_zero_preserved_not_unity(self):
        p,rz=raw_fixture();rz['rz_low']['channels']['DY2M']['Nb1']['RZ']=0.
        out=prepare(p,rz,True)
        self.assertEqual(out['histograms']['nominal_rz']['DY2M'][POOLED[0]]['DY']['gnn_score']['sumw'],[0.]*5)
        self.assertTrue(any(a['valid_boundary_fit'] for a in out['rz_application']))

    def test_full_scope_and_data_required(self):
        p,rz=raw_fixture()
        with self.assertRaisesRegex(ValueError,'Full production'):prepare(p,rz)
        del p['histograms']['lowdm']['nominal']['LLCR']['Nb2']['data']
        with self.assertRaisesRegex(ValueError,'actual CR data'):prepare(p,rz,True)


if __name__=='__main__':unittest.main()
