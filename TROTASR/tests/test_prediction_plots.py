"""Prediction/renderer contracts, not substitute physics fits or production plots."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json
from TROTASR.utils.statistics_inputs import build_inputs
from TROTASR.utils.statistical_templates import assemble,card_text,json_ready
from TROTASR.tests.test_statistics_inputs import fixture
from TROTASR.workflows.prediction_plot_inputs import (
    parse_card,channel_prediction,combine,flat_blocks,expected_labels,read_th1,
    uncertainty_band,renderer,LOW_SIGNALS,ll_display_projection)
from TROTASR.workflows.plotting_gnn_controls import prepare,ll_inclusive,POOLED,current_card_controls
from TROTASR.workflows.plotting_card_predictions import prepare as prepare_cards


def sr_binning_path():
    path = ROOT / 'jsons/highdm_sr_binning.json'
    if not path.is_file():
        raise unittest.SkipTest('Canonical SR binning payload is not installed locally')
    return path


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
    sr_binning_path()
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
    def test_provisional_cr_projection_accepts_covqual_two_without_writing(self):
        from TROTASR.utils.era_merge import CHANNEL_BINS, ERAS
        from TROTASR.workflows import prediction_plot_inputs as prediction

        channels = {}
        for name, size in CHANNEL_BINS.items():
            if not name.startswith('SR_'):
                band = dict(background=[1.] * size, background_unc=[.2] * size)
                channels[name] = dict(data=[1] * size, prefit=band, postfit=band)
        payload = dict(schema='trotasr_cronly_prefit_postfit_v1', sr_observations_included=False,
                       fit_recalculated=False, fit=dict(covariance_quality=2, process_exit_code=0),
                       years=ERAS, channels=channels)
        with TemporaryDirectory() as temp:
            result = Path(temp)
            (result/'controller.lock').touch()
            publication = dict(process_exit_code=-6)
            output = result/'output'
            with patch.object(prediction, 'internal_path', side_effect=Path), \
                    patch.object(prediction, 'read_json', side_effect=[publication, payload]), \
                    patch.object(prediction, 'provisional_cronly_products',
                                 return_value={'projection': result/'projection.json'}) as products:
                self.assertIs(prediction.export_cronly_fit(result, output, write=False), payload)
            products.assert_called_once_with(result, publication, required=('projection',))
            self.assertFalse(output.exists())

    def test_six_era_sr_display_is_explicit_blinded_and_yield_preserving(self):
        from TROTASR.utils.paths import ERAS
        from TROTASR.gnn4lowdm.configuration import configuration
        from TROTASR.workflows import plotting_card_predictions as plots, prediction_plot_inputs as inputs
        analysis = read_json(ROOT/'jsons/analysis.json')
        late = configuration('2024')
        early = deepcopy(late)
        # Exercise distinct early/late edge contracts without opening remote assets.
        for edges in early['sr_binning']['edges_by_category'].values():
            edges[1] *= .9
        merge = {'bins_per_year': 118}
        payload = dict(schema='trotasr_card_prediction_plot_v1',status='complete',
            scope='full_combined_systematic_production',prediction_stage='prefit',
            sr_observations_included=False,rate_initials_applied=True,channels={},
            sr_binning_sha256='fixture-layout',gnn_sha256='fixture-layout',sr_merge=merge,
            years=list(ERAS), bins=214*len(ERAS), th1_channels=8*len(ERAS),
            rate_parameters=48*len(ERAS), sr_data_blinded=True, signal_contamination=False,
            auto_mc_stats=[10,1,1], central_conventions_consistent=False)
        for key in ('run3_systematics','statistical_revision','systematic_evaluation_deferred',
                    'systematic_scope','luminosity_uncertainties','luminosity_model',
                    'systematic_combination','background_estimation_uncertainties','preliminary'):
            payload[key] = deepcopy(analysis[key])
        config_for = lambda year: early if year in ERAS[:4] else late
        with patch.object(plots,'load_merge',return_value=merge), \
             patch.object(plots,'configuration',side_effect=config_for), \
             patch.object(inputs,'configuration',side_effect=config_for):
            for era_index, year in enumerate(ERAS, 1):
                for mode, regions in (('highdm',('SR',)),('lowdm',('SR','LLCR','QCDCR','GCR'))):
                    for region in regions:
                        labels = expected_labels(f'{region}_{mode}_c1_{year}', merge)
                        values = np.full(len(labels), float(era_index))
                        signals = ([s['key'] for s in renderer.SIGNAL_OVERLAYS] if mode=='highdm'
                                   else LOW_SIGNALS if region=='SR' else ())
                        row = dict(labels=labels,sr_data_blinded=region=='SR',background=values,
                            variance=values,groups={'Top':values},group_variances={'Top':values},
                            nuisance_deltas={},signals={name:values for name in signals})
                        if region!='SR':
                            row.update(data=values,data_variance=values)
                        payload['channels'][f'{year}/{region}/{mode}'] = row
            with self.assertRaisesRegex(ValueError,'explicit'):prepare_cards(payload)
            with self.assertRaisesRegex(ValueError,'approved low-dM era groups'):
                prepare_cards(payload,combine_era_sr=True,display_years=ERAS)
            result=prepare_cards(payload,combine_era_sr=True)
            self.assertEqual(set(result),{('highdm','SR')})
            np.testing.assert_array_equal(result['highdm','SR']['background'],np.full(118,21.))
            self.assertNotIn('data',result['highdm','SR'])
            for years, expected in ((ERAS[:4],10.),(ERAS[4:],11.)):
                result=prepare_cards(payload,combine_era_sr=True,display_years=years)
                np.testing.assert_array_equal(result['lowdm','SR']['background'],np.full(30,expected))
                self.assertNotIn('data',result['lowdm','SR'])
            with patch.object(plots,'configuration',side_effect=lambda y: late if y=='2023' else config_for(y)):
                with self.assertRaisesRegex(ValueError,'identical GNN boundaries'):
                    prepare_cards(payload,combine_era_sr=True,display_years=ERAS[:4])
            wrong=deepcopy(payload)
            wrong['run3_systematics']['preserve_per_era_gnn']=False
            with self.assertRaisesRegex(ValueError,'six-era systematic policy'):
                prepare_cards(wrong,combine_era_sr=True)
            payload['channels']['2022/SR/lowdm']['data']=[0.]*30
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

    def test_yearly_controls_use_current_prefit_labels(self):
        payload=dict(status='complete',scope='full_nominal_production',prediction_stage='prefit',
            rate_initials_applied=True,sr_observations_included=False,channels=model_records())
        result=current_card_controls(payload,2024)
        self.assertEqual(set(result),{'LLCR','QCDCR','GCR'})
        for region,r in result.items():
            np.testing.assert_allclose(r['background'],payload['channels']['2024/'+region+'/lowdm']['background'])
            self.assertTrue(all(not b['unit_area'] for b in flat_blocks(r,'lowdm',region)))
        payload['prediction_stage']='postfit'
        with self.assertRaisesRegex(ValueError,'prefit'):current_card_controls(payload,2024)

    def test_five_final_card_plots_exact_years_and_signal_choices(self):
        payload=dict(schema='trotasr_card_prediction_plot_v1',status='complete',
            scope='full_nominal_production',prediction_stage='prefit',sr_observations_included=False,
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
        text = ('bin LLCR_lowdm_c1_2024\nobservation -1\n'
                'bin LLCR_lowdm_c1_2024\nprocess PhotonJet\nprocess 1\nrate -1\n'
                '* autoMCStats 10 1 1\n')
        self.assertEqual(parse_card(text)['columns'], [('LLCR_lowdm_c1_2024', 'PhotonJet')])
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


class BackgroundSummaryTests(unittest.TestCase):
    def test_accepts_current_card_labels_and_rejects_unknown_background(self):
        from TROTASR.workflows import validation_regions as vr
        tasks = {era: [{'year': era, 'input': f'inputs/{era}/mc_shard_00001.root'}]
                 for era in vr.ERAS}
        low = {'Nb1_NISR0': [0., .2, .4, .6, .8, 1.]}
        labels = ('DY', 'GJ', 'WtoLNu', 'Zto2Nu')
        receipt = dict(signal_points_seen=[], bins={label: {'0': [1., 2., 3.]} for label in labels})
        with patch.object(vr, 'geometry', return_value=(low, [])), \
             patch.object(vr, '_read_receipt', return_value=receipt):
            result = vr._collect(tasks, False, background_only=True)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(set(result['samples']), set(labels))
            self.assertEqual(result['samples']['DY'][0], [6., 12., 18.])
            receipt['bins']['undefined'] = {'0': [1., 2., 3.]}
            with self.assertRaisesRegex(ValueError, 'unknown process'):
                vr._collect(tasks, False, background_only=True)
            del receipt['bins']['undefined']
            receipt['signal_points_seen'] = ['T2tt/mStop1200_mLSP500']
            with self.assertRaisesRegex(ValueError, 'Signal or unknown process'):
                vr._collect(tasks, False, background_only=True)


if __name__=='__main__':unittest.main()
