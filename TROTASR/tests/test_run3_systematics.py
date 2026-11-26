"""Six-era systematic routing; no synthetic result is admitted to a fit."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from TROTASR.utils.paths import ERAS, ROOT
from TROTASR.utils.io import read_json
from TROTASR.utils import era_merge
from TROTASR.utils.cr_binning import FIT_GROUPS, N_BINS, N_HIGH_CR, N_RATES
from TROTASR.workflows import build_nominal_grid as builder
from TROTASR.workflows.diagnostic_common import record_years


def luminosity_policy():
    return read_json(ROOT / 'jsons/analysis.json')['luminosity_model']


def config():
    return dict(years=list(ERAS),run3_systematics=copy.deepcopy(builder.RUN3_SYSTEMATICS_POLICY),
                systematic_evaluation_deferred=False,systematic_scope='Six-era experimental systematics',
                luminosity_uncertainties={},luminosity_model=luminosity_policy())


def card_fixture(years=ERAS):
    """Exercise family eligibility independently of process aliases or region."""
    import numpy as np
    channels = {}; data = {}
    for year in years:
        for mode in ('highdm', 'lowdm'):
            for region in ('SR', 'LLCR', 'QCDCR', 'GCR'):
                channel = f'{region}_{mode}_c1_{year}'
                channels[channel] = [dict(year=year)]
                processes = [(f'{binding}_{family}', family, binding == 'controlled')
                             for family in ('Top', 'WJet', 'ZJet', 'QCD', 'VV', 'DYJet', 'PhotonJet')
                             for binding in ('fixed', 'controlled')]
                if region == 'SR':
                    processes.append(('signal', 'signal', mode == 'lowdm'))
                for process, family, controlled in processes:
                    nominal = np.array([2., 3.])
                    data[channel, process] = dict(family=family,
                        parameter=f'norm_{process}_{channel}' if controlled else '', initial=1.2,
                        nominal=nominal, sumw2=np.array([.2, .3]),
                        variations={'CMS_test_shape': {'up': nominal * 1.1, 'down': nominal * .9},
                                    'CMS_test_flat': {'up': nominal.copy(), 'down': nominal.copy()}},
                        extra_lnN=[('CMS_test_measurement', .97, 1.04)])
    return channels, data


class Run3SystematicsTests(unittest.TestCase):
    def test_display_window_admits_only_uploaded_grid_points_in_scope(self):
        from TROTASR.workflows.combine_limits import ready_fit_points
        models={'T2tt':205,'T2bW':345,'T2tb':345}
        outside = [dict(model=m,mass=f'mStop{500+i}_mLSP100') for m in models for i in range(20)]
        inside = [dict(model=m,mass=f'mStop{stop}_mLSP{i}')
                  for m,num in models.items() for i in range(1,num+1)
                  for stop in (1800 if i == num else 600,)]
        source=dict(status='cards_ready', scope='full_combined_systematic_production',
                    era_merged=True, sr_data_blinded=True, points=outside+inside)
        before=copy.deepcopy(source)
        selected, scope = ready_fit_points(source)
        self.assertEqual(era_merge.displayed_points(source['points']), selected)
        self.assertEqual(selected,inside)
        self.assertEqual((scope['total'],scope['excluded_out_of_plot_range']), (895,60))
        self.assertEqual(scope['expected_points_by_model'],models)
        self.assertEqual(source,before)
        with self.assertRaises(ValueError):
            ready_fit_points(dict(source, era_merged=False))
        with self.assertRaisesRegex(ValueError,'955-point inventory'):
            era_merge.displayed_points(source['points'][1:])

    def test_official_luminosity_matrix_preserves_native_process_mask(self):
        import numpy as np
        from TROTASR.utils.statistical_templates import json_ready
        policy = luminosity_policy()
        channels, data = card_fixture()
        before = json_ready(copy.deepcopy(data))
        channels_before = copy.deepcopy(channels)
        policy_before = copy.deepcopy(policy)
        for cr_only in (False, True):
            for revision in (None, builder.CR12_POLICY):
                with self.subTest(cr_only=cr_only, revision=bool(revision)):
                    # Construct the baseline first, so a mutation cannot hide in it.
                    baseline = builder.card_text(channels, data, cr_only=cr_only,
                        statistical_revision=revision, luminosity_uncertainties={})
                    text = builder.card_text(channels, data, cr_only=cr_only,
                        statistical_revision=revision, luminosity_uncertainties={}, luminosity_model=policy)
                    lines = [line.split() for line in text.splitlines()]
                    lumi_lines = [row for row in lines if row[0].startswith('lumi_')]
                    self.assertEqual(len(lumi_lines), len(policy['components']))
                    self.assertEqual({row[0] for row in lumi_lines}, set(policy['components']))
                    self.assertTrue(all(row[1] == 'lnN' for row in lumi_lines))
                    self.assertEqual('\n'.join(line for line in text.splitlines()
                        if not line.startswith('lumi_')) + '\n', baseline)
                    self.assertIn('CMS_test_shape shape ', text)
                    self.assertIn('CMS_test_measurement lnN ', text)
                    self.assertNotIn('CMS_test_flat shape ', text)
                    column_channels = [row[1:] for row in lines if row[0] == 'bin'][-1]
                    column_processes = [row[1:] for row in lines if row[0] == 'process'][0]
                    columns = list(zip(column_channels, column_processes))
                    self.assertEqual(len(column_channels), len(column_processes))
                    self.assertEqual({c for c, _ in columns},
                        {c for c in channels if not cr_only or not c.startswith('SR_')})
                    for row in lumi_lines:
                        factors = policy['components'][row[0]]
                        expected = []
                        for channel, process in columns:
                            record = data[channel, process]
                            era = channel.rsplit('_', 1)[1]
                            eligible = record['family'] == 'signal' or (
                                record['family'] in ('VV', 'DYJet', 'PhotonJet') and not record['parameter'])
                            expected.append(str(factors[era]) if eligible and era in factors else '-')
                        self.assertEqual(row[2:], expected, row[0])
                    self.assertEqual(json_ready(data), before)
                    self.assertEqual(channels, channels_before)
                    self.assertEqual(policy, policy_before)
                    self.assertEqual(builder.card_text(channels, data, cr_only=cr_only,
                        statistical_revision=revision, luminosity_uncertainties={}, luminosity_model=policy), text)
        a=np.array([[1.38,.17,.20],[0.,1.27,.68],[0.,0.,1.44]])
        np.testing.assert_allclose(a.T@a,np.array(policy['calendar_year_covariance_percent2']),rtol=0,atol=.01)
        bad=copy.deepcopy(policy);bad['components']['lumi_13p6TeV_2025']['2025']=1.016
        with self.assertRaises(ValueError): builder.card_text(channels,data,luminosity_model=bad)
        with self.assertRaises(ValueError): builder.card_text(channels,data,luminosity_uncertainties={'2024':1.016},luminosity_model=policy)

    def test_run3_luminosity_policy_is_required_before_input_access(self):
        c = config()
        cases = []
        for key in ('luminosity_model', 'luminosity_uncertainties'):
            missing = copy.deepcopy(c); missing.pop(key)
            cases.append((f'missing_{key}', missing))
            cases.append((f'null_{key}', dict(c, **{key: None})))
        cases.append(('independent_lumi', dict(c, luminosity_uncertainties={'2024': 1.016})))
        for key, value in (('central_normalization_unchanged', False), ('early_normalization_fb', 62.31)):
            bad = copy.deepcopy(c); bad['luminosity_model'][key] = value
            cases.append((key, bad))
        changed = copy.deepcopy(c)
        changed['luminosity_model']['components']['lumi_13p6TeV_2025']['2025'] = 1.016
        cases.append(('changed_factor', changed))
        missing_era = copy.deepcopy(c)
        missing_era['luminosity_model']['components']['lumi_1'].pop('2022EE')
        cases.append(('missing_era', missing_era))
        for name, bad in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, '[Ll]uminosity'):
                    builder.require_run3_systematics_policy(bad)
                with patch.object(builder, 'internal_path', side_effect=AssertionError('Unexpected input access')):
                    with self.assertRaisesRegex(ValueError, '[Ll]uminosity'):
                        builder.grid_contract(Path('unused-config'), bad, Path('unused-grid'), False)
        metadata_only = copy.deepcopy(c)
        metadata_only['luminosity_model'].update(id='receipt-label', user_approved=False, approved_date='changed')
        self.assertTrue(builder.require_run3_systematics_policy(metadata_only))

    def test_two_era_validation_uses_native_luminosity_without_the_run3_model(self):
        from TROTASR.utils.statistical_templates import card_text as native
        years = ERAS[4:]
        with patch.object(builder, 'read_json', side_effect=AssertionError('Unexpected luminosity model access')):
            self.assertFalse(builder.require_run3_systematics_policy(dict(years=list(years))))
        channels, data = card_fixture(years)
        for uncertainties in (None, {'2024':1.016,'2025':1.016}):
            for cr_only in (False, True):
                with self.subTest(uncertainties=uncertainties, cr_only=cr_only):
                    self.assertEqual(builder.card_text(channels, data, cr_only=cr_only,
                        luminosity_uncertainties=uncertainties), native(channels, data,
                        cr_only=cr_only, luminosity_uncertainties=uncertainties))

    def test_six_era_card_preserves_each_era_nuisance_binding(self):
        channels, data = card_fixture()
        for (channel, _), record in data.items():
            era = channel.rsplit('_',1)[1]
            record['variations']['CMS_test_per_era_'+era] = copy.deepcopy(record['variations']['CMS_test_shape'])
        text = builder.card_text(channels, data, statistical_revision=builder.CR12_POLICY,
                                 luminosity_model=luminosity_policy())
        rows = [line.split() for line in text.splitlines()]
        columns = [row[1:] for row in rows if row[0]=='bin'][-1]
        self.assertEqual({c.rsplit('_',1)[1] for c in columns},set(ERAS))
        for era in ERAS:
            shape = next(row for row in rows if row[0]=='CMS_test_per_era_'+era)
            self.assertEqual(shape[1],'shape')
            self.assertEqual(shape[2:], ['1' if col.rsplit('_',1)[1]==era else '-' for col in columns])

    def test_per_era_controls_use_only_the_requested_era(self):
        from TROTASR.workflows.plotting_gnn_controls import current_card_controls
        from TROTASR.workflows.prediction_plot_inputs import expected_labels
        current = read_json(ROOT/'jsons/analysis.json')
        records = {}
        for era,value in (('2024',1.),('2025',2.)):
            for region in ('LLCR','QCDCR','GCR'):
                bkg = np.full(10,value)
                records[f'{era}/{region}/lowdm'] = dict(labels=expected_labels(f'{region}_lowdm_c1_{era}'),
                    sr_data_blinded=False, background=bkg, variance=np.ones(10), groups={'ST':bkg},
                    group_variances={'ST':np.ones(10)}, nuisance_deltas={}, rateparam_initials={},
                    signals={}, data=np.ones(10), data_variance=np.ones(10))
        payload = dict(config(),scope='full_combined_systematic_production', status='complete',
            bins=6 * N_BINS, rate_parameters=6 * N_RATES, signal_contamination=False, auto_mc_stats=[10,1,1],
            sr_data_blinded=True, central_conventions_consistent=False, preliminary=True,
            prediction_stage='prefit', statistical_revision=builder.CR12_POLICY,
            systematic_combination=current['systematic_combination'],
            background_estimation_uncertainties=current['background_estimation_uncertainties'],
            rate_initials_applied=True, sr_observations_included=False, channels=records)
        for era,value in (('2024',1.),('2025',2.)):
            result = current_card_controls(payload,era)
            self.assertEqual(set(result),{'LLCR','QCDCR','GCR'})
            for row in result.values():
                np.testing.assert_array_equal(row['background'],np.full(10,value))

    def test_prediction_metadata_retains_exact_inherited_luminosity_policy(self):
        from TROTASR.workflows.prediction_plot_inputs import (
            _prediction_metadata, prediction_years, select_era)
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        source = dict(config(), statistical_revision=copy.deepcopy(builder.CR12_POLICY),
            rate_parameters=1, signal_contamination=False, auto_mc_stats=[10, 1, 1],
            background_estimation_uncertainties=dict(id='jme_background_shapes_20260929',
                mode='shape', auto_mc_stats=[10, 1, 1], user_approved=True),
            systematic_combination=copy.deepcopy(COMBINATION_POLICY), preliminary=True,
            central_conventions_consistent=False, bins=N_BINS * len(ERAS), th1_channels=8 * len(ERAS),
            sr_data_blinded=True, luminosity_policy_sha256='a'*64)
        before = copy.deepcopy(source)
        derived = _prediction_metadata(source)
        self.assertEqual(record_years(derived), ERAS)
        assert 'luminosity_policy_sha256' not in derived
        for key in ('luminosity_uncertainties', 'luminosity_model'):
            self.assertIs(derived[key], source[key])
        payload = dict(derived, status='complete', scope='full_combined_systematic_production',
            prediction_stage='prefit', rate_initials_applied=True, sr_observations_included=False,
            source_inputs={year: {} for year in ERAS}, channels={
                f'{year}/{region}/{mode}': dict(sr_data_blinded=region == 'SR')
                for year in ERAS for mode, regions in (
                    ('highdm', ('SR',)), ('lowdm', ('SR', 'LLCR', 'QCDCR', 'GCR')))
                for region in regions})
        for year in ERAS:
            view = select_era(payload, year)
            self.assertEqual(prediction_years(view), (year,))
            parent = view['single_era_view']['source_model']
            self.assertEqual(record_years(parent), ERAS)
            for key in ('luminosity_uncertainties', 'luminosity_model'):
                self.assertEqual(view[key], source[key])
                self.assertEqual(parent[key], source[key])
        for key in ('luminosity_uncertainties', 'luminosity_model'):
            missing = copy.deepcopy(source); missing.pop(key)
            result = _prediction_metadata(missing)
            self.assertNotIn(key, result)
            with self.assertRaisesRegex(ValueError, '[Ll]uminosity'):
                record_years(result)
        altered = copy.deepcopy(source)
        altered['luminosity_uncertainties'] = {'2025': 1.016}
        with self.assertRaisesRegex(ValueError, '[Ll]uminosity'):
            record_years(_prediction_metadata(altered))
        self.assertEqual(source, before)


    def test_explicit_six_era_policy(self):
        c = config()
        self.assertTrue(builder.require_run3_systematics_policy(c))
        self.assertEqual(record_years(c), ERAS)
        for key,value in [('systematic_evaluation_deferred',True),('years',list(ERAS[:-1])),
                          ('systematic_scope',''),
                          ('run3_systematics',dict(builder.RUN3_SYSTEMATICS_POLICY,preserve_per_era_gnn=False))]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                builder.require_run3_systematics_policy(dict(c,**{key:value}))

    def test_no_nominal_or_partial_era_can_enter_systematic_combination(self):
        caches = {y:dict(scope='full_combined_systematic_production') for y in ERAS}
        self.assertEqual(builder.combined_input_scope(caches,config()),'full_combined_systematic_production')
        with self.assertRaisesRegex(ValueError,'adopted systematic settings'):
            builder.combined_input_scope(caches,dict(years=list(ERAS)))
        with self.assertRaisesRegex(ValueError,'canonical six-era inputs'):
            builder.grid_contract(Path('unused-config'),dict(years={era:{} for era in ERAS}),
                                  Path('unused-grid'),True)
        for year in ERAS:
            for scope in ('full_nominal_production','full_cms_trota_jme_production','representative_validation'):
                bad=copy.deepcopy(caches);bad[year]['scope']=scope
                with self.subTest(year=year,scope=scope),self.assertRaises(ValueError):
                    builder.combined_input_scope(bad,config())

    def test_card_dispatch_preserves_blinding_and_all_eras(self):
        from TROTASR.workflows import combine_limits as limits
        point=dict(model='T2tt',mass='mStop1200_mLSP500',card='stats/synthetic.txt',card_sha256='test')
        meta=dict(config(),inputs=[],cr_only=False,sr_data_blinded=True,card_sha256='test',
                  auto_mc_stats=[10,1,1],topology='T2tt',mStop=1200,mLSP=500)
        with patch.object(limits,'read_json',return_value=meta), patch.object(Path,'read_text',return_value=''), \
             patch.object(Path,'is_file',return_value=True), patch.object(Path,'stat',return_value=SimpleNamespace(st_size=1)):
            self.assertEqual(limits.verify_card(point),ROOT/'stats/synthetic.txt')
            meta['sr_data_blinded']=False
            with self.assertRaises(ValueError): limits.verify_card(point)


class ZJetFitExclusionTests(unittest.TestCase):
    def test_exact_approved_zjet_fit_exclusion_preserves_original_arrays(self):
        key = ('GCR_highdm_c1_2023BPix','ZJet')
        variance = [0.] * N_HIGH_CR; variance[len(FIT_GROUPS)+2] = .010207432109671547
        row = dict(family='ZJet', parameter='', initial=1., variations={}, extra_lnN=(),
                   nominal=np.zeros(N_HIGH_CR), sumw2=np.array(variance))
        issue = dict(channel=key[0],process=key[1],parameter='',
            problem='zero_integral_nonempty_component',sumw2=variance[len(FIT_GROUPS)+2],variations=[])
        data = {key:row, ('GCR_highdm_c1_2024','ZJet'):dict(row,nominal=np.ones(N_HIGH_CR))}
        retained, remaining, audit = builder.approved_zjet_fit_background(data,[issue],builder.ZJET_FIT_EXCLUSION)
        self.assertEqual(set(retained), {('GCR_highdm_c1_2024','ZJet')})
        self.assertIs(retained['GCR_highdm_c1_2024','ZJet'],data['GCR_highdm_c1_2024','ZJet'])
        self.assertEqual(remaining,[])
        np.testing.assert_array_equal(data[key]['sumw2'],variance)
        self.assertEqual(audit[0]['omitted_sumw2_by_bin'],variance)
        self.assertEqual(audit[0]['auto_mc_stats'],[10,1,1])
        for field, value in [('nominal',np.ones(N_HIGH_CR)),('sumw2',np.zeros(N_HIGH_CR)),('parameter','new'),
                             ('initial',2.),('variations',{'new':{}}),('extra_lnN',[('new',1.1,1.1)]),
                             ('family','Top')]:
            bad = copy.deepcopy(data);bad[key][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                builder.approved_zjet_fit_background(bad,[issue],builder.ZJET_FIT_EXCLUSION)
        from_label, _, _ = builder.approved_zjet_fit_background(
            data,[issue],dict(builder.ZJET_FIT_EXCLUSION,user_approved=False))
        self.assertEqual(set(from_label),set(retained))

    def test_approved_zjet_accepts_only_complete_exactly_zero_flat_endpoints(self):
        from TROTASR.utils.statistical_templates import json_ready
        key = ('GCR_highdm_c1_2023BPix', 'ZJet')
        variance = np.zeros(N_HIGH_CR); variance[len(FIT_GROUPS)+2] = .010207432109671547
        variations = {name: dict(up=np.zeros(N_HIGH_CR), down=[0.] * N_HIGH_CR) for name in (
            'CMS_scale_j_Total_2023BPix', 'CMS_btag_fixedWP_bc_correlated',
            'CMS_NPS26012_scale_met_unclustered_energy_13p6TeV')}
        row = dict(family='ZJet', parameter='', initial=1., variations=variations,
            extra_lnN=(), nominal=np.zeros(N_HIGH_CR), sumw2=variance)
        other = ('GCR_highdm_c1_2024', 'ZJet')
        data = {key: row, other: dict(row, nominal=np.ones(N_HIGH_CR))}
        issue = dict(channel=key[0], process=key[1], parameter='',
            problem='zero_integral_nonempty_component', sumw2=float(variance[len(FIT_GROUPS)+2]), variations=[])
        other_issue = dict(issue, channel=other[0])
        before = json_ready(copy.deepcopy(data))
        issues = [issue, other_issue]; issues_before = copy.deepcopy(issues)
        retained, remaining, audit = builder.approved_zjet_fit_background(
            data, issues, builder.ZJET_FIT_EXCLUSION)
        self.assertEqual(set(retained), {other})
        self.assertIs(retained[other], data[other])
        self.assertEqual(remaining, [other_issue])
        self.assertEqual(json_ready(data), before)
        self.assertEqual(issues, issues_before)
        self.assertEqual(audit[0]['omitted_variations'], json_ready(variations))
        self.assertEqual(audit[0]['omitted_variation_names'], sorted(variations))
        self.assertEqual(audit[0]['omitted_variation_count'], 3)
        self.assertEqual(audit[0]['omitted_endpoint_count'], 6)
        self.assertTrue(audit[0]['root_objects_preserved'])
        self.assertEqual(audit[0]['omitted_sumw2_by_bin'], variance.tolist())
        # The exclusion is fit-only: even the omitted source endpoints remain intact.
        self.assertIs(data[key]['variations'], variations)
        audit[0]['omitted_variations']['CMS_scale_j_Total_2023BPix']['up'][0] = 1.
        np.testing.assert_array_equal(variations['CMS_scale_j_Total_2023BPix']['up'], np.zeros(N_HIGH_CR))
        invalid_pairs = [
            {}, {'up': np.zeros(N_HIGH_CR)}, {'down': np.zeros(N_HIGH_CR)},
            {'up': np.zeros(N_HIGH_CR), 'down': np.zeros(N_HIGH_CR), 'extra': np.zeros(N_HIGH_CR)},
            {'up': np.zeros(N_HIGH_CR - 1), 'down': np.zeros(N_HIGH_CR)},
            {'up': np.zeros((N_HIGH_CR, 1)), 'down': np.zeros(N_HIGH_CR)},
            {'up': np.zeros(N_HIGH_CR), 'down': None},
            {'up': ['0'] * N_HIGH_CR, 'down': np.zeros(N_HIGH_CR)},
            {'up': np.array([0.] * N_HIGH_CR, dtype=object), 'down': np.zeros(N_HIGH_CR)},
        ]
        for value in (1e-300, -1e-300, float('nan'), float('inf'), -float('inf')):
            for direction in ('up', 'down'):
                pair = dict(up=np.zeros(N_HIGH_CR), down=np.zeros(N_HIGH_CR))
                pair[direction][len(FIT_GROUPS)+2] = value
                invalid_pairs.append(pair)
        for index, pair in enumerate(invalid_pairs):
            bad = copy.deepcopy(data); bad[key]['variations'] = {'changed': pair}
            with self.subTest(pair=index), self.assertRaises(ValueError):
                builder.approved_zjet_fit_background(bad, issues, builder.ZJET_FIT_EXCLUSION)
        for invalid in (None, [], {'': dict(up=np.zeros(N_HIGH_CR), down=np.zeros(N_HIGH_CR))}):
            bad = copy.deepcopy(data); bad[key]['variations'] = invalid
            with self.subTest(variations=repr(invalid)), self.assertRaises(ValueError):
                builder.approved_zjet_fit_background(bad, issues, builder.ZJET_FIT_EXCLUSION)
        for field, value in (('nominal', np.full(12, 1e-300)), ('sumw2', variance * 2),
                             ('parameter', 'new'), ('initial', 2.), ('family', 'Top')):
            bad = copy.deepcopy(data); bad[key][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                builder.approved_zjet_fit_background(bad, issues, builder.ZJET_FIT_EXCLUSION)
        changed_scope = copy.deepcopy(builder.ZJET_FIT_EXCLUSION)
        changed_scope['targets'][0]['channel'] = other[0]
        with self.assertRaises(ValueError):
            builder.approved_zjet_fit_background(data, issues, changed_scope)
        bad_issue = dict(issue, variations=['CMS_scale_j_Total_2023BPix'])
        with self.assertRaises(ValueError):
            builder.approved_zjet_fit_background(data, [bad_issue], builder.ZJET_FIT_EXCLUSION)



class EraMergeTests(unittest.TestCase):
    def test_delivered_grid_is_reused_without_opening_inputs(self):
        manifest = {'era_merged': True, 'status': 'cards_ready', 'sr_data_blinded': True}
        path = ROOT/'stats/run3/grid/manifest.json'
        with patch.object(era_merge, 'read_json', return_value=manifest) as read, \
             patch.object(era_merge, 'prepare_merged_cards', side_effect=AssertionError('prepare again')), \
             patch.object(era_merge, 'write_json', side_effect=AssertionError('write accepted receipt')):
            self.assertIs(era_merge.merge_accepted_grid(path), manifest)
        read.assert_called_once_with(path)
        for invalid in (dict(manifest,era_merged=False), dict(manifest,sr_data_blinded=False)):
            with patch.object(era_merge,'read_json',return_value=invalid):
                with self.assertRaisesRegex(ValueError,'blinded era-merged'):
                    era_merge.merge_accepted_grid(path)

    def native_fixture(self):
        channels, background, columns, processes, rates = {}, {}, [], [], []
        for era in ERAS:
            for name,n in era_merge.CHANNEL_BINS.items():
                channel = name+'_'+era
                sr = channel.startswith('SR_')
                channels[channel] = [dict(name=f'{name}_bin{i}', observation=None if sr else 4.) for i in range(n)]
                background[channel,'Top'] = dict(family='Top', nominal=[2.] * n, initial=1.5)
                if sr:
                    background[channel,'signal'] = dict(family='signal', nominal=[999.] * n, initial=1.)
                n_rates = {'LLCR_highdm_c1': 4, 'QCDCR_highdm_c1': 4, 'GCR_highdm_c1': 8}.get(name, 6)
                for i in range(n_rates):
                    process = f'background{i}'
                    parameter = f'CMS_NPS26012_ll_norm_{name}_{i}'
                    if name == 'SR_highdm_c1' and i in (4, 5):
                        nb = 'Nb1' if i == 4 else 'Nb2plus'
                        process = f'ZJet_{nb}_u0'
                        parameter = f'CMS_NPS26012_sgamma_shape_highdm_{nb}_bin1'
                        background[channel, process] = dict(family='ZJet', nominal=[.5] * n, initial=1.)
                    if name == 'GCR_highdm_c1' and i in (6, 7):
                        nb = 'Nb1' if i == 6 else 'Nb2plus'
                        process = f'PhotonJet_{nb}_u0'
                        parameter = f'CMS_NPS26012_sgamma_shape_highdm_{nb}_bin1'
                    columns.append(channel)
                    processes.append(process)
                    rates.append(f'{parameter}_{era} rateParam {channel} {process}'
                                 + (' 2' if era=='2022EE' else ' 1')+' [0.01,5]')
        text = '\n'.join([
            'shapes * * ../templates/template.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC',
            'bin '+' '.join(channels), 'observation '+' '.join(['-1']*len(channels)),
            'bin '+' '.join(columns), 'process '+' '.join(processes),
            'process '+' '.join(str(i%6+1) for i in range(len(processes))),
            'rate '+' '.join(['-1']*len(columns)), *rates, '* autoMCStats 10 1 1'])+'\n'
        return channels, background, text

    def test_publishes_directly_from_memory_and_reuses_shared_observations(self):
        channels, background, text = self.native_fixture()
        point = dict(model='T2tt', mass='mStop1200_mLSP500')
        writes = []
        def write_observation(path,sums,labels):
            self.assertEqual(sums['SR_highdm_c1']['values'],[24.] * 118)
            self.assertEqual(sums['LLCR_highdm_c1']['values'],[24.] * N_HIGH_CR)
            self.assertAlmostEqual(sums['SR_highdm_c1']['variances'][0],24.)
            self.assertEqual(sums['LLCR_highdm_c1']['variances'][0],24.)
            self.assertEqual(labels['SR_lowdm_c1'][0],'SR_lowdm_c1_bin0')
            writes.append(path)
            path.write_bytes(b'synthetic-observation')
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            grid = Path(folder)
            path = grid/'datacards/datacard_T2tt_mStop1200_mLSP500.txt'
            (grid/'templates').mkdir()
            (grid/'templates/template.root').write_bytes(b'synthetic-template')
            meta = dict(fit_ready=True,sr_data_blinded=True,inputs=[dict(path='templates',sha256='a'*64)])
            with patch.object(era_merge,'write_observation_sum',side_effect=write_observation):
                merge = era_merge.prepare_merged_cards(path,text,meta,channels,background)
                self.assertEqual(era_merge.prepare_merged_cards(path,text,meta,channels,background),merge)
                self.assertEqual(len(writes),1)
                self.assertFalse(path.exists(), 'Never wrote the per-era source cards')
                self.assertEqual(
                    era_merge.prepare_merged_cards(path,text+'# changed source\n',meta,channels,background),
                    merge)
                self.assertEqual(len(writes),1)
            with patch.object(builder,'_CARD_CONTEXT',dict(era_merge=merge)), \
                 patch.object(builder,'render_point_card',return_value=(path,text,meta)):
                result = builder.card_worker(point)
            self.assertTrue(result['era_merged'])
            card = path.read_text()
            self.assertTrue(card.startswith('imax 8\n'))
            self.assertIn('shapes background0_2022EE SR_highdm_c1 ../templates/template.root',card)
            self.assertIn('rateParam SR_highdm_c1 background0_2022EE (@0*2) CMS_NPS26012_ll_norm_SR_highdm_c1_0',card)
            self.assertIn('rateParam SR_highdm_c1 background0_2022 1 [0.01,2.5]',card)
            side = read_json(path.with_suffix('.manifest.json'))
            self.assertEqual(side['dependent_rateparam_formulas'], N_RATES + 2 * len(ERAS))
            self.assertEqual(side['rate_parameters'],N_RATES)
            self.assertNotIn('source_card',side)
            self.assertFalse((grid/'history').exists())
            with patch.object(builder,'point_card_metadata',return_value=meta), \
                 patch.object(builder,'render_point_card',side_effect=AssertionError('completed merged card rerendered')):
                self.assertEqual(builder.existing_card(dict(output=grid,point_issues={},era_merge=merge),point),result)

    def test_cr_only_removes_sr_shapes_and_columns_and_preserves_cr_parameter(self):
        channels = list(era_merge.CHANNEL_BINS)
        controls = [c for c in channels if not c.startswith('SR_')]
        columns = [channels[0], *channels]
        processes = ['signal', *('Top' for _ in channels)]
        normalization = f'alpha rateParam {controls[0]} Top 1 [0.01,5]'
        rows = ['bin '+' '.join(channels), 'bin '+' '.join(columns),
                'process '+' '.join(processes), 'process '+' '.join(['0']+['1']*len(channels)),
                'rate '+' '.join(['-1']*len(columns)), '* autoMCStats 10 1 1',
                normalization,
                f'eraScale rateParam {controls[1]} Top (@0) alpha']
        rows += [f'shapes * {c} template.root $CHANNEL/$PROCESS $CHANNEL/$SYSTEMATIC' for c in channels]
        cr = era_merge.cr_only_card({'lines': rows})
        self.assertIn('imax 6', cr)
        self.assertNotIn('SR_', cr)
        self.assertNotIn('signal', cr)
        self.assertIn('observation -1 -1 -1 -1 -1 -1', cr)
        self.assertIn(normalization, cr)
        self.assertIn(f'eraScale rateParam {controls[1]} Top (@0) alpha', cr)
        with self.assertRaisesRegex(ValueError, 'lost a shared normalization declaration'):
            era_merge.cr_only_card({'lines': [r for r in rows if not r.startswith('alpha ')]})


class SignalReplacementCardTests(unittest.TestCase):
    def test_corrected_merge_contract_and_runs_are_required_before_projection(self):
        from TROTASR.gnn4lowdm.configuration import configuration_path
        from TROTASR.utils.io import write_json
        from TROTASR.workflows import signal_grid_inputs as signals
        year = '2025'
        point = dict(model='T2tt',mass='mStop1200_mLSP500')
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            merged = Path(temporary)/'systematics_2025.json.gz'
            normalization = Path(temporary)/'normalization_2025.json.gz'
            write_json(normalization,dict(signal_replacement=dict(files=1,datasets=1,
                luminosity_changed=False,background_factors_changed=False),
                signal_mass_points={point['mass']:dict(sumw_mass_point=100.,normalization_factor=.25)}))
            norm_sha = 'legacy-normalization'
            config_path = configuration_path(year)
            gnn_digest = 'legacy-gnn'
            gnn = read_json(config_path)
            sr_digest = 'd'*64
            physics = dict(year=year,shape_mode='cms_trota_jme',variations=['nominal'],
                nominal_calibrations_only=True, signal_sr_only=True, normalization_sha256=norm_sha,
                sr_binning_sha256=sr_digest,configuration_sha256=gnn_digest)
            scope = 'full_cms_trota_jme_production'
            coverage = dict(full_input_inventory=True,expected_files=1,completed_files=1,
                            failed_files=0,expected_events=10,processed_events=10)
            replacement = dict(files=1,datasets=1,retained_events=10,signal_uncertainties=False,
                background_and_data_unchanged=True,central='cms_trota_jme_nominal',
                signal_input='regenerated_signal_uuid')
            payload = dict(status='complete',scope=scope, sr_data_blinded=True,
                highdm_sr_source_axes='exact_nb_exact_topology_native_recoil_v2',
                contract=physics, coverage=coverage, signal_replacement=replacement,
                input_metadata={'1':dict(is_signal=True,is_data=False,
                    signal_sumw_by_genmodel={'GenModel_T2tt_1200_500':100.})},
                histograms={mode:dict(nominal={'SR':{}}) for mode in ('highdm','lowdm')},
                physical_histograms={})
            def refresh_fixture():
                write_json(merged,payload)
                return dict(status='complete',scope=scope,output=str(merged.relative_to(ROOT)),
                    coverage=coverage,
                    physics_contract=physics,signal_replacement=replacement,normalization=dict(
                        path=str(normalization.relative_to(ROOT)),sha256=norm_sha,uuid_runs_sha256='e'*64))
            def project(receipt):
                return signals.extract_replacement_year(merged,year,[point],{'year':year},gnn,
                    production_receipt=receipt,accepted_contract=dict(
                        gnn_sha256_by_era={year:gnn_digest},sr_binning_sha256=sr_digest),
                    normalization_path=normalization)
            with patch.object(signals,'project_signals',return_value=({}, {}, {})) as projector:
                receipt = refresh_fixture()
                self.assertEqual(project(receipt)['source']['normalization'],str(normalization.relative_to(ROOT)))
                self.assertTrue(projector.call_args.kwargs['nominal_only'])
                for flag in ('nominal_calibrations_only', 'signal_sr_only'):
                    payload['contract'] = dict(physics, **{flag: False})
                    with self.assertRaisesRegex(ValueError, 'merged signal input'):
                        project(refresh_fixture())
                payload['contract'] = physics
                refresh_fixture()
                old_record = dict(physics, regions=['SR', 'LLCR'], execution={'different': True})
                self.assertEqual(project(dict(receipt, physics_contract=old_record)), project(receipt))
                payload['histograms']['lowdm']['nominal']['LLCR'] = {}
                with self.assertRaisesRegex(ValueError,'non-SR region'):
                    project(refresh_fixture())
                del payload['histograms']['lowdm']['nominal']['LLCR']
                payload['histograms']['highdm']['jesTotalUp'] = {}
                with self.assertRaisesRegex(ValueError,'non-nominal endpoint'):
                    project(refresh_fixture())
                del payload['histograms']['highdm']['jesTotalUp']
                payload['physical_histograms'] = {'highdm':{}}
                with self.assertRaisesRegex(ValueError,'unexpected auxiliary histograms'):
                    project(refresh_fixture())
                payload['physical_histograms'] = {}
                payload['input_metadata']['1']['signal_sumw_by_genmodel']['GenModel_T2tt_1200_500'] = 99.
                with self.assertRaisesRegex(ValueError,'Runs mismatch'):
                    project(refresh_fixture())
                payload['histograms']['lowdm']['nominal']['SR']['blind'] = {'data_obs':{}}
                with self.assertRaisesRegex(ValueError,'Observed SR histograms'):
                    project(refresh_fixture())

    def test_only_twelve_signal_alias_targets_change_and_small_receipt_is_atomic(self):
        from TROTASR.utils.statistical_templates import template_name
        from TROTASR.workflows.signal_grid_inputs import replacement_signal_object
        point = dict(model='T2tt',mass='mStop1200_mLSP500')
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            top = Path(temporary)
            source = top/'frozen/datacards/point.txt'
            target = top/'new_signal/grid/datacards/point.txt'
            templates = target.parent.parent/'templates'
            source.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)
            templates.mkdir()
            (source.parent/'background.root').write_bytes(b'synthetic background')
            shapes = ['shapes Top LLCR_highdm_c1 background.root LLCR/Top LLCR/Top_$SYSTEMATIC']
            extra = ['# frozen card', 'alpha rateParam LLCR_highdm_c1 Top 1 [0.01,5]', '* autoMCStats 10 1 1']
            cols = []
            for mode in ('highdm','lowdm'):
                channel = f'SR_{mode}_c1'
                for era in ERAS:
                    native = f'{channel}_{era}'
                    process = f'signal_{era}'
                    root = template_name(native)
                    (templates/root).write_bytes(b'synthetic replacement')
                    obj = replacement_signal_object(native,point)
                    shapes.append(f'shapes {process} {channel} original.root {obj} {obj}_$SYSTEMATIC')
                    cols.append((channel,process))
            body = [*extra, 'bin '+' '.join(c for c,_ in cols)+' LLCR_highdm_c1',
                    'process '+' '.join(p for _,p in cols)+' Top',
                    'process '+' '.join(map(str,range(0,-12,-1)))+' 1']
            text = '\n'.join(shapes+body)+'\n'
            displaced = builder.rewrite_replacement_signal_shapes(text,source,target,point,templates)
            actual = displaced.splitlines()
            self.assertEqual(actual[len(shapes):],body)
            self.assertEqual((target.parent/actual[0].split()[3]).resolve(),source.parent/'background.root')
            for line in actual[1:13]:
                shape = line.split()
                root = templates/template_name(f'{shape[2]}_{shape[1].split("_",1)[1]}')
                self.assertEqual((target.parent/shape[3]).resolve(),root)
            with self.assertRaisesRegex(ValueError,'twelve SR signal shape aliases'):
                builder.rewrite_replacement_signal_shapes('\n'.join(shapes[:-1]+body)+'\n',source,target,point,templates)
            meta = dict(signal_input_revision='test-generation',sr_data_blinded=True)
            builder.write_replacement_card(target,displaced,meta)
            builder.write_replacement_card(target,displaced,meta)
            self.assertEqual(read_json(target.with_suffix('.manifest.json')),meta)
            self.assertFalse(source.exists())
            builder.write_replacement_card(target,displaced+'# comment only',meta)
            self.assertEqual(target.read_text(), displaced)
            with self.assertRaises(FileExistsError):
                builder.write_replacement_card(target,displaced.replace('[0.01,5]', '[0.01,6]'),meta)


if __name__=='__main__':
    unittest.main()
