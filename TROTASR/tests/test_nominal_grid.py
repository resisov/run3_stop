"""Grid-boundary regressions: full bin layout, never synthetic physics results."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from TROTASR.tests.test_statistics_inputs import fixture
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.paths import ROOT
from TROTASR.utils.signal_models import signal_mass_key
from TROTASR.utils.statistics_inputs import build_inputs
from TROTASR.utils.statistical_templates import assemble, load_theory, profiled_theory
from TROTASR.workflows.signal_grid_inputs import (MODELS, FIELDS, project_signals, signal_data,
    check_reference, extract_year, requested_grid, represented_signals, require_fit_support_issues)
from TROTASR.workflows.build_nominal_grid import (build, point_card, combine_cached, point_signals,
    fit_background, verify_signal_objects)
from TROTASR.utils.sr_merge import load_merge, bin_groups, project
from TROTASR.utils.statistical_templates import template_name
from TROTASR.workflows.combine_limits import IntegrityCache
from TROTASR.workflows import build_nominal_grid as builder


MASS = 'mStop1200_mLSP500'


def grid_fixture():
    return dict(models={model: [MASS] for model in MODELS},
                expected_points_by_model={model: 1 for model in MODELS})


def year_fixture(year):
    values = fixture(year)
    payload = values[0]
    for mode in ('highdm', 'lowdm'):
        for samples in payload['histograms'][mode]['nominal']['SR'].values():
            for model in ('T2tb', 'T2bW'):
                samples[signal_mass_key(model, 1200, 500)] = copy.deepcopy(samples[MASS])
    payload['input_metadata']['signals'] = dict(is_signal=True,
        signal_sumw_by_genmodel={'GenModel_' + model + '_1200_500': 100. for model in MODELS})
    payload['contract'].update(sr_binning_sha256=sha256(ROOT / 'highdm_sr_binning.json'),
        configuration_sha256=sha256(ROOT / 'gnn4lowdm/config.json'))
    return values


def native_cr_fixture():
    groups = {}
    for region in ('LLCR', 'QCDCR', 'GCR'):
        rows = []
        for nb in ('Nb1', 'Nb2plus'):
            for i in range(8):
                rows.append(dict(name=f'{region}_highdm_{nb}_bin{i}', region=region,
                    regime='highdm', nb_group=nb, control_group=nb, observation=i+1,
                    backgrounds={'Top': dict(nominal=[i-4], sumw2=[i+2],
                        variations={'test': dict(up=[i+2], down=[i-5])})},
                    rate_params={'Top':f'CMS_NPS26012_ll_norm_highdm_{nb}_bin{i}_2024'},
                    rate_initial={'Top':1.}, extra_lnN={}, physical={}))
        groups[region+'_highdm'] = rows
    return dict(year='2024', groups=groups)


class CanonicalCR12Tests(unittest.TestCase):
    def test_exact_opt_in_and_parameter_mapping(self):
        self.assertFalse(builder.require_statistical_revision(None))
        self.assertTrue(builder.require_statistical_revision(builder.CR12_POLICY))
        for field, value in (('rate_parameter_range', [0,10]), ('signal_contamination', True),
                             ('preserve_sr_bins', [73,30]), ('auto_mc_stats', [10,0,1])):
            bad = dict(builder.CR12_POLICY, **{field:value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                builder.require_statistical_revision(bad)
        for kind in ('ll','qcd'):
            for nb in ('Nb1','Nb2plus'):
                for year in ('2024','2025'):
                    for i,j in enumerate((0,1,2,3,4,4,5,5)):
                        self.assertEqual(builder.cr12_parameter(f'CMS_NPS26012_{kind}_norm_highdm_{nb}_bin{i}_{year}'),
                            f'CMS_NPS26012_{kind}_norm_highdm_{nb}_bin{j}_{year}')
        for name in ('', 'CMS_NPS26012_ll_norm_lowdm_Nb1_inclusive_2024',
                     'CMS_NPS26012_sgamma_shape_highdm_Nb1_bin4_2025'):
            self.assertEqual(builder.cr12_parameter(name), name)

    def test_signed_cr_sum_variance_observation_and_zero_endpoints(self):
        old = native_cr_fixture()
        for rows in old['groups'].values():
            rows[4]['backgrounds']['Top']['nominal'] = [-2.]
            rows[5]['backgrounds']['Top']['nominal'] = [3.]
            for i in (4,5): rows[i]['backgrounds']['Top']['variations']['test']['up'] = [0.]
        preserved = copy.deepcopy(old)
        grouped = builder.cr12_payload(old)
        self.assertEqual(old, preserved)
        for region, rows in grouped['groups'].items():
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[4]['observation'], 11)
            self.assertEqual(rows[4]['backgrounds']['Top']['nominal'], [1.])
            self.assertEqual(rows[4]['backgrounds']['Top']['sumw2'], [13.])
            self.assertEqual(rows[4]['backgrounds']['Top']['variations']['test']['up'], [0.])
        _, data, _, _ = assemble([grouped], preserve_flat=True)
        self.assertEqual(data['LLCR_highdm_c1_2024','Top_Nb1_u4']['nominal'][4], 1.)

    def test_cr12_rejects_incomplete_or_mixed_bindings(self):
        for problem in ('missing', 'initial', 'extra'):
            p = native_cr_fixture()
            if problem == 'missing': p['groups']['LLCR_highdm'].pop()
            if problem == 'initial': p['groups']['LLCR_highdm'][5]['rate_initial']['Top'] = 2.
            if problem == 'extra': p['groups']['LLCR_highdm'][5]['extra_lnN']['Top'] = [dict(name='extra',down=.9,up=1.1)]
            with self.subTest(problem=problem), self.assertRaises(ValueError): builder.cr12_payload(p)

    def test_card_default_identical_and_explicit_range(self):
        from TROTASR.utils.statistical_templates import card_text
        channels,data,_,_ = assemble([builder.cr12_payload(native_cr_fixture())], preserve_flat=True)
        self.assertEqual(builder.card_text(channels,data), card_text(channels,data))
        text = builder.card_text(channels,data,statistical_revision=builder.CR12_POLICY)
        params = [l.split() for l in text.splitlines() if ' rateParam ' in l]
        self.assertTrue(params)
        self.assertTrue(all(p[-1] == '[0.01,5]' for p in params))
        self.assertIn('* autoMCStats 10 1 1', text)
        next(iter(data.values()))['initial'] = 6.
        with self.assertRaises(ValueError): builder.card_text(channels,data,statistical_revision=builder.CR12_POLICY)

    def test_cr12_reconciles_only_audited_native_cr_only_fix(self):
        old = native_cr_fixture()
        fixed = old['groups']['LLCR_highdm'][5]
        par = fixed['rate_params'].pop('Top')
        initial = fixed['rate_initial'].pop('Top')
        with self.assertRaises(ValueError): builder.cr12_payload(old)
        old['cr_only_unity_audit'] = [dict(channel=fixed['name'], process='Top',
            parameter=par, previous_initial=initial, fixed=1.)]
        preserved = copy.deepcopy(old)
        row = builder.cr12_payload(old)['groups']['LLCR_highdm'][4]
        self.assertEqual(old, preserved)
        self.assertEqual(row['rate_params']['Top'], builder.cr12_parameter(par))
        self.assertEqual(row['backgrounds']['Top']['nominal'], [1.])
        self.assertEqual(row['backgrounds']['Top']['sumw2'], [13.])
        self.assertEqual(len(row['cr12_binding_reconciliation']), 1)
        other = old['groups']['LLCR_highdm'][4]
        old['cr_only_unity_audit'].append(dict(channel=other['name'], process='Top',
            parameter=other['rate_params'].pop('Top'), previous_initial=other['rate_initial'].pop('Top'), fixed=1.))
        self.assertFalse(builder.cr12_payload(old)['groups']['LLCR_highdm'][4]['rate_params'])
        bad = copy.deepcopy(preserved)
        bad['cr_only_unity_audit'][0]['parameter'] = par.replace('Nb1', 'Nb2plus')
        with self.assertRaises(ValueError): builder.cr12_payload(bad)

    def test_cr12_full_fixture_preserves_sr_low_and_signal_arrays(self):
        caches = {}
        for year in (2024,2025):
            values = year_fixture(year)
            model = build_inputs(*values,'T2tt',1200,500)
            signals,source,_ = project_signals(values[0],model,grid_fixture(),read_json(ROOT/'gnn4lowdm/config.json'))
            for rows in model['groups'].values():
                for row in rows: row['physical'].pop('signal',None)
            caches[str(year)] = dict(background=model,signals=signals,source_bins=source)
        merge = dict(path='sr_category_merge.json',sha256=sha256(ROOT/'sr_category_merge.json'))
        old, old_data, _, old_sources = combine_cached(caches,merge)
        new, new_data, _, new_sources = combine_cached(caches,merge,statistical_revision=builder.CR12_POLICY)
        self.assertEqual(sum(map(len,new.values())),428)
        self.assertEqual(len(new),16)
        for key, record in old_data.items():
            if key[0].startswith('SR_') or '_lowdm_' in key[0]:
                for field in ('nominal','sumw2'): np.testing.assert_array_equal(new_data[key][field],record[field])
                for name,pair in record['variations'].items():
                    for direction,a in pair.items(): np.testing.assert_array_equal(new_data[key]['variations'][name][direction],a)
        before,_,_ = point_signals(old,old_sources,caches,'T2tt',MASS)
        after,_,_ = point_signals(new,new_sources,caches,'T2tt',MASS)
        check_reference(before,after)
        self.assertTrue(all(c.startswith('SR_') for c,p in after))

    def combined_exclusion_fixture(self):
        from TROTASR.utils.corrections import NONJME_ENDPOINTS
        from TROTASR.utils.statistics_model import nps_nuisance_name
        data, issues, policy = NominalGridTests().exclusion_fixture()
        for year in ('2024','2025'):
            key = 'SR_highdm_c1_'+year,'QCD_Nb1_u6'
            for e in NONJME_ENDPOINTS:
                if not e.endswith('Up'): continue
                n = nps_nuisance_name(e[:-2],year)
                data[key]['variations'][n] = dict(up=np.array([.125]),down=np.array([0.]))
                issues.append(dict(channel=key[0],process=key[1],problem='zero_integral_variation',nuisance=n,direction='down'))
        return data,issues,policy

    def test_nonjme_exclusion_keeps_all_arrays_and_old_subset_guard(self):
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        data, issues, policy = self.combined_exclusion_fixture(); saved = copy.deepcopy(data)
        proof = builder.original_exclusion_audit(data,issues,policy,COMBINATION_POLICY)
        self.assertEqual(len(proof),2)
        self.assertTrue(all(r['original_jme_subset_verified'] for r in proof))
        self.assertTrue(all(len(r['additional_nonjme_endpoints_preserved']) == 8 for r in proof))
        for key, record in data.items():
            for name,pair in record['variations'].items():
                for d,a in pair.items(): np.testing.assert_array_equal(a,saved[key]['variations'][name][d])
        for year in ('2024','2025'):
            old_key = 'QCDCR_highdm_c1_'+year,'QCD_Nb1_u6'
            data['QCDCR_highdm_c1_'+year,'QCD_Nb1_u5'] = data.pop(old_key)
        for record in data.values(): record['parameter'] = builder.cr12_parameter(record['parameter'])
        other = dict(channel='SR_lowdm_c1_2024',process='other',problem='zero_integral_variation')
        fit, remaining, audit = builder.cr12_fit_background(data,issues+[other],proof)
        self.assertEqual(remaining,[other]); self.assertEqual(len(data)-len(fit),2)
        self.assertTrue(all(r['parameter'].endswith('bin5_'+r['channel'][-4:]) for r in audit))
        self.assertTrue(all(fit[k] is data[k] for k in fit))

    def test_combined_exclusion_cannot_hide_changed_jme_or_malformed_nonjme(self):
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        for problem in ('jes','weight','nominal','nan','negative','issue','policy'):
            data,issues,policy = self.combined_exclusion_fixture(); combination = dict(COMBINATION_POLICY)
            r = data['SR_highdm_c1_2024','QCD_Nb1_u6']
            nonjme = next(n for n in r['variations'] if n.startswith('CMS_scale_e_'))
            if problem == 'jes': r['variations']['CMS_scale_j_Total_2024']['up'][0] = float('nan')
            if problem == 'weight': r['variations']['flat']['up'][0] = .1
            if problem == 'nominal': r['nominal'][0] = 1e-12
            if problem == 'nan': r['variations'][nonjme]['up'][0] = float('nan')
            if problem == 'negative': r['variations'][nonjme]['up'][0] = -.1
            if problem == 'issue': issues[1]['direction'] = 'up'
            if problem == 'policy': combination['reanchor'] = True
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                builder.original_exclusion_audit(data,issues,policy,combination)

    def test_combined_exclusion_accepts_updated_finite_jes_yield(self):
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        data, issues, policy = self.combined_exclusion_fixture()
        data['SR_highdm_c1_2025','QCD_Nb1_u6']['variations']['CMS_scale_j_Total_2025']['up'][0] = .037
        proof = builder.original_exclusion_audit(data, issues, policy, COMBINATION_POLICY)
        self.assertEqual(len(proof), 2)

    def test_absent_sparse_nonjme_stays_absent(self):
        from TROTASR.workflows.merge_nominal import COMBINATION_POLICY
        data,issues,policy = NominalGridTests().exclusion_fixture()
        proof = builder.original_exclusion_audit(data,issues,policy,COMBINATION_POLICY)
        self.assertTrue(all(not r['additional_nonjme_endpoints_preserved'] for r in proof))
        self.assertTrue(all(len(r['absent_nonjme_nuisances']) == 8 for r in proof))
        self.assertTrue(all(set(data[r['channel'],r['process']]['variations']) ==
            {'CMS_scale_j_Total_'+r['channel'][-4:],'flat'} for r in proof))


class NominalGridTests(unittest.TestCase):
    def exclusion_fixture(self):
        data, issues = {}, []
        for y, up in (('2024', .05757514512913886), ('2025', .0253831485651911)):
            sr, cr, p = 'SR_highdm_c1_'+y, 'QCDCR_highdm_c1_'+y, 'QCD_Nb1_u6'
            n = 'CMS_scale_j_Total_'+y
            data[sr,p] = dict(nominal=np.array([0.]), sumw2=np.array([0.]), family='QCD',
                parameter='CMS_NPS26012_qcd_norm_highdm_Nb1_bin6_'+y, initial=1., extra_lnN=(),
                variations={n: dict(up=np.array([up]), down=np.array([0.])),
                            'flat': dict(up=np.array([0.]), down=np.array([0.]))})
            data[cr,p] = dict(data[sr,p], nominal=np.array([5.]), sumw2=np.array([1.]), variations={})
            issues += [dict(channel=sr, process=p, problem='zero_integral_nonempty_component'),
                dict(channel=sr, process=p, problem='zero_integral_variation', nuisance=n, direction='down')]
        policy = dict(id='sr_qcd_nb1_u6_20260928', user_approved=True,
            preserve_all_root_objects=True, auto_mc_stats=[10,1,1],
            targets=[dict(channel='SR_highdm_c1_'+y, process='QCD_Nb1_u6') for y in ('2024','2025')])
        return data, issues, copy.deepcopy(policy)

    def test_exclusion_exactly_two_sr_columns_preserves_root_arrays_and_cr(self):
        data, issues, policy = self.exclusion_fixture()
        before = copy.deepcopy(data)
        fit, remaining, audit = fit_background(data, issues, policy)
        self.assertFalse(remaining); self.assertEqual(len(audit), 2)
        self.assertEqual(set(data)-set(fit), {('SR_highdm_c1_'+y, 'QCD_Nb1_u6') for y in ('2024','2025')})
        for key in data:
            np.testing.assert_array_equal(data[key]['nominal'], before[key]['nominal'])
            np.testing.assert_array_equal(data[key]['sumw2'], before[key]['sumw2'])
            for n, pair in data[key]['variations'].items():
                for d, a in pair.items(): np.testing.assert_array_equal(a, before[key]['variations'][n][d])
        self.assertTrue(all(fit[k] is data[k] for k in fit))
        self.assertEqual(fit_background(data, issues)[0], data)
        from TROTASR.utils.statistical_templates import card_text
        channels = {c: [dict(name=c+'_bin0', region=c.split('_')[0], year=c[-4:])] for c,p in data}
        card = card_text(channels, fit)
        self.assertIn('* autoMCStats 10 1 1', card)
        self.assertEqual(sum('rateParam' in line for line in card.splitlines()), 2)
        self.assertNotIn('rateParam SR_', card)
        self.assertEqual(card_text(channels, data, cr_only=True), card_text(channels, fit, cr_only=True))

    def test_exclusion_rejects_scope_expansion_or_changed_support(self):
        for modification in ('target', 'approval', 'nominal', 'nan_up', 'negative_up', 'zero_up', 'other_shape', 'cr', 'issue'):
            data, issues, policy = self.exclusion_fixture()
            r = data['SR_highdm_c1_2024','QCD_Nb1_u6']
            if modification == 'target': policy['targets'][0]['channel'] = 'QCDCR_highdm_c1_2024'
            if modification == 'approval': policy['user_approved'] = False
            if modification == 'nominal': r['nominal'][0] = 1e-12
            if modification == 'nan_up': r['variations']['CMS_scale_j_Total_2024']['up'][0] = float('nan')
            if modification == 'negative_up': r['variations']['CMS_scale_j_Total_2024']['up'][0] = -.1
            if modification == 'zero_up': r['variations']['CMS_scale_j_Total_2024']['up'][0] = 0.
            if modification == 'other_shape': r['variations']['flat']['up'][0] = .1
            if modification == 'cr': data['QCDCR_highdm_c1_2024','QCD_Nb1_u6']['nominal'][0] = 0
            if modification == 'issue': issues[1]['direction'] = 'up'
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                fit_background(data, issues, policy)
        data, issues, policy = self.exclusion_fixture()
        extra = dict(channel='SR_lowdm_c1_2024', process='other', problem='zero_integral_variation')
        self.assertEqual(fit_background(data, issues+[extra], policy)[1], [extra])

    def test_exclusion_accepts_updated_sf_yield_and_records_actual_value(self):
        data, issues, policy = self.exclusion_fixture()
        key = ('SR_highdm_c1_2025', 'QCD_Nb1_u6')
        data[key]['variations']['CMS_scale_j_Total_2025']['up'][0] = .037
        fit, remaining, audit = fit_background(data, issues, policy)
        self.assertNotIn(key, fit)
        self.assertFalse(remaining)
        row = next(r for r in audit if r['channel'] == key[0])
        self.assertEqual(row['omitted_variations']['CMS_scale_j_Total_2025']['up'], .037)
        self.assertEqual(data[key]['variations']['CMS_scale_j_Total_2025']['up'][0], .037)

    def test_extracted_resume_requires_unchanged_inputs(self):
        old = dict(inputs={'2025':'input'}, code={'workflows/build_nominal_grid.py':'old',
                                                'workflows/signal_grid_inputs.py':'same'})
        current = copy.deepcopy(old)
        current['code']['workflows/build_nominal_grid.py'] = 'new'
        self.assertTrue(builder.same_extraction_contract(old, current))
        current['code']['workflows/signal_grid_inputs.py'] = 'changed'
        self.assertTrue(builder.same_extraction_contract(old, current))
        current = copy.deepcopy(old); current['inputs']['2025'] = 'changed'
        self.assertFalse(builder.same_extraction_contract(old, current))

    def test_zero_support_preservation_does_not_make_background_fit_ready(self):
        caches = {}
        for year in (2024, 2025):
            model = build_inputs(*year_fixture(year), 'T2tt', 1200, 500)
            injected = False
            for group, bins in model['groups'].items():
                for b in bins:
                    b['physical'].pop('signal', None)
                    if group != 'SR_highdm':
                        continue
                    for name, record in b['backgrounds'].items():
                        if name == 'QCD_Nb1_u0':
                            record.update(nominal=np.array([0.]), sumw2=np.array([0.]),
                                variations={'CMS_scale_j_Total_'+str(year):
                                    {'up': np.array([.1 if not injected else 0.]), 'down': np.array([0.])}})
                            injected = True
            self.assertTrue(injected)
            caches[str(year)] = dict(background=model, source_bins={})
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            combine_cached(caches)
        issues = []
        channels, data, _, _ = combine_cached(caches, fit_issues=issues)
        self.assertTrue(issues)
        self.assertTrue(all(i['channel'].startswith('SR_') for i in issues))
        self.assertTrue(any(i['problem'] == 'zero_integral_nonempty_component' for i in issues))
        for year in (2024, 2025):
            records = [r for (c, _), r in data.items() if c.startswith('SR_highdm')
                and c.endswith(str(year)) and r['family'] == 'QCD'
                and r['parameter'].endswith('Nb1_bin0_'+str(year))]
            self.assertEqual(len(records), 1)
            self.assertEqual(float(records[0]['nominal'].sum()), 0.)
            self.assertEqual(float(records[0]['variations']['CMS_scale_j_Total_'+str(year)]['up'].sum()), .1)
        self.assertEqual({template_name(c) for c in channels},
            {'template_'+m+'_'+y+'.root' for m in ('highdm','lowdm') for y in ('2024','2025')})

    def test_preservation_never_waives_invalid_input_or_parameter_issues(self):
        require_fit_support_issues([dict(problem='zero_integral_nonempty_component')])
        for issue in ('sr_only_parameter', 'data_without_background', 'nonfinite'):
            with self.assertRaisesRegex(ValueError, 'Not a zero-support'):
                require_fit_support_issues([dict(problem=issue)])

    def test_keep_empty_signal_keeps_zero_templates_and_fit_issues(self):
        values = year_fixture(2024)
        for mode in ('highdm', 'lowdm'):
            for samples in values[0]['histograms'][mode]['nominal']['SR'].values():
                del samples[MASS]
        model = build_inputs(*values, 'T2tt', 1200, 500)
        projected, source, _ = project_signals(values[0], model, grid_fixture(), read_json(ROOT/'gnn4lowdm/config.json'))
        channels, _, _, _ = assemble([model])
        actual, issues, _ = signal_data(channels, {('2024', k): v for k, v in source.items()},
            {'2024': dict(signals=projected['T2tt'])}, MASS, keep_empty=True)
        self.assertEqual(len(actual), 2)
        self.assertTrue(issues)
        self.assertTrue(all(float(r['nominal'].sum()) == 0 for r in actual.values()))

    def test_four_final_files_and_clip_before_adopted_sr118_projection(self):
        caches={}
        for year in (2024,2025):
            values=year_fixture(year)
            values[0]['histograms']['highdm']['nominal']['SR']['1,0,0,1,0'][MASS]['recoil']['sumw'][0]=-10.
            model=build_inputs(*values,'T2tt',1200,500)
            signals,source,_=project_signals(values[0],model,grid_fixture(),read_json(ROOT/'gnn4lowdm/config.json'))
            for bins in model['groups'].values():
                for b in bins:b['physical'].pop('signal',None)
            caches[str(year)]=dict(background=model,signals=signals,source_bins=source)
        raw,data,_,sources=combine_cached(caches)
        before,_,_=point_signals(raw,sources,caches,'T2tt',MASS)
        merge=dict(path='sr_category_merge.json',sha256=sha256(ROOT/'sr_category_merge.json'))
        channels,after_data,_,merged_sources=combine_cached(caches,merge)
        after,issues,_=point_signals(channels,merged_sources,caches,'T2tt',MASS)
        self.assertFalse(issues)
        self.assertEqual(sum(map(len,channels.values())),452)
        self.assertEqual({template_name(c) for c in channels},{'template_'+m+'_'+y+'.root' for m in ('highdm','lowdm') for y in ('2024','2025')})
        groups=bin_groups(load_merge(merge))
        for key,record in before.items():
            for field in ('nominal','sumw2'):
                expected=project(record[field],groups) if 'highdm' in key[0] else record[field]
                np.testing.assert_array_equal(after[key][field],expected)

    def test_all_shape_signal_endpoints_match_single_point_projection(self):
        values=year_fixture(2024)
        values[0]['contract']['shape_mode']='cms_trota_jme'
        for mode,h in values[0]['histograms'].items():
            for n in ('jesTotalUp','jesTotalDown','jerUp','jerDown'):
                h[n]=copy.deepcopy(h['nominal'])
        h=values[0]['histograms']['highdm']['jesTotalUp']['SR']
        h['1,0,1,0,0']=h.pop('1,0,0,1,0')
        model=build_inputs(*values,'T2tt',1200,500)
        signals,source,_=project_signals(values[0],model,grid_fixture(),read_json(ROOT/'gnn4lowdm/config.json'))
        channels,expected,_,issues=assemble([model],preserve_flat=True)
        self.assertFalse(issues)
        actual,issues,_=signal_data(channels,{('2024',k):v for k,v in source.items()},{'2024':dict(signals=signals['T2tt'])},MASS)
        self.assertFalse(issues)
        check_reference({k:v for k,v in expected.items() if k[1]=='signal'},actual)

    def test_all_three_topologies_match_full_single_point_path(self):
        values = year_fixture(2024)
        gnn = read_json(ROOT / 'gnn4lowdm/config.json')
        model_input = build_inputs(*values, 'T2tt', 1200, 500)
        projected, source_bins, audit = project_signals(values[0], model_input, grid_fixture(), gnn)
        for model in MODELS:
            point = build_inputs(*values, model, 1200, 500)
            channels, data, _, problems = assemble([point])
            self.assertFalse(problems)
            packed, issues, clipped = signal_data(channels,
                {('2024', name): sources for name, sources in source_bins.items()},
                {'2024': dict(signals=projected[model])}, MASS)
            self.assertFalse(issues); self.assertFalse(clipped)
            check_reference({k: v for k, v in data.items() if k[1] == 'signal'}, packed)
            self.assertEqual(sum(map(len, channels.values())), 270)
        self.assertEqual(audit['bins'], dict(highdm=162, lowdm=30))

    def test_signed_projection_then_clip_preserves_variance(self):
        values = year_fixture(2024)
        leaf = values[0]['histograms']['highdm']['nominal']['SR']['1,0,0,1,0'][MASS]['recoil']
        leaf['sumw'][0] = -10.
        model = build_inputs(*values, 'T2tt', 1200, 500)
        projected, source, _ = project_signals(values[0], model, grid_fixture(), read_json(ROOT / 'gnn4lowdm/config.json'))
        channels, expected, _, _ = assemble([model])
        actual, issues, clipped = signal_data(channels, {('2024', k): v for k, v in source.items()},
            {'2024': dict(signals=projected['T2tt'])}, MASS)
        self.assertFalse(issues)
        self.assertTrue(clipped)
        self.assertEqual(clipped[0]['before'], -5.)  # -10 + 2 + 3 in the pooled-Nb topology
        check_reference({k: v for k, v in expected.items() if k[1] == 'signal'}, actual)
        h = next(v for (c, p), v in actual.items() if 'highdm' in c)
        self.assertEqual(h['sumw2'][0], 14.)

    def test_runs_inventory_distinguishes_zero_acceptance(self):
        values = year_fixture(2024)
        represented = represented_signals(values[0])
        for mode in ('highdm', 'lowdm'):
            for samples in values[0]['histograms'][mode]['nominal']['SR'].values():
                del samples[MASS]
        self.assertIn('T2tt/' + MASS, represented)
        model = build_inputs(*values, 'T2tt', 1200, 500)
        projected, source, _ = project_signals(values[0], model, grid_fixture(), read_json(ROOT / 'gnn4lowdm/config.json'))
        channels, _, _, _ = assemble([model])
        actual, issues, _ = signal_data(channels, {('2024', k): v for k, v in source.items()},
            {'2024': dict(signals=projected['T2tt'])}, MASS)
        self.assertFalse(actual)
        self.assertEqual(issues, [dict(problem='zero_signal_acceptance')])

    def test_unpaired_shape_campaign_is_not_silently_included(self):
        values = year_fixture(2024)
        model = build_inputs(*values, 'T2tt', 1200, 500)
        values[0]['histograms']['highdm']['jesTotalUp'] = {}
        with self.assertRaisesRegex(ValueError, 'Unpaired systematic endpoints'):
            project_signals(values[0], model, grid_fixture(), read_json(ROOT / 'gnn4lowdm/config.json'))

    def test_original_requested_grid_includes_old_failed_points_not_old_limits(self):
        grid = requested_grid(ROOT / 'stats/signal_grid.json')
        self.assertEqual(grid['expected_points_by_model'], dict(T2tt=205, T2bW=354, T2tb=365))
        self.assertIn('mStop1450_mLSP1300', grid['models']['T2tt'])
        self.assertIn('mStop200_mLSP1', grid['models']['T2tb'])
        self.assertNotIn('expected', (ROOT / 'stats/signal_grid.json').read_text().replace('expected_points_by_model', '').replace('expected_limits.json', ''))

    def test_named_signal_binding_and_signal_only_theory(self):
        payloads = [build_inputs(*year_fixture(y), 'T2tb', 1200, 500) for y in (2024, 2025)]
        channels, data, _, issues = assemble(payloads)
        self.assertFalse(issues)
        text = point_card(channels, data, 'T2tb', MASS,
            profiled_theory(load_theory(ROOT / 'stats/stop_xsec_13p6TeV.json')),
            ROOT / 'stats/test/datacards', ROOT / 'stats/test/templates')
        self.assertEqual(sum(line.startswith('shapes signal ') for line in text.splitlines()), 4)
        self.assertIn('$CHANNEL/signal_T2tb_' + MASS, text)
        self.assertIn('* autoMCStats 10 1 1', text)
        names = next(line.split()[1:] for line in text.splitlines()
                     if line.startswith('process ') and 'signal' in line.split())
        nuisance = next(line.split()[2:] for line in text.splitlines()
                        if line.startswith('CMS_NPS26012_cross_section_stop_13p6TeV lnN'))
        self.assertEqual(len(names), len(nuisance))
        self.assertTrue(all((n != '-') == (p == 'signal') for p, n in zip(names, nuisance)))
        self.assertNotIn('CR_SR', text)

    def test_template_integrity_cache_invalidates_changed_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            path = Path(folder) / 'hash_test.json'
            write_json(path, dict(value=1))
            cache = IntegrityCache()
            with patch('TROTASR.workflows.combine_limits.sha256', wraps=sha256) as digest:
                a = cache(path); self.assertEqual(a, cache(path)); self.assertEqual(digest.call_count, 1)
                write_json(path, dict(value=2))
                self.assertNotEqual(a, cache(path)); self.assertEqual(digest.call_count, 2)

    def test_two_year_validation_only_is_resumable_and_never_writes_root(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            folder = Path(folder)
            grid_path = folder / 'grid.json'; write_json(grid_path, grid_fixture())
            config = dict(grid=str(grid_path.relative_to(ROOT)), theory='stats/stop_xsec_13p6TeV.json', years={})
            for year in (2024, 2025):
                values = year_fixture(year); paths = {}
                for key, value in zip(FIELDS, values):
                    path = folder / str(year) / (key + '.json.gz')
                    if key != 'hists':
                        value['provenance']['hist_input_sha256'] = sha256(ROOT / paths['hists'])
                    write_json(path, value); paths[key] = str(path.relative_to(ROOT))
                config['years'][str(year)] = paths
            cfg_path = folder / 'config.json'; write_json(cfg_path, config)
            output = folder / 'grid'
            self.assertEqual(build(cfg_path, output, validate_only=True), 0)
            digest = sha256(output / 'inputs/2024.json.gz')
            self.assertEqual(build(cfg_path, output, validate_only=True), 0)
            self.assertEqual(digest, sha256(output / 'inputs/2024.json.gz'))
            state = read_json(output / 'state.json')
            self.assertEqual(state['status'], 'validated_inputs_not_root_templates')
            self.assertEqual(state['bins'], 540)
            self.assertEqual(state['ready_inputs'], 3)
            self.assertFalse(state['root_integrity_checked'])
            self.assertFalse(list(folder.rglob('*.root')))
            with self.assertRaisesRegex(ValueError, 'Full-input production'):
                extract_year(config['years']['2024'], grid_fixture())
            measurement = ROOT / config['years']['2024']['sgamma']
            bad = read_json(measurement); bad['provenance']['hist_input_sha256'] = 'wrong'
            write_json(measurement, bad)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                extract_year(config['years']['2024'], grid_fixture(), allow_validation=True)

    def test_preserved_template_validation_retains_zero_support_and_blocks_fit(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
            folder = Path(folder)
            grid_path = folder/'grid.json'; write_json(grid_path, grid_fixture())
            config = dict(grid=str(grid_path.relative_to(ROOT)), theory='stats/stop_xsec_13p6TeV.json', years={})
            for year in (2024, 2025):
                values = year_fixture(year)
                payload = values[0]; payload['contract']['shape_mode'] = 'cms_trota_jme'
                for mode, hist in payload['histograms'].items():
                    for endpoint in ('jesTotalUp','jesTotalDown','jerUp','jerDown'):
                        hist[endpoint] = copy.deepcopy(hist['nominal'])
                for endpoint in ('nominal','jesTotalDown','jerUp','jerDown'):
                    for category, samples in payload['histograms']['highdm'][endpoint]['SR'].items():
                        if not category.startswith('1,'):
                            continue
                        for sample, record in samples.items():
                            if sample.startswith('QCD'):
                                record['recoil']['sumw'][0] = 0.
                                record['recoil']['sumw2'][0] = 0.
                paths = {}
                for key, value in zip(FIELDS, values):
                    path = folder/str(year)/(key+'.json.gz')
                    if key != 'hists':
                        value['provenance']['hist_input_sha256'] = sha256(ROOT/paths['hists'])
                    write_json(path, value); paths[key] = str(path.relative_to(ROOT))
                config['years'][str(year)] = paths
            cfg = folder/'config.json'; write_json(cfg, config)
            with self.assertRaisesRegex(ValueError, 'Unsupported statistical'):
                build(cfg, folder/'strict', validate_only=True)
            output = folder/'preserved'
            self.assertEqual(build(cfg, output, validate_only=True, preserve_unsupported=True), 2)
            state = read_json(output/'state.json')
            self.assertEqual(state['status'], 'blocked')
            self.assertEqual(state['template_points'], 3)
            self.assertEqual(state['ready_inputs'], 0)
            self.assertTrue(state['fit_compatibility']['cr_only_ready'])
            self.assertTrue(state['fit_compatibility']['background_issues'])
            self.assertEqual(state['fit_compatibility']['auto_mc_stats'], [10,1,1])
            self.assertFalse(state['root_integrity_checked'])
            self.assertFalse(list(folder.rglob('*.root')))
            self.assertEqual(build(cfg, output, validate_only=True, preserve_unsupported=True), 2)


if __name__ == '__main__':
    unittest.main()
