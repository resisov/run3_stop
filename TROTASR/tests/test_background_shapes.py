"""The 2026-09-29 adopted representation change; no physics toy fit claims."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.utils.paths import ROOT
from TROTASR.utils.statistical_templates import card_text
from TROTASR.workflows.build_nominal_grid import background_error_shapes, background_shape_names
from TROTASR.validation.background_shapes_20260929.builder_candidate import audit_cached_inputs


POLICY = dict(id='jme_background_shapes_20260929', mode='shape',
              auto_mc_stats=[10, 1, 1], user_approved=True)


def fixture():
    data, channels = {}, {}
    for m in ('highdm', 'lowdm'):
        for y in ('2024', '2025'):
            c = 'SR_'+m+'_c1_'+y
            names = sorted(n for n in background_shape_names() if '_'+m+'_' in n and n.endswith(y))
            data[c, 'ZJet_Nb1_u0'] = dict(family='ZJet', parameter='norm_'+m+'_'+y, initial=1.3,
                nominal=np.array([0., 2., 5.]), sumw2=np.array([0., 1., 3.]),
                extra_lnN=tuple((n, .9, 1.1) for n in names)+(('unrelated', .8, 1.2),),
                variations={'existing': dict(up=np.array([1., 3., 5.]), down=np.array([0., 1., 4.]))})
            channels[c] = [dict(name='bin'+str(i), region='SR', year=y) for i in range(3)]
    return channels, data


class BackgroundShapes(unittest.TestCase):
    def test_exact_approved_names(self):
        names = background_shape_names()
        self.assertEqual(len(names), 34)
        self.assertEqual(sum('_RZstat_' in n for n in names), 8)

    def test_default_is_identity(self):
        _, data = fixture()
        result, audit = background_error_shapes(data)
        self.assertIs(result, data)
        self.assertIsNone(audit)

    def test_only_extra_type_and_new_arrays_change(self):
        channels, data = fixture()
        original = copy.deepcopy(data)
        new, audit = background_error_shapes(data, POLICY)
        self.assertEqual(set(data), set(new))
        self.assertEqual(audit['nuisances'], 34)
        for k, record in data.items():
            for name in ('nominal', 'sumw2'):
                self.assertIs(new[k][name], record[name])
                np.testing.assert_array_equal(record[name], original[k][name])
            for name in ('family', 'parameter', 'initial'):
                self.assertEqual(record[name], new[k][name])
            self.assertEqual(new[k]['extra_lnN'], (('unrelated', .8, 1.2),))
            self.assertIs(new[k]['variations']['existing'], record['variations']['existing'])
            self.assertEqual(set(record['variations']), {'existing'})
            for name, down, up in record['extra_lnN']:
                if name == 'unrelated': continue
                for d, f in (('down', down), ('up', up)):
                    np.testing.assert_array_equal(new[k]['variations'][name][d], record['nominal']*f)
                    self.assertEqual(new[k]['variations'][name][d][0], 0.)
        before, after = card_text(channels, data), card_text(channels, new)
        keep = lambda t: [l for l in t.splitlines() if 'rateParam' in l or 'autoMCStats' in l]
        self.assertEqual(keep(before), keep(after))
        self.assertIn('* autoMCStats 10 1 1', after)
        for n in background_shape_names():
            self.assertIn(n+' shape ', after)
            self.assertNotIn(n+' lnN ', after)

    def test_flat_supplied_endpoint_kept_but_not_active(self):
        channels, data = fixture()
        name = sorted(background_shape_names())[0]
        for r in data.values():
            r['extra_lnN'] = tuple((n, 1., 1.) if n == name else (n,d,u) for n,d,u in r['extra_lnN'])
        new, _ = background_error_shapes(data, POLICY)
        self.assertTrue(any(name in r['variations'] for r in new.values()))
        self.assertNotIn(name+' shape ', card_text(channels, new))

    def test_guards(self):
        for problem in ('policy', 'missing', 'duplicate', 'mixed', 'nan', 'negative', 'unknown', 'scope', 'family'):
            _, data = fixture(); policy = copy.deepcopy(POLICY)
            key = next(iter(data)); r = data[key]; n,d,u = r['extra_lnN'][0]
            if problem == 'policy': policy['auto_mc_stats'] = [10,0,1]
            if problem == 'missing': r['extra_lnN'] = r['extra_lnN'][1:]
            if problem == 'duplicate': r['extra_lnN'] += ((n,d,u),)
            if problem == 'mixed': r['variations'][n] = r['variations']['existing']
            if problem == 'nan': r['extra_lnN'] = ((n,float('nan'),u),)+r['extra_lnN'][1:]
            if problem == 'negative': r['nominal'][0] = -1
            if problem == 'unknown': r['extra_lnN'] += (('CMS_NPS26012_RZstat_unknown',d,u),)
            if problem == 'scope': data['GCR_highdm_c1_2024',key[1]] = data.pop(key)
            if problem == 'family': r['family'] = 'QCD'
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                background_error_shapes(data, policy)

    def test_positive_factor_commutes_with_clip_then_projection(self):
        # signed sums are clipped in native bins BEFORE summing merged bins.
        signed = np.array([-2., 5., 0., 1.])
        for f in (.7, 1.4):
            projected_then_scaled = np.maximum(signed,0).reshape(2,2).sum(axis=1)*f
            scaled_clipped_projected = np.maximum(signed*f,0).reshape(2,2).sum(axis=1)
            np.testing.assert_allclose(projected_then_scaled, scaled_clipped_projected)
        self.assertNotEqual(max(sum(signed[:2]),0), sum(np.maximum(signed[:2],0)))

    def test_cache_reuse_strict_hashes_and_extraction_contract(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
            base = Path(temp); source = base/'old'; output = base/'new'
            source.mkdir(); (source/'inputs').mkdir()
            prior = base/'prior.json'; write_json(prior, dict(years={'2024':{}}, physics='fixed'))
            builder = base/'builder.py'; builder.write_text('# prior snapshot\n')
            old = dict(config_sha256=sha256(prior), code={'workflows/build_nominal_grid.py':sha256(builder), 'extractor':'abc'}, inputs={'x':'hash'}, validate_only=False)
            write_json(source/'contract.json', old)
            hashes={}
            for y in ('2024','2025'):
                p = source/'inputs'/(y+'.json.gz'); write_json(p, dict(year=y)); hashes[y]=sha256(p)
                write_json(p.with_suffix('.receipt.json'), dict(contract=old,sha256=hashes[y]))
            reuse=dict(source_output=str(source.relative_to(ROOT)),contract_sha256=sha256(source/'contract.json'),
                prior_config=str(prior.relative_to(ROOT)),prior_builder=str(builder.relative_to(ROOT)),cache_sha256=hashes)
            config=dict(years={'2024':{}},physics='fixed',reused_input_cache=reuse,background_estimation_uncertainties=POLICY)
            contract=copy.deepcopy(old); contract['code']['workflows/build_nominal_grid.py']='new'; contract['config_sha256']='new'
            result=audit_cached_inputs(output,contract,config)
            self.assertEqual(result['source'],source)
            for field in ('inputs','code','config','cache','source'):
                c=copy.deepcopy(contract); cfg=copy.deepcopy(config); out=output
                if field=='inputs': c['inputs']['x']='changed'
                if field=='code': c['code']['extractor']='changed'
                if field=='config': cfg['physics']='changed'
                if field=='cache': cfg['reused_input_cache']['cache_sha256']['2024']='wrong'
                if field=='source': out=source
                with self.subTest(field=field),self.assertRaises(ValueError): audit_cached_inputs(out,c,cfg)
            self.assertEqual(read_json(source/'contract.json'),old)


if __name__ == '__main__':
    unittest.main()
