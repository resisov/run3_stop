"""Execute only compact, explicitly synthetic validation; never local ROOT I/O."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from TROTASR.tests.test_statistics_inputs import fixture, SR_LAYOUT, use_synthetic_sr_layout, SYNTHETIC_LAYOUT_DIGEST
from TROTASR.utils.io import write_json, read_json
from TROTASR.utils.paths import ROOT
from TROTASR.utils.statistics_inputs import build_inputs
from TROTASR.utils.statistical_templates import assemble
from TROTASR.workflows.build_template import main, prepare
from TROTASR.workflows.build_datacard import main as card_main


def assert_stale_digest_ignored(testcase, module, invoke, require_coverage=False):
    """Check input routing without fabricating a local physics layout."""
    class ModelReached(Exception):
        pass

    names = ('hists', 'sgamma', 'rz_high', 'rz_low', 'double_ratio')
    inputs = {n:dict(status='complete', provenance=dict(campaign_year='2025',
                hist_input_sha256='historical-merged-digest')) for n in names[1:]}
    inputs['hists'] = dict(status='complete', sr_data_blinded=True, input_metadata={},
        highdm_sr_source_axes='exact_nb_exact_topology_native_recoil_v2',
        coverage=dict(full_input_inventory=True, failed_files=0, expected_files=1, completed_files=1),
        scope='full_nominal_production',
        contract=dict(year='2025', sr_binning_sha256='current-digest', configuration_sha256='current-digest'))
    with patch.object(module,'read_json',side_effect=lambda p:inputs[p.name]):
        with patch.object(module,'build_inputs',side_effect=ModelReached), testcase.assertRaises(ModelReached):
            invoke(names)
        inputs['sgamma']['provenance']['campaign_year'] = '2024'
        with testcase.assertRaisesRegex(ValueError,'Measured factor year mismatch: sgamma'):
            invoke(names)
        inputs['sgamma']['provenance']['campaign_year'] = '2025'
        if require_coverage:
            inputs['hists']['coverage']['completed_files'] = 0
            with testcase.assertRaisesRegex(ValueError,'Full-input production coverage'):
                invoke(names)


class TemplateInputBoundaryTests(unittest.TestCase):
    def test_stale_merged_hash_is_ignored_but_wrong_measurement_era_is_rejected(self):
        from TROTASR.workflows import build_template as module
        assert_stale_digest_ignored(self, module,
            lambda names: prepare(*(ROOT/n for n in names),'T2tt',1200,500))

    def test_card_requires_actual_verified_root_templates(self):
        card = ROOT/'results/synthetic-card-must-not-be-written.txt'
        with patch('TROTASR.workflows.build_datacard.read_json',return_value={
                'status':'validated_inputs_not_root_templates','root_integrity_checked':False}), \
             patch.object(sys,'argv',['card','--templates',str(ROOT/'results'), '--output',str(card)]):
            with self.assertRaisesRegex(ValueError,'Real ROOT templates'):
                card_main()

    def test_external_path_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError,'escapes TROTASR'):
            prepare(ROOT.parent/'external_histograms.json',*(ROOT/n for n in (
                'sgamma','rz_high','rz_low','double_ratio')),'T2tt',1200,500)


class TemplateWorkflowTests(unittest.TestCase):
    def setUp(self):
        if not SR_LAYOUT.is_file():
            from TROTASR.workflows import build_template as module
            use_synthetic_sr_layout(self, module)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'results')
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        values=fixture()
        values[0]['contract'].update(sr_binning_sha256=SYNTHETIC_LAYOUT_DIGEST,
                                    configuration_sha256='legacy-gnn')
        self.paths=[]
        for name,value in zip(('hists','sgamma','rz_high','rz_low','double_ratio'),values):
            path=self.folder/(name+'.json.gz')
            if name!='hists':
                value['provenance']['hist_input_sha256']='legacy-input'
            write_json(path,value)
            self.paths.append(path)
        self.output=self.folder/'templates'
        self.argv=['build_template','--topology','T2tt','--mstop','1200','--mlsp','500','--output',str(self.output)]
        for name,path in zip(('hists','sgamma','rz-high','rz-low','double-ratio'),self.paths):
            self.argv+=['--'+name,str(path)]

    def test_validate_only_is_reusable_and_never_claims_root_success(self):
        with patch.object(sys,'argv',self.argv+['--validate-only']),redirect_stdout(io.StringIO()):
            main()
            before=(self.output/'model_manifest.json').read_bytes()
            main()
        manifest=read_json(self.output/'model_manifest.json')
        self.assertEqual(manifest['status'],'validated_inputs_not_root_templates')
        self.assertFalse(manifest['root_integrity_checked'])
        self.assertEqual(sum(manifest['bins'].values()),270)
        self.assertFalse(list(self.folder.rglob('*.root')))
        self.assertEqual(before,(self.output/'model_manifest.json').read_bytes())
        self.assertFalse(manifest['full_workflow_complete'])

    def test_two_year_packing_keeps_540_bins_and_16_channels(self):
        payloads=[build_inputs(*fixture(y),'T2tt',1200,500) for y in (2024,2025)]
        channels,_,_,issues=assemble(payloads)
        self.assertFalse(issues)
        self.assertEqual(len(channels),16)
        self.assertEqual(sum(map(len,channels.values())),540)

if __name__=='__main__': unittest.main()
