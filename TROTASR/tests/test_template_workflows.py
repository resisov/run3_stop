"""Execute only compact, explicitly synthetic validation; never local ROOT I/O."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from TROTASR.tests.test_statistics_inputs import fixture
from TROTASR.utils.io import write_json, read_json, sha256
from TROTASR.utils.paths import ROOT
from TROTASR.utils.statistics_inputs import build_inputs
from TROTASR.utils.statistical_templates import assemble
from TROTASR.workflows.build_template import main, prepare
from TROTASR.workflows.build_datacard import main as card_main


class TemplateWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'results')
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        values=fixture()
        values[0]['contract'].update(sr_binning_sha256=sha256(ROOT/'highdm_sr_binning.json'),
                                    configuration_sha256=sha256(ROOT/'gnn4lowdm/config.json'))
        self.paths=[]
        for name,value in zip(('hists','sgamma','rz_high','rz_low','double_ratio'),values):
            path=self.folder/(name+'.json.gz')
            if name!='hists':
                value['provenance']['hist_input_sha256']=sha256(self.paths[0])
            write_json(path,value)
            self.paths.append(path)
        self.output=self.folder/'templates'
        self.argv=['build_template','--topology','T2tt','--mstop','1200','--mlsp','500','--output',str(self.output)]
        for name,path in zip(('hists','sgamma','rz-high','rz-low','double-ratio'),self.paths):
            self.argv+=['--'+name,str(path)]

    def test_validate_only_is_reusable_and_never_claims_root_success(self):
        with patch.object(sys,'argv',self.argv+['--validate-only']),redirect_stdout(io.StringIO()):
            main()
            before=sha256(self.output/'model_manifest.json')
            main()
        manifest=read_json(self.output/'model_manifest.json')
        self.assertEqual(manifest['status'],'validated_inputs_not_root_templates')
        self.assertFalse(manifest['root_integrity_checked'])
        self.assertEqual(sum(manifest['bins'].values()),270)
        self.assertFalse(list(self.folder.rglob('*.root')))
        self.assertEqual(before,sha256(self.output/'model_manifest.json'))
        self.assertFalse(manifest['full_workflow_complete'])

    def test_unrelated_measurement_hash_rejected(self):
        factor=read_json(self.paths[1]); factor['provenance']['hist_input_sha256']='wrong'
        write_json(self.paths[1],factor)
        with self.assertRaisesRegex(ValueError,'does not match'):
            prepare(*self.paths,'T2tt',1200,500)

    def test_card_requires_actual_verified_root_templates(self):
        with patch.object(sys,'argv',self.argv+['--validate-only']),redirect_stdout(io.StringIO()):
            main()
        card=self.folder/'card.txt'
        with patch.object(sys,'argv',['card','--templates',str(self.output),'--output',str(card)]):
            with self.assertRaisesRegex(ValueError,'Real ROOT templates'):
                card_main()
        self.assertFalse(card.exists())

    def test_two_year_packing_keeps_540_bins_and_16_channels(self):
        payloads=[build_inputs(*fixture(y),'T2tt',1200,500) for y in (2024,2025)]
        channels,_,_,issues=assemble(payloads)
        self.assertFalse(issues)
        self.assertEqual(len(channels),16)
        self.assertEqual(sum(map(len,channels.values())),540)

    def test_external_path_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError,'escapes TROTASR'):
            prepare(ROOT.parent/'external_histograms.json',*self.paths[1:],'T2tt',1200,500)


if __name__=='__main__': unittest.main()
