"""Synthetic tests for structural ROOT checks and closed-file resume."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TROTASR.utils.paths import ERAS
from TROTASR.utils.statistical_templates import template_name
from TROTASR.workflows import build_nominal_grid as builder


class StructuralTemplateTests(unittest.TestCase):
    channel = 'SR_highdm_c1_2022'
    mass = 'mStop1200_mLSP500'

    def setUp(self):
        channels = {self.channel: [dict(year='2022', name='bin0')]}
        sources = {('2022', 'bin0'): [('highdm', 0)]}
        background = {(self.channel, 'Top'): dict(variations={'JES': {}})}
        caches = {'2022': dict(signals={'T2tt': {self.mass: {'highdm': dict(variations={'JES': {}})}}})}
        self.filename = template_name(self.channel)
        self.expected = builder.minimal_template_inventory(
            channels, [dict(model='T2tt', mass=self.mass)], caches, sources, background)[self.filename]
        self.keys = {
            'data_obs', 'Top', 'Top_JESUp', 'Top_JESDown',
            'signal_T2tt_'+self.mass, 'signal_T2tt_'+self.mass+'_JESUp',
            'signal_T2tt_'+self.mass+'_JESDown'}
        self.classes = {self.channel: 'TDirectoryFile',
                        **{self.channel+'/'+n: 'TH1D' for n in self.keys}}

    def verify(self, classes):
        class Root:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def classnames(self, **kwargs): return classes
            def __getitem__(self, _):
                raise AssertionError('Structural checking must not load numeric data')
        with patch.dict('sys.modules', uproot=SimpleNamespace(open=lambda *a, **kw: Root())):
            return builder.verify_minimal_template(Path(self.filename), self.expected)

    def test_reports_real_structure_without_claiming_numeric_readback(self):
        self.assertEqual(set(self.expected['names']), {self.channel+'/'+name for name in self.keys})
        row = self.verify(self.classes)
        self.assertEqual((row['objects'], row['signal_objects'], row['background_and_observation_objects']), (7,3,4))
        self.assertTrue(row['required_types_checked'])
        self.assertTrue(row['file_open_header_checked'])
        self.assertFalse(row['numeric_readback_performed'])
        self.assertFalse(row['all_expected_objects_read_back'])

    def test_missing_extra_wrong_type_and_missing_directory_are_rejected(self):
        path = self.channel+'/Top'
        changes = [({k:v for k,v in self.classes.items() if k!=path}, 'objects'),
                   (dict(self.classes, extra='TH1D'), 'objects'),
                   (dict(self.classes, **{path:'TTree'}), 'class'),
                   ({k:v for k,v in self.classes.items() if k!=self.channel}, 'directory')]
        for classes, reason in changes:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self.verify(classes)


class WrittenTemplateResumeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.output = Path(folder.name)
        self.templates = self.output/'templates'
        self.templates.mkdir()
        self.contract = dict(inputs={year:{} for year in ERAS})
        for mode in ('highdm','lowdm'):
            for year in ERAS:
                (self.templates/template_name('SR_'+mode+'_c1_'+year)).write_bytes(b'synthetic-closed-file')

    def test_inventory_only_never_rewrites_receipt_or_reopens_root(self):
        before = {p.name: p.read_bytes() for p in self.templates.iterdir()}
        with patch.object(builder,'read_json',return_value={'status':'verifying_templates'}) as read, \
             patch.object(builder,'write_json',side_effect=AssertionError('receipt rewrite')), \
             patch.object(builder,'verify_minimal_template',side_effect=AssertionError('premature ROOT check')):
            builder.resume_written_templates(self.output, self.contract)
        read.assert_called_once_with(self.output/'state.json')
        self.assertEqual({p.name:p.read_bytes() for p in self.templates.iterdir()},before)
        self.assertEqual(set(p.name for p in self.output.iterdir()), {'templates'})

    def test_only_closed_complete_template_stage_may_resume(self):
        with patch.object(builder,'read_json',return_value={'status':'writing_templates'}):
            with self.assertRaisesRegex(ValueError,'completed pre-card write stage'):
                builder.resume_written_templates(self.output,self.contract)
        with patch.object(builder,'read_json',return_value={'status':'verifying_templates'}):
            (self.output/'datacards').mkdir()
            with self.assertRaisesRegex(ValueError,'completed pre-card write stage'):
                builder.resume_written_templates(self.output,self.contract)
            (self.output/'datacards').rmdir()
            stray = self.templates/'extra.root'
            stray.write_bytes(b'not registered')
            with self.assertRaisesRegex(ValueError,'ROOT inventory'):
                builder.resume_written_templates(self.output,self.contract)
            stray.unlink()
            first = next(self.templates.iterdir())
            first.write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'missing or empty'):
                builder.resume_written_templates(self.output,self.contract)


if __name__ == '__main__':
    unittest.main()
