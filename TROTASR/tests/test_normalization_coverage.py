import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from TROTASR.utils.io import read_json,write_json
from TROTASR.utils.paths import ROOT
from TROTASR.workflows.normalization_metadata import build_signal_mass_points, load_signal_xsec


class CommonCrossSectionTest(unittest.TestCase):
    primary = 'TTto4Q_TuneCP5_13p6TeV_powheg-pythia8'

    def reference(self):
        return dict(datasets={'1': dict(dataset=self.primary+
            '-RunIII2024Summer24NanoAODv15-v2____1_', xsec_pb=419.82,
            is_data=False, is_signal=False, sumw=123., luminosity_pb=999.)})

    def test_production_labels_do_not_select_cross_section(self):
        from TROTASR.workflows.normalization_metadata import primary_dataset, background_xsec_index, apply_background_xsecs
        from TROTASR.workflows.normalization_metadata import build_physical
        ref = self.reference()
        index = background_xsec_index([('existing', ref)])
        self.assertEqual(primary_dataset(ref['datasets']['1']['dataset']), self.primary)
        for era, lumi, denominator in (('2022',7980.4,100.), ('2022EE',26671.7,200.),
                                       ('2023',17794.,300.), ('2023BPix',9451.,400.)):
            records = {'42': dict(dataset=f'/{self.primary}/Run3Summer{era}NanoAODv12-v9/NANOAODSIM',
                                   xsec_pb=None, is_data=False, is_signal=False,
                                   sumw=denominator, files_processed=8)}
            apply_background_xsecs(records, index)
            physical, _, factors = build_physical(records, lumi)
            self.assertEqual(records['42']['xsec_pb'], 419.82)
            self.assertEqual(records['42']['sumw'], denominator)
            self.assertEqual(factors['42']['normalization_factor'], 419.82*lumi/denominator)
            self.assertEqual(factors['42']['physical_dataset_sumw'], denominator)
        self.assertEqual(ref, self.reference())

    def test_different_process_energy_and_filter_are_not_aliased(self):
        from TROTASR.workflows.normalization_metadata import background_xsec_index, apply_background_xsecs
        index = background_xsec_index([('existing', self.reference())])
        for name in (self.primary.replace('4Q','2L2Nu'), self.primary.replace('13p6','13'),
                     self.primary.replace('_Tune','_HT-600_Tune')):
            records = {'1':dict(dataset=f'/{name}/any/NANOAODSIM', xsec_pb=None)}
            self.assertEqual(len(apply_background_xsecs(records, index)), 1)
            self.assertIsNone(records['1']['xsec_pb'])

    def test_conflicts_are_not_hidden_by_era_or_source_order(self):
        from TROTASR.workflows.normalization_metadata import background_xsec_index
        a, b = self.reference(), self.reference()
        b['datasets']['1']['xsec_pb'] = 400.
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            background_xsec_index([('old', a), ('new', b)])

    def test_existing_values_and_data_signal_are_preserved(self):
        from TROTASR.workflows.normalization_metadata import background_xsec_index, apply_background_xsecs
        index = background_xsec_index([('existing', self.reference())])
        records = {'1':dict(dataset=self.primary, xsec_pb=419.82),
                   '2':dict(dataset='data', is_data=True, xsec_pb=None),
                   '3':dict(dataset='SMS', is_signal=True, xsec_pb=None)}
        before=copy.deepcopy(records)
        self.assertEqual(apply_background_xsecs(records, index), [])
        self.assertEqual(records['1']['xsec_pb'],before['1']['xsec_pb'])
        self.assertEqual(records['2'],before['2'])
        self.assertEqual(records['3'],before['3'])


    def test_wz_name_permutation_keeps_every_generated_bin(self):
        from TROTASR.workflows.normalization_metadata import adopted_background_xsecs, apply_background_xsecs
        index = adopted_background_xsecs()
        for process, variable in (('WtoLNu', 'PTLNu'), ('Zto2Nu', 'PTNuNu')):
            for interval in ('40to100', '100to200', '200to400', '400to600', '600'):
                for jets in ('1J', '2J'):
                    tail = 'TuneCP5_13p6TeV_amcatnloFXFX-pythia8'
                    early = f'{process}-2Jets_{variable}-{interval}_{jets}_{tail}'
                    canonical = f'{process}-2Jets_Bin-{jets}-{variable}-{interval}_{tail}'
                    records = {'1':dict(dataset=f'/{early}/campaign/NANOAODSIM',xsec_pb=None)}
                    self.assertEqual(apply_background_xsecs(records,index),[])
                    self.assertEqual(records['1']['xsec_pb'],index[canonical]['xsec_pb'])
                    self.assertEqual(records['1']['xsec_source']['matched_sample'],canonical)

    def test_name_mapping_does_not_alias_generator_or_parton_multiplicity(self):
        from TROTASR.workflows.normalization_metadata import cross_section_sample_key
        base = 'WtoLNu-2Jets_PTLNu-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8'
        for other in (base.replace('1J','0J'),base.replace('FXFX','FXFXold'),
                      base.replace('13p6','13'),base.replace('PTLNu','PTNuNu')):
            self.assertEqual(cross_section_sample_key(other),other)

    def test_extension_statistics_share_one_cross_section(self):
        from TROTASR.workflows.normalization_metadata import build_era_physical
        records={str(i):dict(dataset=f'/{self.primary}/Run3Summer22{ext}/NANOAODSIM',
                            physical_dataset=f'/{self.primary}/Run3Summer22{ext}/NANOAODSIM',
                            xsec_pb=419.82,sumw=weight)
                 for i,(ext,weight) in enumerate((('',100.),('_ext1',300.)))}
        before=copy.deepcopy(records)
        physical,_,factors=build_era_physical(records,7980.4)
        self.assertEqual(len(physical),1)
        self.assertEqual(records,before)
        for row in factors.values():
            self.assertEqual(row['physical_dataset_sumw'],400.)
            self.assertEqual(row['normalization_factor'],419.82*7980.4/400.)
        total=sum(records[k]['sumw']*v['normalization_factor'] for k,v in factors.items())
        self.assertAlmostEqual(total,419.82*7980.4)

    def test_registered_official_values_preserve_sample_definitions(self):
        from TROTASR.utils.paths import ROOT
        from TROTASR.workflows.normalization_metadata import adopted_background_xsecs
        index=adopted_background_xsecs()
        self.assertEqual(index['QCD-4Jets_Bin-HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8']['xsec_pb'],1960000.)
        self.assertEqual(index['TTWW_TuneCP5_13p6TeV_madgraph-pythia8']['xsec_pb'],.008191)
        if (ROOT/'jsons/early_run3_assets.json').is_file():
            self.assertEqual(index['QCD-4Jets_HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8']['xsec_pb'],1961000.)
            self.assertEqual(index['TTWW_TuneCP5_13p6TeV_madgraph-madspin-pythia8']['xsec_pb'],.008203)


if __name__=='__main__':unittest.main()
