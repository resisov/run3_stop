import copy
import importlib.util
import unittest
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json

AVAILABLE=importlib.util.find_spec('PIL') is not None
if AVAILABLE:
    from TROTASR.workflows.verify_btag_reference_figures import verify_panel


@unittest.skipUnless(AVAILABLE,'Local read-only PNG inspection dependency unavailable')
class BtagReferenceBindingTest(unittest.TestCase):
    def setUp(self):
        base=ROOT/'validation/reference_assets_20260923/provenance/btag_efficiencies'
        if not (base/'png_numeric_ocr.json').exists():self.skipTest('Local QA artifacts unavailable')
        self.data=read_json(base/'plot_data.json')['panels']
        self.ocr=read_json(base/'png_numeric_ocr.json')['panels']
        self.figure=ROOT/'plots/reference_assets_20260923'/self.data[0]['an_path']
        if not self.figure.exists():self.skipTest('Preserved PNG available on local QA host only')

    def test_actual_all_fifteen_panels(self):
        results=[]
        for panel,record in zip(self.data,self.ocr):
            image=ROOT/'plots/reference_assets_20260923'/panel['an_path']
            results.append(verify_panel(image,panel,record['observations']))
        self.assertEqual(sum(r['numeric_annotations'] for r in results),354)
        self.assertEqual(sum(r['blank_zero_denominator_cells'] for r in results),6)

    def test_changed_efficiency_rejected(self):
        changed=copy.deepcopy(self.data[0]);changed['efficiency'][1][0]-=.01
        with self.assertRaisesRegex(ValueError,'Numeric annotation mismatch'):
            verify_panel(self.figure,changed,self.ocr[0]['observations'])

    def test_missing_annotation_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing/ambiguous'):
            verify_panel(self.figure,self.data[0],self.ocr[0]['observations'][1:])

    def test_wrong_cell_position_rejected(self):
        changed=copy.deepcopy(self.ocr[0]['observations']);changed[0]['x']+=.02
        with self.assertRaisesRegex(ValueError,'Missing/ambiguous'):
            verify_panel(self.figure,self.data[0],changed)

    def test_unexplained_missing_efficiency_rejected(self):
        changed=copy.deepcopy(self.data[0]);changed['efficiency'][1][0]=None
        with self.assertRaisesRegex(ValueError,'without zero denominator'):
            verify_panel(self.figure,changed,self.ocr[0]['observations'])

    def test_binning_change_rejected(self):
        changed=copy.deepcopy(self.data[0]);changed['pt_edges'][2]=55
        with self.assertRaisesRegex(ValueError,'bin boundaries'):
            verify_panel(self.figure,changed,self.ocr[0]['observations'])


if __name__=='__main__':unittest.main()
