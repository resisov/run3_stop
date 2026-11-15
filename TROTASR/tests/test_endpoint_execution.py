import copy
import unittest
from TROTASR.utils.corrections import NONJME_ENDPOINTS
from TROTASR.workflows.merge_nominal import assemble_endpoints, validate_histograms


def payload(endpoint):
    leaf=dict(sumw=[2.],sumw2=[4.],entries=[1],edges=[0.,1.])
    return dict(contract=dict(object_task_endpoint=endpoint,input='same',fingerprint={'size':3},year=2024),
        histograms={'highdm':{endpoint:{'x':leaf}}},physical_histograms={},
        input_metadata={'a':dict(sumw=5.,events_read=7)},
        dy_rz={'value':1} if endpoint=='nominal' else {},dy_rz_variations={},
        audit=dict(events_read=7,migrations={'x':1} if endpoint=='nominal' else {},
            chunks=[{'endpoint':endpoint}],weights=[{'endpoint':endpoint}],
            object_endpoints={} if endpoint=='nominal' else {endpoint:dict(events=7)}))


class EndpointExecutionTests(unittest.TestCase):

    def test_histograms_keep_signed_yields_but_reject_invalid_arrays(self):
        good=dict(x=dict(sumw=[[-1.,2.]],sumw2=[[1.,4.]],entries=[[1,1]],edges=[[0,1],[0,1,2]]))
        validate_histograms(good)
        for field,value in [('sumw',float('nan')),('sumw',float('inf')),
                            ('sumw2',-1.),('entries',-1.),('entries',1.5)]:
            bad=copy.deepcopy(good);bad['x'][field][0][0]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                validate_histograms(bad)
        bad=copy.deepcopy(good);bad['x']['sumw']=[[True,False]]
        with self.assertRaises(ValueError):validate_histograms(bad)
        bad=copy.deepcopy(good);del bad['x']['sumw2']
        with self.assertRaises(ValueError):validate_histograms(bad)


    def test_disjoint_merge_counts_metadata_once(self):
        result=assemble_endpoints((payload(e) for e in ['nominal',*NONJME_ENDPOINTS]),NONJME_ENDPOINTS)
        self.assertEqual(len(result['histograms']['highdm']),17)
        self.assertEqual(result['audit']['events_read'],7)
        self.assertEqual(result['input_metadata']['a']['sumw'],5.)
        self.assertEqual(result['audit']['migrations'],{'x':1})
        self.assertEqual(result['histograms']['highdm']['nominal']['x']['sumw'],[2.])

    def test_missing_endpoint_rejected(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            assemble_endpoints([payload('nominal')],NONJME_ENDPOINTS)

    def test_duplicate_endpoint_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            assemble_endpoints([payload('nominal'),payload('up'),payload('up')],['up'])

    def test_shift_before_nominal_rejected(self):
        with self.assertRaisesRegex(ValueError,'first'):
            assemble_endpoints([payload('up'),payload('nominal')],['up'])

    def test_changed_metadata_rejected(self):
        p=payload('up');p['input_metadata']['a']['sumw']=6.
        with self.assertRaisesRegex(ValueError,'accounting'):
            assemble_endpoints([payload('nominal'),p],['up'])

    def test_duplicate_nominal_rejected(self):
        p=payload('up');p['audit']['migrations']={'x':1}
        with self.assertRaisesRegex(ValueError,'nominal'):
            assemble_endpoints([payload('nominal'),p],['up'])

    def test_endpoint_accepts_different_file_record(self):
        p = payload('up')
        p['contract']['fingerprint'] = {'size': 4}
        self.assertEqual(assemble_endpoints([payload('nominal'), p], ['up']),
                         assemble_endpoints([payload('nominal'), payload('up')], ['up']))

    def test_unexpected_hist_endpoint_rejected(self):
        p=payload('up');p['histograms']['highdm']['nominal']={}
        with self.assertRaisesRegex(ValueError,'contamination'):
            assemble_endpoints([payload('nominal'),p],['up'])

    def test_empty_shards_are_valid(self):
        values=[payload(e) for e in ['nominal','up']]
        for p in values:
            p.update(histograms={},dy_rz={},input_metadata={})
            p['audit'].update(events_read=0,migrations={},object_endpoints={},chunks=[],weights=[])
        result=assemble_endpoints(values,['up'])
        self.assertEqual(result['audit']['events_read'],0)


if __name__=='__main__':unittest.main()
