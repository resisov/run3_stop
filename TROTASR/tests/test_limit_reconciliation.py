import tempfile
import unittest
from pathlib import Path
from unittest import mock
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows import reconcile_limit_fits as w


def block(rows):
    return 'Will search for NLL crossing by bisection\n'+''.join(
        'At r = %s: delta(nll) = %s\n' % pair for pair in rows)


class RangeTests(unittest.TestCase):
    def test_partial_three_blocks_use_only_first_median(self):
        text=block([(20,.1),(30,.2),(39.84375,.44734)])+block([(40,99)])+block([])
        record={'attempts':[{'name':'initial','log':'first'},{'name':'refined','log':'second'}]}
        with mock.patch.object(Path,'read_text',return_value=text):
            seed,evidence=w.range_spec(record)
        self.assertEqual(seed,310)
        self.assertTrue(all(x['quantile']==.5 for x in evidence))

    def test_no_partial_evidence_rejected(self):
        with mock.patch.object(Path,'read_text',return_value=block([(20,.1),(30,.2),(39,.4)])):
            with self.assertRaisesRegex(ValueError,'missed range'):
                w.range_spec({'attempts':[{'log':'a'}]})

    def test_finite_exact_two_configs(self):
        c=w.range_configs(430)
        self.assertEqual([a['strategy'] for a in c],[0,1])
        self.assertTrue(all(a['tolerance']==.1 and a['rmax']==430 for a in c))
        for seed in [0,40,10001,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):w.range_configs(seed)


class PersistenceTests(unittest.TestCase):
    def fixture(self,base,kind='range_parser'):
        point=dict(model='T2tt',mass='mStop1700_mLSP1500',card_sha256='card')
        key=point['model']+'/'+point['mass'];work=base/key;work.mkdir(parents=True)
        original=dict(status='failed',pid=None,attempts=[],workspace_sha256='ws')
        if kind=='serialization':
            original['attempts']=[dict(name='existing',tolerance=.1,exit_code=0,
                log=str((work/'existing/combine.log').relative_to(ROOT)),passed=False)]
        write_json(work/'state.json',original)
        spec=dict(point=point,key=key,kind=kind,source_state_sha256=sha256(work/'state.json'))
        if kind=='range_parser':
            spec.update(workspace_sha256='ws',logs_sha256={},rmax=310,attempts=w.range_configs(310))
        else:
            log,product=w.attempt_output(point,original['attempts'][0])
            spec.update(attempt='existing',log=str(log.relative_to(ROOT)),output=str(product.relative_to(ROOT)),
                log_sha256='log',output_sha256='product')
        return work,original,spec

    def digest(self,path):
        p=Path(path)
        if p.name=='workspace.root':return 'ws'
        if p.suffix=='.root':return 'product'
        if p.name=='combine.log':return 'log'
        return sha256(p)

    def execute(self,base,spec,runner,passed=True):
        with mock.patch.object(w,'verify_card'),mock.patch.object(w,'sha256',side_effect=self.digest), \
             mock.patch.object(w,'validate_saved_limit',return_value=(passed,{'quantiles':{'0.5':1.}})):
            return w.run_point(spec,base,runner,{},None,'contract')

    def test_two_failed_attempts_never_repeated(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results',prefix='.limit_test_') as name:
            base=Path(name);work,old,spec=self.fixture(base)
            runner=mock.Mock();runner.run.return_value=dict(status='complete',exit_code=0)
            self.assertEqual(self.execute(base,spec,runner,False),'failed')
            self.assertEqual(self.execute(base,spec,runner,False),'failed')
            self.assertEqual(runner.run.call_count,2)
            self.assertEqual(len(read_json(work/'state.json')['attempts']),2)
            self.assertEqual(read_json(base/'history/validation_20260924'/spec['key']/'state.json'),old)

    def test_success_no_second_attempt_and_reused(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results',prefix='.limit_test_') as name:
            base=Path(name);work,old,spec=self.fixture(base)
            runner=mock.Mock();runner.run.return_value=dict(status='complete',exit_code=0)
            self.assertEqual(self.execute(base,spec,runner),'complete')
            self.assertEqual(self.execute(base,spec,runner),'complete')
            self.assertEqual(runner.run.call_count,1)

    def test_serialization_does_not_launch_any_fit(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results',prefix='.limit_test_') as name:
            base=Path(name);work,old,spec=self.fixture(base,'serialization');runner=mock.Mock()
            self.assertEqual(self.execute(base,spec,runner),'complete')
            runner.run.assert_not_called()
            record=read_json(work/'state.json')
            self.assertEqual(record['attempts'],old['attempts'])
            self.assertEqual(record['validation_reconciliation']['kind'],'serialization')

    def test_nonzero_exit_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results',prefix='.limit_test_') as name:
            base=Path(name);work,old,spec=self.fixture(base)
            runner=mock.Mock();runner.run.return_value=dict(status='failed',exit_code=1)
            self.assertEqual(self.execute(base,spec,runner),'failed')
            self.assertEqual(runner.run.call_count,2)

    def test_changed_state_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results',prefix='.limit_test_') as name:
            base=Path(name);work,old,spec=self.fixture(base)
            write_json(work/'state.json',dict(old,changed=True))
            with self.assertRaisesRegex(ValueError,'Point changed'):
                self.execute(base,spec,mock.Mock())

    def test_changed_serialized_product_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results',prefix='.limit_test_') as name:
            base=Path(name);work,old,spec=self.fixture(base,'serialization');spec['output_sha256']='different'
            with self.assertRaisesRegex(ValueError,'Saved fit/log changed'):
                self.execute(base,spec,mock.Mock())

    def test_other_range_point_or_third_attempt_rejected(self):
        for mutation in ['key','attempts']:
            with tempfile.TemporaryDirectory(dir=ROOT/'results',prefix='.limit_test_') as name:
                base=Path(name);work,old,spec=self.fixture(base)
                if mutation=='key':
                    with mock.patch.object(w,'RANGE_KEYS',set()):
                        with self.assertRaisesRegex(ValueError,'scope broadened'):
                            self.execute(base,spec,mock.Mock())
                else:
                    spec['attempts'].append(dict(spec['attempts'][0],name='third'))
                    with self.assertRaisesRegex(ValueError,'scope broadened'):
                        self.execute(base,spec,mock.Mock())


if __name__=='__main__':unittest.main()
