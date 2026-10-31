import statistics
import unittest
from TROTASR.workflows.limit_validation import crossing_check
from TROTASR.workflows.combine_limits import fit_command, collect, ATTEMPTS


class LimitCrossingTests(unittest.TestCase):
    def log(self, miss=False, bracket=False, wide=False):
        normal=statistics.NormalDist()
        values={q:float(i+1) for i,q in enumerate((.025,.16,.5,.84,.975))}
        text=''
        for q in (.5,.025,.16,.84,.975):
            target=.5*(normal.inv_cdf(q)+normal.inv_cdf(1-q*.05))**2
            text+='Will search for NLL crossing by bisection\n'
            r=values[q]
            if bracket:
                width=.5 if wide else .0001
                text+='At r = %s: delta(nll) = %s\n'%(r-width,target-.1)
                text+='At r = %s: delta(nll) = %s\n'%(r+width,target+.1)
            else:
                text+='At r = %s: delta(nll) = %s\n'%(r,target+(1. if miss else 0.))
        return text,values

    def test_exact_crossings_pass(self):
        text,values=self.log()
        self.assertTrue(crossing_check(text,values,.1)[0])

    def test_finite_values_without_crossings_fail(self):
        text,values=self.log(miss=True)
        self.assertFalse(crossing_check(text,values,.1)[0])

    def test_missing_search_fails(self):
        _,values=self.log()
        self.assertFalse(crossing_check('success',values,.1)[0])

    def test_narrow_brackets_pass(self):
        text,values=self.log(bracket=True)
        self.assertTrue(crossing_check(text,values,.1)[0])

    def test_wide_unfinished_brackets_fail(self):
        text,values=self.log(bracket=True,wide=True)
        self.assertFalse(crossing_check(text,values,.1)[0])

    def test_adopted_command_and_finite_attempts(self):
        command=fit_command('workspace.root','mStop1200_mLSP500',20.,.1)
        self.assertEqual(command[command.index('--run')+1],'blind')
        self.assertEqual(command[command.index('--cminDefaultMinimizerStrategy')+1],'0')
        self.assertEqual(command[command.index('--minosAlgo')+1],'bisection')
        self.assertIn('--picky',command)
        self.assertEqual(ATTEMPTS,(('bisection_s0',.1,1.),('bisection_s0_refined',.01,2.)))

    def test_partial_collect_does_not_claim_complete(self):
        points=[dict(model='T2tt',mass='mStop1200_mLSP500'),dict(model='T2tt',mass='mStop1300_mLSP500')]
        results={'T2tt/mStop1200_mLSP500':dict(status='complete',validation=dict(
            quantiles={'.025':1.,'.16':2.,'.5':3.,'.84':4.,'.975':5.}))}
        p=collect(points,results,'T2tt','full_nominal_production')
        self.assertEqual(p['status'],'partial')
        self.assertEqual(p['missing_points'],['mStop1300_mLSP500'])
        self.assertEqual(p['points']['mStop1200_mLSP500']['expected'],3.)
        self.assertFalse(p['observed_sr_used'])


if __name__=='__main__':unittest.main()
