import unittest
from unittest.mock import patch
from TROTASR.workflows import campaign_resources as control


class MemoryControlTests(unittest.TestCase):
    def test_collection_preserves_values_and_order(self):
        data=[object(),object(),object()]
        with patch.object(control.gc,'collect') as collect:
            self.assertEqual(list(control.collect_chunks(iter(data))),data)
            self.assertEqual(collect.call_count,3)

    def test_collection_on_early_close(self):
        with patch.object(control.gc,'collect') as collect:
            generator=control.collect_chunks(iter([1,2]))
            self.assertEqual(next(generator),1)
            generator.close()
            collect.assert_called_once_with()

    def plan(self):
        return dict(max_workers=50, aggregate_slots=100,
                    minimum_available_memory_bytes=32*control.GIB,
                    admission_bytes_per_worker=6*control.GIB,
                    memory_control_policy=dict(initial_workers=32,ramp_seconds=600,peak_safety_factor=1.25))

    def test_initial_cap_and_slow_ramp(self):
        plan=self.plan()
        for elapsed,cap in ((0,32),(599,32),(600,33),(10800,50),(90000,50)):
            result,audit=control.reserved_capacity(plan,0,1000*control.GIB,[],0,elapsed)
            self.assertEqual(result,cap)

    def test_started_workers_still_reserve_six_gib_each(self):
        rss=[control.GIB]*32
        limit,audit=control.reserved_capacity(self.plan(),0,180*control.GIB,rss,control.GIB,3600)
        self.assertEqual(limit,32)
        self.assertEqual(audit['reserved_growth_bytes'],160*control.GIB)

    def test_later_growth_raises_future_budget(self):
        _,audit=control.reserved_capacity(self.plan(),0,100*control.GIB,[7*control.GIB],7*control.GIB,0)
        self.assertEqual(audit['worker_budget_bytes'],9*control.GIB)
        self.assertEqual(audit['reserved_growth_bytes'],2*control.GIB)

    def test_global_cpu_budget_and_low_memory(self):
        plan=self.plan()
        self.assertEqual(control.reserved_capacity(plan,80,200*control.GIB,[control.GIB]*10,0,0)[0],19)
        self.assertEqual(control.reserved_capacity(plan,0,20*control.GIB,[control.GIB]*10,0,0)[0],10)

    def test_adaptive_budget_matches_2024_formula_not_six_gib_floor(self):
        plan=self.plan();plan[control.FIELD]['budget_mode']='adaptive_peak_growth'
        _,audit=control.reserved_capacity(plan,0,200*control.GIB,[control.GIB],
                                         2*control.GIB,0,control.GIB//4)
        self.assertEqual(audit['worker_budget_bytes'],int(2*control.GIB*1.1)+control.GIB//2)

    def test_adaptive_growth_never_reduces_budget(self):
        plan=self.plan();plan[control.FIELD]['budget_mode']='adaptive_peak_growth'
        for growth in (0,-control.GIB):
            _,audit=control.reserved_capacity(plan,0,100*control.GIB,[],0,0,growth)
            self.assertEqual(audit['worker_budget_bytes'],control.GIB)

    def test_adaptive_keeps_global_and_memory_reserves(self):
        plan=self.plan();plan[control.FIELD]['budget_mode']='adaptive_peak_growth'
        self.assertEqual(control.reserved_capacity(plan,95,200*control.GIB,[],0,0)[0],4)
        self.assertEqual(control.reserved_capacity(plan,0,33*control.GIB,[],0,0)[0],1)

    def scaled_plan(self):
        plan=self.plan()
        plan.update(max_workers=98,admission_bytes_per_worker=2*control.GIB)
        plan[control.FIELD].update(post_collection_stages=[64,80,98],budget_quantum_bytes=control.GIB//4,
                                   legacy_workers_exited_verified=True,collect_after_chunk=True)
        return plan

    def test_new_worker_stages_never_exceed_98(self):
        for elapsed,cap in ((0,64),(599,64),(600,80),(1199,80),(1200,98),(90000,98)):
            limit,audit=control.reserved_capacity(self.scaled_plan(),0,1000*control.GIB,[],0,elapsed)
            self.assertEqual(limit,cap)

    def test_new_worker_budget_tracks_new_peak_with_margin(self):
        plan=self.scaled_plan()
        _,audit=control.reserved_capacity(plan,0,220*control.GIB,[],int(1.5*control.GIB),0)
        self.assertEqual(audit['worker_budget_bytes'],2*control.GIB)
        _,audit=control.reserved_capacity(plan,0,220*control.GIB,[],2*control.GIB,0)
        self.assertEqual(audit['worker_budget_bytes'],int(2.5*control.GIB))

    def test_new_worker_scaling_keeps_32_gib_and_aggregate_cpu(self):
        plan=self.scaled_plan()
        self.assertEqual(control.reserved_capacity(plan,20,1000*control.GIB,[],0,1200)[0],79)
        self.assertEqual(control.reserved_capacity(plan,0,40*control.GIB,[],0,1200)[0],4)
        self.assertEqual(control.reserved_capacity(plan,0,96*control.GIB,[control.GIB]*64,0,1200)[0],64)

    def test_new_scaling_requires_legacy_exit_and_collection(self):
        for field in ('legacy_workers_exited_verified','collect_after_chunk'):
            plan=self.scaled_plan();plan[control.FIELD][field]=False
            with self.assertRaisesRegex(ValueError,'verified legacy exit'):
                control.reserved_capacity(plan,0,1000*control.GIB,[],0,0)

    def orphan(self):
        return control.OrphanProcess(dict(pid=900,start_identity=0,attempt=2,history='does_not_exist',
            identity=dict(pid=900,start_ticks=123,arguments=['python','worker'])))

    def test_retained_live_worker(self):
        proc=self.orphan()
        with patch.object(control,'process_identity',return_value=proc.record['identity']):
            self.assertIsNone(proc.poll())

    def test_empty_cmdline_exit_race_does_not_crash_controller(self):
        proc=self.orphan()
        with patch.object(control,'process_identity',return_value=dict(proc.record['identity'],arguments=[])):
            self.assertIsNone(proc.poll())
        with patch.object(control,'process_identity',return_value=None):
            self.assertEqual(proc.poll(),1)

    def test_reused_pid_is_not_adopted_and_does_not_crash_controller(self):
        proc=self.orphan()
        with patch.object(control,'process_identity',return_value=dict(proc.record['identity'],start_ticks=124)):
            self.assertEqual(proc.poll(),1)
            self.assertIn('different process',proc.identity_mismatch)

    def test_changed_command_fails_only_this_attempt(self):
        proc=self.orphan()
        with patch.object(control,'process_identity',return_value=dict(proc.record['identity'],arguments=['different'])):
            self.assertEqual(proc.poll(),1)
            self.assertIn('command changed',proc.identity_mismatch)

    def test_resubmission_exactly_once(self):
        plan={control.FIELD:dict(resubmissions={'a':dict(failed_attempt=2,error='resource exit')})}
        state={'a':dict(status='failed',attempt=2,error='resource exit')}
        self.assertEqual(control.retry_keys(plan,state),['a'])
        for changes in (dict(attempt=3),dict(error='physics error'),dict(status='complete')):
            self.assertEqual(control.retry_keys(plan,{'a':dict(state['a'],**changes)}),[])


if __name__=='__main__':unittest.main()
