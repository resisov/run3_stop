"""Conservative memory reservations and identity-safe orphan supervision."""
import math
import os
import contextlib
import gc
from pathlib import Path
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json
from TROTASR.workflows.normalization_recovery import process_identity

FIELD = 'memory_control_policy'
GIB = 1024**3


def collect_chunks(chunks):
    """Release unreachable inference objects after all endpoints of a chunk."""
    for chunk in chunks:
        try:
            yield chunk
        finally:
            gc.collect()


@contextlib.contextmanager
def chunk_collection(enabled):
    if not enabled:
        yield
        return
    from TROTASR.utils.private_scales import TopWEvents
    original = TopWEvents.iterate
    def iterate(reader, *args, **kwargs):
        yield from collect_chunks(original(reader, *args, **kwargs))
    TopWEvents.iterate = iterate
    try:
        yield
    finally:
        TopWEvents.iterate = original


def memory_usage(pid):
    try:
        fields = dict(line.split(':', 1) for line in (Path('/proc')/str(pid)/'status').read_text().splitlines() if ':' in line)
        return {key: int(fields.get(key, '0 kB').split()[0])*1024 for key in ('VmRSS', 'VmHWM')}
    except FileNotFoundError:
        return {'VmRSS': 0, 'VmHWM': 0}


def reserved_capacity(plan, external, available, rss, observed_peak, elapsed):
    """Reserve each active worker's future growth, not just its current RSS."""
    spec = plan[FIELD]
    quantum = spec.get('budget_quantum_bytes', GIB)
    budget = max(plan['admission_bytes_per_worker'],
                 math.ceil(observed_peak*spec['peak_safety_factor']/quantum)*quantum)
    growth = sum(max(0, budget-value) for value in rss)
    slots = max(0, (available-plan['minimum_available_memory_bytes']-growth)//budget)
    stages = spec.get('post_collection_stages')
    if stages:
        if not spec.get('legacy_workers_exited_verified') or not spec.get('collect_after_chunk'):
            raise ValueError('Post-collection scaling requires verified legacy exit and active collection')
        ramp = min(plan['max_workers'], stages[min(int(elapsed//spec['ramp_seconds']),len(stages)-1)])
    else:
        ramp = min(plan['max_workers'], spec['initial_workers']+int(elapsed//spec['ramp_seconds']))
    limit = max(0, min(ramp, plan['aggregate_slots']-1-external, len(rss)+slots))
    return int(limit), dict(worker_budget_bytes=budget, reserved_growth_bytes=growth,
                           observed_peak_bytes=observed_peak, ramp_limit=ramp)


class OrphanProcess:
    def __init__(self, record):
        self.record, self.pid = record, record['pid']
        self.identity_mismatch = None

    def poll(self):
        actual = process_identity(self.pid)
        if actual is not None:
            if actual['start_ticks'] != self.record['identity']['start_ticks']:
                # The original process has exited. Never adopt/signal the new PID.
                self.identity_mismatch = 'original PID now belongs to a different process'
            elif not actual['arguments']:
                # /proc/cmdline can empty during exit; recheck next cycle.
                return None
            elif actual != self.record['identity']:
                self.identity_mismatch = 'original process command changed; detached without signaling'
                return 1
            else:
                return None
        path = ROOT/self.record['history']
        return 0 if path.exists() and read_json(path).get('status') == 'complete' else 1


def adopt_worker(plan, campaign, key, state):
    record = plan.get(FIELD, {}).get('retained_workers', {}).get(key)
    if record is None:
        return None
    if (state.get('pid'), state.get('attempt')) != (record['pid'], record['attempt']):
        raise ValueError('Retained memory-control worker state differs: '+key)
    process = OrphanProcess(record)
    process.poll()
    return process, open(os.devnull, 'w'), record['attempt']


def retry_keys(plan, states):
    """One explicitly recorded resource resubmission, never a blind loop."""
    result = []
    for key, record in plan.get(FIELD, {}).get('resubmissions', {}).items():
        state = states.get(key, {})
        if (state.get('status') == 'failed' and state.get('attempt') == record['failed_attempt']
                and state.get('error') == record['error']):
            result.append(key)
    return sorted(result)
