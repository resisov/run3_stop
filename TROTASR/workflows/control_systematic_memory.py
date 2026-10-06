"""One-time, identity-checked reduction of this campaign's orphan workers."""
import fcntl
import json
import os
from pathlib import Path
import signal
import time
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json, write_json
from TROTASR.workflows.normalization_recovery import process_identity


def main():
    campaign = ROOT/'campaigns/systematics_20260924'
    report = campaign/'memory_control.json'
    if report.exists():
        raise RuntimeError('Reduction already recorded; inspect before another action')
    with (campaign/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = read_json(campaign/'controller.json')
        ident = process_identity(controller['pid'])
        if ident and ident['start_ticks'] == controller['start_ticks']:
            raise RuntimeError('Controller still alive; no worker changes made')
        states = {p.stem: read_json(p) for p in (campaign/'state').glob('*.json')}
        active = []
        for key, state in states.items():
            if state['status'] != 'running':
                continue
            ident = process_identity(state['pid'])
            if not ident:
                continue
            expected = [str(ROOT/'workflows/nominal_campaign.py'), 'worker', '--campaign',
                        str(campaign), '--key', key, '--attempt', str(state['attempt'])]
            if ident['arguments'][-len(expected):] != expected:
                raise RuntimeError('Unexpected worker identity: '+key)
            active.append(dict(key=key, state=state, identity=ident))
        # Preserve the oldest work first, with corrected second attempts first.
        active.sort(key=lambda r: (-r['state']['attempt'], r['state']['started'], r['key']))
        audit = dict(status='reducing', started=time.time(), target_workers=32,
                     worker_cap=50, previous_controller=controller,
                     before_active=len(active), retained=active[:32], interrupted=[],
                     plan_before=read_json(campaign/'plan.json'), states_before=states,
                     completed_outputs_deleted=0, inputs_modified=0)
        write_json(report, audit)
        for record in active[32:]:
            ident = record['identity']
            if process_identity(ident['pid']) != ident:
                continue
            receipt = campaign/'history'/(record['key']+'.attempt%s.json'%record['state']['attempt'])
            if receipt.exists() and read_json(receipt).get('status') == 'complete':
                continue
            os.kill(ident['pid'], signal.SIGTERM)
            record['signal'] = 'SIGTERM'
            record['time'] = time.time()
            audit['interrupted'].append(record)
            write_json(report, audit)
        for _ in range(100):
            alive = [r for r in audit['interrupted'] if process_identity(r['identity']['pid']) == r['identity']]
            if not alive:
                break
            time.sleep(.1)
        audit.update(status='reduced' if not alive else 'termination_pending', finished=time.time(),
                     interrupted_still_alive=[r['key'] for r in alive])
        write_json(report, audit)
        print(json.dumps({k: audit[k] for k in ('status','before_active','target_workers','worker_cap',
                                               'interrupted_still_alive')}))
        print('Interrupted unfinished workers:', len(audit['interrupted']))


if __name__ == '__main__':
    main()
