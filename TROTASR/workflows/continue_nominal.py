"""One-shot handoff to the tested 130-file repair after the original run drains.

No histogram implementation or physics definition lives here. Never kill a
worker, promote twice, or restart after an interrupted journal without review.
The existing chain, retry limit, resource admission and products runner are used.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256
from TROTASR.workflows.nominal_campaign import THREAD_ENV
from TROTASR.workflows.promote_normalization_recovery import run as promote
from TROTASR.workflows.diagnostic_common import live_resources


def process_identity(pid):
    if not isinstance(pid, int) or pid < 2:
        return None
    try:
        folder = Path('/proc')/str(pid)
        fields = (folder/'stat').read_text().split(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=int(fields[19]),
                    command=(folder/'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError:
        return None


def originals_alive(records):
    active = []
    for original in records:
        actual = process_identity(original['pid'])
        if actual is not None:
            if actual != original:
                raise RuntimeError('Original PID was reused or command changed')
            active.append(actual['pid'])
    return active


def wait_until(predicate, deadline, pause=30):
    while not predicate():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Bounded recovery wait exhausted; inspect before retry')
        time.sleep(min(pause, remaining))


def launch(script, arguments, log):
    with log.open('a') as stream:
        return subprocess.Popen([sys.executable, '-u', str(ROOT/'workflows'/script), *arguments],
            cwd=ROOT, env=dict(os.environ, **THREAD_ENV), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)


def run(campaign, timeout):
    if sys.platform != 'linux' or not 60 <= timeout <= 21600:
        raise ValueError('hep2 only; bounded wait must be 60--21600 seconds')
    campaign = internal_path(campaign)
    products = ROOT/'stats'/campaign.name
    recovery = campaign/'recovery/missing_normalization_20260923'
    path = campaign/'continuation_state.json'
    deadline = time.monotonic() + timeout
    with (campaign/'continuation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise FileExistsError('Existing recovery handoff journal; never resubmit blindly')
        audit = read_json(recovery/'audit.json')
        expected = sorted(k for y in audit['years'].values() for k in y['missing_keys'])
        if len(expected) != 130 or sha256(campaign/'plan.json') != audit['plan_sha256']:
            raise ValueError('Not the audited original 130-file recovery')
        if (recovery/'promotion.json').exists() or (recovery/'checkpoint_before_promotion').exists():
            raise ValueError('Promotion was already attempted; inspect it')
        sources = [campaign/'state.json', campaign/'chain_state.json', products/'products_state.json']
        scripts = ['nominal_campaign.py', 'run_nominal_chain.py', 'run_nominal_products.py']
        originals = []
        original_pids = []
        for source, script in zip(sources, scripts):
            record = read_json(source)
            original_pids.append(record.get('pid'))
            identity = process_identity(record.get('pid'))
            if identity is not None:
                if str(ROOT/'workflows'/script) not in identity['command'] and ('workflows/'+script) not in identity['command']:
                    raise ValueError('Unexpected original controller: '+script)
                originals.append(identity)
        state = dict(status='waiting_for_original_controllers', pid=os.getpid(), started=time.time(),
            source_sha256=sha256(Path(__file__)), timeout_seconds=timeout,
            original_controllers=originals, retry_files=130, children={},
            histogram_recovery_complete=False, full_workflow_complete=False)
        def save():
            state['updated'] = time.time()
            write_json(path, state)
        try:
            save()
            def drained():
                records = [read_json(p) for p in sources]
                if [r.get('pid') for r in records] != original_pids:
                    raise RuntimeError('Another controller replaced the original run')
                state['active_original_pids'] = originals_alive(originals)
                state['original_progress'] = {k: records[0].get(k) for k in ('completed', 'active', 'pending')}
                if sorted(records[0].get('failed', [])) != expected:
                    raise ValueError('Failure set changed; do not promote automatically')
                save()
                return not state['active_original_pids']
            wait_until(drained, deadline)
            # Reserve room for this process, the chain, its histogram controller,
            # and the products controller. Their worker admission counts all users' slots.
            wait_until(lambda: live_resources()[0] <= 96 and live_resources()[1] >= 36*1024**3, deadline)
            state['status'] = 'promoting_once'; save()
            if promote(campaign, apply=True) != 0:
                raise RuntimeError('Promotion did not finish; no automatic second attempt')
            promotion = read_json(recovery/'promotion.json')
            if promotion['status'] != 'promoted_pending_controller_restart' or promotion['retry_keys'] != expected:
                raise ValueError('Unexpected promotion receipt')
            state['promotion_sha256'] = sha256(recovery/'promotion.json')
            state['status'] = 'starting_existing_chain'; save()
            chain = launch('run_nominal_chain.py', ['--campaign', str(campaign)], campaign/'logs/chain.log')
            state['children']['chain'] = dict(pid=chain.pid, command='run_nominal_chain.py'); save()
            def chain_ready():
                if chain.poll() is not None:
                    raise RuntimeError('Resumed chain exited during startup; inspect its log')
                record = read_json(campaign/'chain_state.json')
                return record.get('pid') == chain.pid and record.get('status') == 'running'
            wait_until(chain_ready, min(deadline, time.monotonic()+60), pause=1)
            state['status'] = 'starting_existing_products'; save()
            product = launch('run_nominal_products.py', ['--campaign', str(campaign), '--output', str(products)],
                             campaign/'logs/products.log')
            state['children']['products'] = dict(pid=product.pid, command='run_nominal_products.py'); save()
            def products_ready():
                if product.poll() is not None:
                    raise RuntimeError('Resumed products exited during startup; inspect its log')
                p = products/'products_state.json'
                return p.exists() and read_json(p).get('pid') == product.pid
            wait_until(products_ready, min(deadline, time.monotonic()+60), pause=1)
            state.update(status='handed_off_to_existing_controllers', finished=time.time())
            save()
            return 0
        except Exception as error:
            state.update(status='needs_attention_no_automatic_retry', error=repr(error),
                         traceback=traceback.format_exc(), finished=time.time())
            save()
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=14400)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.campaign, arguments.timeout))
