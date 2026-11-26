"""Per-file histogram residency and measured worker admission.

This import-only module is separate from the shared process/fit resource
runtime, so histogram admission updates do not replace that live-imported code.
"""
import os
from pathlib import Path


def histogram_residents(inventory, indices, *, proc=Path('/proc')):
    """Credit only the anonymous RSS of our own, still-identical file workers."""
    if not indices:
        return {}
    children = set()
    for file in (proc / str(os.getpid()) / 'task').glob('*/children'):
        try:
            children.update(map(int, file.read_text().split()))
        except FileNotFoundError:
            pass
    resident = {}
    for pid in children:
        worker = proc / str(pid)
        try:
            before = (worker/'stat').read_text().rsplit(') ', 1)[1].split()
            args = (worker/'cmdline').read_bytes().split(b'\0')
            if b'histograms' not in args or b'--task-index' not in args or b'--inventory' not in args:
                continue
            if args[args.index(b'--inventory')+1] != os.fsencode(inventory):
                continue
            index = int(args[args.index(b'--task-index')+1])
            if index not in indices:
                continue
            anon = next((int(s.split()[1])*1024 for s in (worker/'status').read_text().splitlines()
                         if s.startswith('RssAnon:')), 0)
            after = (worker/'stat').read_text().rsplit(') ', 1)[1].split()
            if before[19] == after[19] and after[0] != 'Z':
                resident[index] = anon
        except (OSError, IndexError, ValueError):
            continue  # Unknown or already finished: no memory credit.
    return resident


class HistogramAdmission:
    """Per-inventory peak estimates for ordinary histogram workers (not hard limits)."""

    def __init__(self, ceiling, default_bytes=6 << 30):
        self.stage_ceiling = ceiling
        self.default = default_bytes
        self.samples = {}
        self.failed = set()

    def observe(self, key, rss, vms=None, *, success=True):
        if key is None:
            return
        if not success:
            self.failed.add(key)  # never lower this class's estimate on a failed/smaller retry
        if type(rss) is int and rss > 0:
            n, peak = self.samples.get(key, (0, 0))
            self.samples[key] = (n+int(success), max(peak, rss))

    def reservation(self, key):
        n, peak = self.samples.get(key, (0, 0))
        padded = (peak*3 + 1)//2 + (512 << 20)
        return max(1 << 30 if n >= 6 and key not in self.failed else self.default, padded)

    def admit(self, memory, active, pending, *, account_slots, now, dispatch_limit):
        active = list(active)
        free = memory - (32 << 30) - sum(max(0, book - rss) for book, rss in active)
        slots = min(self.stage_ceiling, account_slots) - len(active)
        grants = []
        for key in pending:
            need = self.reservation(key)
            if len(grants) >= min(slots, dispatch_limit) or free < need:
                break
            grants.append(need)
            free -= need
        return grants


class JMEPhysicalAdmission(HistogramAdmission):
    """Book physical memory from RSS, without an address limit or sample-count gate."""
    def reservation(self, key):
        peak = self.samples.get(key, (0, 0))[1]
        if not peak:
            peak = max((p for _,p in self.samples.values()), default=0)
        return max(1 << 30, (peak*3 + 1)//2 + (512 << 20)) if peak else self.default


class DirectTopPtAdmission(HistogramAdmission):
    """Fill all requested top-pT slots without a worker-count ramp or sample gate."""
    worker_limit_bytes = 32 << 30

    def __init__(self, workers):
        super().__init__(workers)

    def reservation(self, key):
        if key is None or key in self.failed:
            return self.worker_limit_bytes
        # Single-chunk classes carry event and compressed-MiB upper bounds.
        # A booking is an estimate, not a new address-space limit.
        _, _, _, events, mib = key
        n, peak = self.samples.get(key, (0, 0))
        if n:
            return min(self.worker_limit_bytes, max(1 << 30, (peak * 3 + 1)//2 + (512 << 20)))
        return (2 if events <= 100_000 and mib <= 100 else 4) << 30


class RetainedJMEWorkers:
    """Drain only explicitly handed-over workers. Never signal or accept a PID as a receipt."""
    def __init__(self, inventory, tasks, records):
        import fcntl
        import stat
        self.inventory = inventory
        self.other_population = False
        origins = {str(row.get('source_inventory', inventory)) for row in records}
        if origins and origins != {str(inventory)}:
            from TROTASR.utils.paths import ROOT
            base = ROOT/'stats/run3/dataset_replay/top_pt'
            if (inventory != base/'existing_jme_weights_nominal_pass.json'
                    or origins != {str(base/'existing_jme_weights.json')}):
                raise ValueError('Unexpected inventory in TT handover')
            self.inventory = base/'existing_jme_weights.json'
            self.other_population = True
        self.rows = {}
        seen = set()
        for row in records:
            i = row['index']
            if (type(i) is not int or not 0 <= i < len(tasks) or i in self.rows
                    or not all(type(row[k]) is int and row[k] > 0 for k in ('pid','ticks','booking'))):
                raise ValueError('Invalid retained JME file-worker identity')
            streams = {}
            for name in ('stdout','stderr'):
                fd, inode = row[name]
                info = os.fstat(fd)
                if (fd < 3 or fd in seen or not stat.S_ISFIFO(info.st_mode) or info.st_ino != inode
                        or fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY):
                    raise ValueError('Invalid retained JME output pipe')
                seen.add(fd)
                os.set_inheritable(fd, False)
                os.set_blocking(fd, False)
                streams[name] = fd
            self.rows[i] = dict(row, streams=streams, tail=b'')

    def close(self):
        for row in self.rows.values():
            for fd in row['streams'].values():
                os.close(fd)
            row['streams'].clear()

    def poll(self):
        finished, bookings = {}, []
        for i,row in list(self.rows.items()):
            for name,fd in list(row['streams'].items()):
                while True:
                    try:
                        data = os.read(fd, 65536)
                    except BlockingIOError:
                        break
                    if not data:
                        os.close(fd)
                        del row['streams'][name]
                        break
                    if name == 'stderr':
                        row['tail'] = (row['tail']+data)[-2000:]
            proc = Path('/proc')/str(row['pid'])
            def identity():
                st = (proc/'stat').read_text().rsplit(') ',1)[1].split()
                return int(st[19]) == row['ticks'] and st[0] not in ('Z','X')
            rss = 0
            try:
                living = identity()
                if living:
                    args = (proc/'cmdline').read_bytes().split(b'\0')
                    if (proc.stat().st_uid == os.getuid() and b'--inventory' in args and b'--task-index' in args
                            and args[args.index(b'--inventory')+1] == os.fsencode(self.inventory)
                            and args[args.index(b'--task-index')+1] == str(i).encode()):
                        rss = next((int(l.split()[1])*1024 for l in (proc/'status').read_text().splitlines()
                                    if l.startswith('RssAnon:')),0)
                    living = identity()
            except (FileNotFoundError,ProcessLookupError):
                living = False
            if living or row['streams']:
                bookings.append((row['booking'],rss if living else 0))
            else:
                finished[i] = row['tail'].decode(errors='replace')
                del self.rows[i]
        return finished,bookings
