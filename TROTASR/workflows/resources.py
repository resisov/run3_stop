"""Process identity and account-wide worker admission."""
import os
import stat
import subprocess
from pathlib import Path


_PROC_LOCKS = Path('/proc/locks')


def holds_flock(pid, path):
    """Check one Linux exclusive flock on the current regular-file inode."""
    if type(pid) is not int or pid < 2:
        return False
    path = Path(path)
    try:
        before = path.stat()  # Follows the same path resolution as open().
        if not stat.S_ISREG(before.st_mode):
            return False
        rows = _PROC_LOCKS.read_text().splitlines()
        after = path.stat()
    except (OSError, UnicodeError):
        return False
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        return False
    identity = os.major(after.st_dev), os.minor(after.st_dev), after.st_ino
    owners = []
    for row in rows:
        fields = row.split()
        # A waiting request has a separate '->' field and does not hold a lock.
        if (len(fields) != 8 or not fields[0].endswith(':') or not fields[0][:-1].isdecimal()
                or fields[1:4] != ['FLOCK', 'ADVISORY', 'WRITE'] or fields[6:] != ['0', 'EOF']):
            continue
        try:
            major, minor, inode = fields[5].split(':')
            if (int(major, 16), int(minor, 16), int(inode, 10)) == identity:
                owners.append(int(fields[4], 10))
        except ValueError:
            continue
    return owners == [pid]


def process_identity(pid):
    """Kernel start time prevents adopting an unrelated reused PID."""
    if not isinstance(pid, int) or pid < 2:
        return None
    try:
        folder = Path('/proc')/str(pid)
        fields = (folder/'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=int(fields[19]),
                    arguments=[x.decode() for x in (folder/'cmdline').read_bytes().split(b'\0') if x])
    except FileNotFoundError:
        return None


def external_process_slots(rows, own_pid, *, all_account=False):
    ours = {own_pid}
    while True:
        more = {pid for pid, parent, _ in rows if parent in ours}
        if more <= ours:
            break
        ours |= more
    return sum(pid not in ours and (all_account or comm.startswith(('python', 'root', 'combine', 'text2workspace', 'xrdcp', 'rsync'))
               or comm in ('hadd', 'cp')) for pid, _, comm in rows)

def top_pt_external_slots(rows, own_pid, *, argv_for, sleeping=None):
    """Count an identified waiting QCD pool parent and its one compute child once."""
    from functools import lru_cache
    from collections import defaultdict
    children = defaultdict(list)
    known = {pid for pid, _, _ in rows}
    for pid, parent, _ in rows:
        children[parent].append(pid)
    ours, todo = set(), [own_pid]
    while todo:
        pid = todo.pop()
        if pid not in ours:
            ours.add(pid)
            todo.extend(children[pid])

    @lru_cache(None)
    def role(pid):
        try:
            a = tuple(argv_for(pid) or ())
        except (OSError, UnicodeError):
            return None
        if not a or not Path(a[0]).name.startswith('python'):
            return None
        args = a[1:]
        while args and args[0] in ('-u', '-B', '-s'):
            args = args[1:]
        if args and Path(args[0]).name == 'qcd_apply.py' and '--workers' in args[1:]:
            return 'qcd'
        if (len(args) == 3 and args[0] == '-c' and args[2] == '--multiprocessing-fork'
                and args[1].startswith('from multiprocessing.spawn import spawn_main; spawn_main(')):
            return 'spawn'
        return None

    count = sum(pid not in ours for pid, _, _ in rows)
    for pid, parent, comm in rows:
        if (pid in ours or parent not in known or not comm.startswith('python') or len(children[pid]) != 1
                or (sleeping is not None and pid not in sleeping)):
            continue
        if role(parent) == 'qcd' and role(pid) == role(children[pid][0]) == 'spawn':
            count -= 1
    return count


def _process_argv(pid):
    return [a.decode() for a in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if a]


def resources(*, all_account=False, top_pt_pool=False):
    raw = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid=,ppid=,stat=,comm='], text=True)
    snapshot = [s.split(None, 3) for s in raw.splitlines()]
    rows = [(int(p), int(pp), c) for p, pp, state, c in snapshot if not state.startswith('Z')]
    if top_pt_pool:
        sleeping = {int(p) for p, _, state, _ in snapshot if state.startswith('S')}
        external = top_pt_external_slots(rows, os.getpid(), argv_for=_process_argv, sleeping=sleeping)
    else:
        external = external_process_slots(rows, os.getpid(), all_account=all_account)
    memory = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                  if line.startswith('MemAvailable:'))
    return external, memory

def capacity(plan, external, memory, active):
    # MemAvailable already excludes allocations of active children. Reserve
    # additional memory only for newly admitted workers, not a second time.
    by_memory = active + max(0, (memory-plan['minimum_available_memory_bytes'])//plan['admission_bytes_per_worker'])
    return max(0, min(plan['max_workers'], plan['aggregate_slots']-1-external, by_memory))


def memory_usage(pid):
    try:
        fields = dict(line.split(':', 1) for line in (Path('/proc')/str(pid)/'status').read_text().splitlines() if ':' in line)
        return {key: int(fields.get(key, '0 kB').split()[0])*1024 for key in ('VmRSS', 'VmHWM')}
    except FileNotFoundError:
        return {'VmRSS': 0, 'VmHWM': 0}
