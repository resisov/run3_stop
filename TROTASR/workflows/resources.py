"""Process identity and account-wide worker admission."""
import os
import subprocess
from pathlib import Path


def process_identity(pid):
    """Kernel start time prevents adopting an unrelated reused PID."""
    if not isinstance(pid, int) or pid < 2:
        return None
    try:
        folder = Path('/proc')/str(pid)
        fields = (folder/'stat').read_text().split(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=int(fields[19]),
                    arguments=[x.decode() for x in (folder/'cmdline').read_bytes().split(b'\0') if x])
    except FileNotFoundError:
        return None


def external_process_slots(rows, own_pid):
    ours = {own_pid}
    while True:
        more = {pid for pid, parent, _ in rows if parent in ours}
        if more <= ours:
            break
        ours |= more
    return sum(pid not in ours and (comm.startswith(('python', 'root', 'combine', 'xrdcp', 'rsync'))
               or comm in ('hadd', 'cp')) for pid, _, comm in rows)

def resources():
    raw = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid=,ppid=,comm='], text=True)
    rows = [(int(p), int(pp), c) for p, pp, c in (s.split(None, 2) for s in raw.splitlines())]
    external = external_process_slots(rows, os.getpid())
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
