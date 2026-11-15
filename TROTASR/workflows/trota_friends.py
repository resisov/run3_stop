"""Backfill row-aligned, corrected nominal TROTA friend ROOTs for accepted MC."""
import argparse
from collections import Counter, deque
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import shutil
import signal
import time

from TROTASR.utils.io import write_json
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.workflows.resources import memory_usage
from TROTASR.workflows.resources import process_identity

OUTPUT = ROOT / "inputs/friends/trota_nominal"
SCHEMA = "trotasr_trota_friends_state_v1"
GIB = 2**30
DISK_RESERVE = 80 * GIB
MEMORY_RESERVE = 32 * GIB
MEMORY_EMERGENCY = 24 * GIB
WORKER_BYTES = 6 * GIB
SHARDS_PER_WORKER = 8
DEFAULT_WORKERS = 4
MAX_WORKERS = 16


def friend_path(task):
    path = OUTPUT / str(task["year"]) / Path(task["input"]).name
    if internal_path(path) != path:
        raise ValueError("Noncanonical TROTA friend path: " + str(path))
    return path


def _disk_budget(task):
    # Budget compressed tag bytes, not another copy of the input ROOT. Actual
    # disk reserve is independently checked inside the writer and controller.
    return 2**20 + 400 * task["entries"]


def _worker(pipe, parent):
    # Linux sends this only to our own worker if its controller dies. Check
    # PPID after installing the guard to cover a death before the guard ran.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "Could not protect TROTA friend worker")
    if os.getppid() != parent:
        return
    from TROTASR.utils.trota_friend import write_friend
    from TROTASR.workflows.validation_regions import _task_source

    with pipe:
        for _ in range(SHARDS_PER_WORKER):
            try:
                task = pipe.recv()
            except EOFError:
                return
            if task is None:
                return
            try:
                report = write_friend(task, _task_source(task), friend_path(task))
            except Exception as exc:
                disk_stop = isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT)
                pipe.send((None, f"{type(exc).__name__}: {exc}"[:1000], disk_stop))
            else:
                pipe.send((report["events"], None, False))


@dataclass(eq=False)
class _Slot:
    process: object
    pipe: object
    era: str
    task: dict = None
    completed: int = 0
    retiring: bool = False


def _start(context, era):
    parent, child = context.Pipe()
    process = context.Process(target=_worker, args=(child, os.getpid()))
    try:
        process.start()
    except BaseException:
        parent.close()
        raise
    finally:
        child.close()
    return _Slot(process, parent, era)


def _retire(slot):
    if not slot.retiring:
        slot.retiring = True
        try:
            slot.pipe.send(None)
        except (EOFError, OSError):
            pass


def _reap(slot):
    slot.process.join()
    slot.process.close()
    slot.pipe.close()


def _cached(task):
    from TROTASR.utils.trota_friend import validate_friend
    from TROTASR.workflows.validation_regions import _task_source

    target = friend_path(task)
    if not target.exists():
        return None
    return validate_friend(task, _task_source(task), target)


def _ordered_tasks(all_tasks, pilot):
    from TROTASR.workflows.validation_regions import _pilot

    if pilot:
        return {era: _pilot(tasks) for era, tasks in all_tasks.items()}
    return all_tasks


def run(*, workers=DEFAULT_WORKERS, pilot=False):
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"TROTA friend workers must be in [1, {MAX_WORKERS}]")
    if not Path("/proc/meminfo").is_file():
        raise RuntimeError("Use the existing hep2 numerical runtime for TROTA friends")
    from TROTASR.workflows.validation_regions import (
        _available_memory, _resource_room, inventory,
    )

    if internal_path(OUTPUT) != OUTPUT:
        raise ValueError("TROTA friend directory is not canonical")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        all_tasks = inventory()
        selected = _ordered_tasks(all_tasks, pilot)
        coverage = {
            era: dict(total=len(tasks), completed=0, failed=0,
                      events_total=sum(t["entries"] for t in tasks), events_completed=0)
            for era, tasks in selected.items()
        }
        total = sum(map(len, selected.values()))
        full_total = sum(map(len, all_tasks.values()))
        identity = process_identity(os.getpid())
        pending, active, failures = deque(), [], []
        done = checked = peak = 0
        last_snapshot = 0.0
        started = time.time()

        def failed(task, error, phase):
            failures.append(dict(input=task["input"], error=error, phase=phase,
                                 entries=task["entries"]))
            coverage[str(task["year"])]["failed"] += 1

        def completed(task, events):
            nonlocal done
            if events != task["entries"]:
                raise ValueError("Friend writer returned a nonmatching event count")
            done += 1
            counts = coverage[str(task["year"])]
            counts["completed"] += 1
            counts["events_completed"] += events

        def snapshot(status, *, force=False, reason=None):
            nonlocal last_snapshot
            now = time.monotonic()
            if not force and now - last_snapshot < 15:
                return
            busy = [slot.task for slot in active if slot.task is not None]
            write_json(OUTPUT / "state.json", dict(
                schema=SCHEMA, status=status, reason=reason, pilot=pilot,
                pid=os.getpid(), start_ticks=identity["start_ticks"], started=started,
                updated=time.time(), total=total, full_total=full_total, scanned=checked,
                completed=done, active=len(busy), pending=total-done-len(failures)-len(busy),
                failures=failures, pool_workers=len(active), workers=workers, coverage=coverage,
                active_eras=dict(Counter(str(t["year"]) for t in busy)),
                pool_peak_rss_gib=peak/GIB, mc_only=True,
                source_nanoaod_read=False, observations_read=False, source_root_changed=False))
            last_snapshot = now

        snapshot("checking_friends", force=True)
        for era, tasks in selected.items():
            for task in tasks:
                try:
                    existing = _cached(task)
                except Exception as exc:
                    failed(task, f"{type(exc).__name__}: {exc}"[:1000], "existing_friend_check")
                else:
                    if existing is None:
                        pending.append(task)
                    else:
                        completed(task, existing["events"])
                checked += 1
                snapshot("checking_friends")

        guard = None
        context = mp.get_context("spawn")
        try:
            idle_since = None
            snapshot("running", force=True)
            while pending or active:
                available = _available_memory()
                disk = shutil.disk_usage(OUTPUT).free
                if available < MEMORY_EMERGENCY or disk < DISK_RESERVE:
                    guard = "memory_stopped" if available < MEMORY_EMERGENCY else "disk_stopped"
                    break
                rss = [memory_usage(slot.process.pid) for slot in active]
                peak = max([peak, *(m["VmHWM"] for m in rss)])
                allowance = max(WORKER_BYTES, (peak * 6 + 4) // 5)
                disk_budget = disk - DISK_RESERVE - sum(
                    _disk_budget(w.task) for w in active if w.task is not None)
                # Allow SMT headroom for this stage. Load already includes our
                # active shards; the resulting target includes them as well.
                cpu_budget = 3 * len(os.sched_getaffinity(0)) // 4
                busy = sum(slot.task is not None for slot in active)
                target = min(workers, max(1, busy + int(cpu_budget - os.getloadavg()[0])))
                memory_budget = 0
                if pending and guard is None:
                    slots, memory_budget = _resource_room(
                        reserve_vr_data=True, reserve_bytes=MEMORY_RESERVE)
                    memory_budget -= sum(max(0, allowance - m["VmRSS"]) for m in rss)
                    # Leave a spare slot for multiprocessing's resource tracker.
                    slots -= 1
                    while pending:
                        task = pending[0]
                        era = str(task["year"])
                        required_disk = _disk_budget(task)
                        if required_disk > disk_budget:
                            break
                        slot = next((w for w in active if w.era == era and w.task is None
                                     and not w.retiring), None)
                        if slot is None:
                            if (len(active) >= target or slots <= 0 or memory_budget < allowance):
                                break
                            slot = _start(context, era)
                            active.append(slot)
                            slots -= 1
                            memory_budget -= allowance
                        if memory_budget < 0 or sum(w.task is not None for w in active) >= target:
                            break
                        slot.task = pending.popleft()
                        try:
                            slot.pipe.send(slot.task)
                        except (EOFError, OSError):
                            _retire(slot)
                        disk_budget -= required_disk
                        last_snapshot = 0.0
                next_era = str(pending[0]["year"]) if pending else None
                for slot in active:
                    if slot.task is None and (slot.era != next_era or guard is not None
                                              or not pending or memory_budget < 0
                                              or disk_budget < _disk_budget(pending[0])):
                        _retire(slot)
                if not active and pending and guard is None:
                    now = time.monotonic()
                    idle_since = now if idle_since is None else idle_since
                    if now - idle_since >= 60:
                        guard = "waiting_for_headroom"
                    snapshot("waiting_for_headroom")
                elif active:
                    idle_since = None
                    snapshot("draining" if guard else "running", reason=guard)
                if not active:
                    if guard is not None or not pending:
                        break
                    time.sleep(2)
                    continue
                watches = [w.process.sentinel for w in active]
                watches.extend(w.pipe for w in active if w.task is not None)
                ready = set(wait(watches, timeout=2))
                for slot in active[:]:
                    exited = slot.process.sentinel in ready
                    if slot.task is not None and (slot.pipe in ready or (exited and slot.pipe.poll())):
                        try:
                            events, error, disk_stop = slot.pipe.recv()
                        except (EOFError, OSError):
                            _retire(slot)
                        else:
                            if error is None:
                                completed(slot.task, events)
                            elif disk_stop:
                                pending.appendleft(slot.task)
                                guard = "disk_stopped"
                            else:
                                failed(slot.task, error, "writer")
                            slot.task = None
                            slot.completed += 1
                            if slot.completed >= SHARDS_PER_WORKER:
                                slot.retiring = True
                            last_snapshot = 0.0
                    if exited:
                        if slot.task is not None:
                            try:
                                existing = _cached(slot.task)
                            except Exception as exc:
                                failed(slot.task, f"{type(exc).__name__}: {exc}"[:1000], "worker_exit")
                            else:
                                if existing is None:
                                    failed(slot.task, f"Worker exited {slot.process.exitcode} without a friend ROOT",
                                           "worker_exit")
                                else:
                                    completed(slot.task, existing["events"])
                        _reap(slot)
                        active.remove(slot)
                        last_snapshot = 0.0
        except KeyboardInterrupt:
            guard = "interrupted"
            raise
        except BaseException:
            guard = "controller_error"
            raise
        finally:
            running_tasks = [slot.task for slot in active if slot.task is not None]
            for slot in active:
                # Only our exact multiprocessing children, never PIDs read
                # from another workflow state or from a global process scan.
                if slot.process.is_alive():
                    slot.process.kill()
            for slot in active:
                _reap(slot)
            active.clear()
            for task in running_tasks:
                try:
                    existing = _cached(task)
                except Exception as exc:
                    failed(task, f"{type(exc).__name__}: {exc}"[:1000], "interrupted_friend_check")
                else:
                    if existing is not None:
                        completed(task, existing["events"])
            write_json(OUTPUT / "bad_files.json", dict(
                status="requires_investigation" if failures else "none", pilot=pilot,
                files=failures, missing_events=sum(row["entries"] for row in failures),
                source_corruption_asserted=False, complete_friend_coverage=done == total))
            status = guard or ("needs_attention" if done != total else
                               "pilot_complete" if pilot else "complete")
            snapshot(status, force=True, reason=guard)
        return int(guard is not None or done != total)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m TROTASR trota-friends", description=__doc__,
                                     allow_abbrev=False)
    parser.add_argument("--workers", type=int, choices=range(1, MAX_WORKERS+1), default=DEFAULT_WORKERS)
    parser.add_argument("--pilot", action="store_true",
                        help="Validate one accepted background and signal shard per era")
    args = parser.parse_args(argv)
    # The top-level CLI uses runpy; spawn must see a stable importable module.
    from TROTASR.workflows.trota_friends import run as stage
    return stage(workers=args.workers, pilot=args.pilot)


if __name__ == "__main__":
    raise SystemExit(main())
