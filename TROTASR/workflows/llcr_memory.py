"""Use measured per-file memory for opt-in LLCR worker admission.

Reservations for calibrated files are estimates, not memory limits.
The dispatcher still applies the worker address-space and host/account guards.
"""

from dataclasses import dataclass
from itertools import islice, zip_longest
from pathlib import Path
from stat import S_ISREG
from typing import Optional


MIB = 1 << 20
GIB = 1 << 30
HOST_RESERVE = 32 * GIB
BASELINE_WORKERS = 6
RAMP_STEPS = (6, 8, 12, 16, 24, 32, 40, 48)
MIN_SAMPLES = 12
RAMP_SECONDS = 60
RAMP_COMPLETIONS = 4
SMALL_FILE_EVENTS = 100_000
SMALL_FILE_BYTES = 100 * MIB
MEASURED_FILE_EVENTS = 1_000_000
MEASURED_FILE_BYTES = 1024 * MIB
IMMEDIATE_MAX_WORKERS = 72
COMPLETED_REDUCTION_SAMPLES = 3
Bucket = tuple[str, str, str, int, int]


def small_file_bucket(mode, era, kind, entries, chunk_size, size_bytes) -> Optional[Bucket]:
    """Classify only known small files; never learn across execution modes or eras."""
    if (mode not in ('jme', 'nonjme') or era not in
            ('2022', '2022EE', '2023', '2023BPix', '2024', '2025')
            or kind not in ('background', 'data') or (mode == 'nonjme' and kind == 'data')
            or type(entries) is not int or entries < 0
            or type(chunk_size) is not int or chunk_size <= 0
            or type(size_bytes) is not int or size_bytes <= 0):
        return None
    # The stored-endpoint path can retain file-local data outside a chunk.
    # Do not classify a multi-chunk file as a small single-chunk file.
    if entries > chunk_size or entries > SMALL_FILE_EVENTS or size_bytes > SMALL_FILE_BYTES:
        return None
    # Keep genuinely empty shards in a separate class; they cannot train a
    # nonempty worker's reservation.
    events = next(n for n in (0, 1_000, 10_000, 40_000, SMALL_FILE_EVENTS) if entries <= n)
    compressed = next(n for n in (4, 16, 64, 100) if size_bytes <= n * MIB)
    return mode, era, kind, events, compressed


def measured_file_bucket(mode, era, kind, entries, chunk_size, size_bytes) -> Optional[Bucket]:
    """Extend measured admission to known single-chunk files up to 1 GiB.

    Retain the fine small-file classes. Larger files use separate event and
    compressed-size upper bounds; evidence never crosses those bounds, mode,
    era or input kind. This classifier changes neither the chunk size nor the
    hard address-space limit applied by the dispatcher.
    """
    small = small_file_bucket(mode, era, kind, entries, chunk_size, size_bytes)
    if small is not None:
        return small
    if (mode not in ('jme', 'nonjme') or era not in
            ('2022', '2022EE', '2023', '2023BPix', '2024', '2025')
            or kind not in ('background', 'data') or (mode == 'nonjme' and kind == 'data')
            or type(entries) is not int or entries < 0
            or type(chunk_size) is not int or chunk_size <= 0
            or type(size_bytes) is not int or size_bytes <= 0):
        return None
    if (entries > chunk_size or entries > MEASURED_FILE_EVENTS
            or size_bytes > MEASURED_FILE_BYTES):
        return None
    # An empty file cannot calibrate the estimate for a nonempty worker.
    events = next(n for n in (0, 250_000, 500_000, MEASURED_FILE_EVENTS) if entries <= n)
    compressed = next(n for n in (256, 512, 1024) if size_bytes <= n * MIB)
    return mode, era, kind, events, compressed


def scale72_file_bucket(mode, era, kind, entries, chunk_size, size_bytes) -> Optional[Bucket]:
    """Use narrower measured classes for an explicitly enlarged shared pool.

    The known-small classes are unchanged. Other eligible single-chunk files
    use 100,000-event and 64-MiB upper bounds, with no sharing across mode, era
    or input kind. Unknown, multi-chunk and >1-GiB files stay unclassified.
    Only metadata is classified; event selection and ROOT reading do not change.
    """
    small = small_file_bucket(mode, era, kind, entries, chunk_size, size_bytes)
    if small is not None:
        return small
    if measured_file_bucket(mode, era, kind, entries, chunk_size, size_bytes) is None:
        return None
    event_step = 100_000
    byte_step = 64 * MIB
    events = ((entries + event_step - 1) // event_step) * event_step
    compressed = ((size_bytes + byte_step - 1) // byte_step) * 64
    return mode, era, kind, events, compressed


def small_handoff_frontier(documents, states, root, chunk_size, *, active=(), prefix=6):
    """Inspect the resumed queue after its active (mode, index) tasks drain.

    The decision requires small heads in both nonempty streams and a small
    interleaved prefix. Only those candidates are stat'ed; no ROOT is opened.
    The summary describes the frontier, not a completion-time estimate.
    """
    if type(prefix) is not int or prefix < 1:
        raise ValueError('The LLCR handoff prefix must be positive')
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('The LLCR handoff chunk must be positive')
    modes = ('jme', 'nonjme')
    active = set(active)
    root = Path(root)
    queues = {}
    heads = {}
    sizes = {}
    inspected = {}
    pending = {}
    above_event_limit = {}

    def inspect(row):
        mode, index, task = row
        key = mode, index
        if key not in inspected:
            source = root / task['input']
            if source not in sizes:
                try:
                    st = source.stat()
                    sizes[source] = st.st_size if S_ISREG(st.st_mode) else None
                except OSError:
                    sizes[source] = None
            size = sizes[source]
            entries = task.get('entries')
            bucket = small_file_bucket(mode, task.get('year'), task.get('vr_kind'),
                                       entries, chunk_size, size)
            inspected[key] = dict(mode=mode, index=index, entries=entries,
                                  size_bytes=size, small=bucket is not None)
        return inspected[key]

    for mode in modes:
        state = states[mode]
        tasks = documents[mode]['tasks']
        results = state.get('results') or {}
        replacement = state.get('replacement') or {}
        pending_replacements = set(replacement.get('pending_outputs', ()))
        queue = []
        for index, task in enumerate(tasks):
            result = results.get(task['input']) or {}
            complete = (task['output'] not in pending_replacements
                        and result.get('status') == 'complete'
                        and result.get('output') == task['output'])
            if not complete and (mode, index) not in active:
                queue.append((mode, index, task))
        queues[mode] = queue
        heads[mode] = inspect(queue[0]) if queue else None
        pending[mode] = len(queue)
        above_event_limit[mode] = sum(
            type(task.get('entries')) is int and task['entries'] > min(chunk_size, SMALL_FILE_EVENTS)
            for _, _, task in queue)

    interleaved = (row for pair in zip_longest(*(queues[mode] for mode in modes))
                   for row in pair if row is not None)
    front = [inspect(row) for row in islice(interleaved, prefix)]
    eligible = bool(front) and all(row['small'] for row in front) and all(
        row is None or row['small'] for row in heads.values())
    return eligible, dict(pending=pending, above_event_limit=above_event_limit,
                          heads=heads, prefix=front)


@dataclass
class _Evidence:
    successes: int = 0
    rss_peak: int = 0
    vm_peak: int = 0


class AdaptiveAdmission:
    """Track peak envelopes and admit only a memory-safe estimated queue prefix."""

    def __init__(self, max_workers, worker_limit_bytes, *, min_samples=MIN_SAMPLES):
        if type(max_workers) is not int or not 1 <= max_workers <= 48:
            raise ValueError('Adaptive LLCR needs a 1..48 shared worker ceiling')
        if type(worker_limit_bytes) is not int or worker_limit_bytes <= 0:
            raise ValueError('Adaptive LLCR needs a positive hard worker limit')
        if type(min_samples) is not int or min_samples < BASELINE_WORKERS:
            raise ValueError('Adaptive LLCR needs at least six completed samples per memory class')
        self.worker_limit_bytes = worker_limit_bytes
        self.min_samples = min_samples
        self._steps = tuple(n for n in RAMP_STEPS if n < max_workers) + (max_workers,)
        self._stage = 0
        self._stage_started = None
        self._stage_filled = False
        self._stage_completions = 0
        self._evidence = {}
        self._blocked_buckets = set()

    @property
    def stage_ceiling(self):
        return self._steps[self._stage]

    @property
    def calibrated_buckets(self):
        return sum(self.reservation(key) < self.worker_limit_bytes for key in self._evidence)

    @property
    def blocked_buckets(self):
        return tuple(sorted(self._blocked_buckets))

    def observe(self, bucket, peak_rss_bytes, peak_vm_bytes=None, *, success=True):
        """Learn from one completed file; clear confidence on a reported failure."""
        if bucket is None or bucket in self._blocked_buckets:
            return
        if not success:
            self._evidence.pop(bucket, None)
            self._blocked_buckets.add(bucket)
            self._stage = 0
            self._stage_started = None
            self._stage_filled = False
            self._stage_completions = 0
            return
        if type(peak_rss_bytes) is not int or peak_rss_bytes <= 0:
            return
        if peak_vm_bytes is not None and (type(peak_vm_bytes) is not int or peak_vm_bytes < peak_rss_bytes):
            return
        sample = self._evidence.setdefault(bucket, _Evidence())
        sample.successes += 1
        sample.rss_peak = max(sample.rss_peak, peak_rss_bytes)
        sample.vm_peak = max(sample.vm_peak, peak_vm_bytes or 0)
        if self._stage_filled and self.reservation(bucket) < self.worker_limit_bytes:
            self._stage_completions += 1

    def reservation(self, bucket):
        """Book a rounded upper estimate with a startup floor and safety margin."""
        sample = self._evidence.get(bucket)
        hard = self.worker_limit_bytes
        if (bucket in self._blocked_buckets or sample is None or sample.successes < self.min_samples
                or sample.vm_peak >= 7 * hard // 8):
            return hard
        # Use the maximum complete-child RSS, never the average or live RSS.
        # Add 50% and 512 MiB; 4 GiB covers startup even for tiny shards.
        quantum = 256 * MIB
        envelope = (3 * sample.rss_peak + 1) // 2 + 512 * MIB
        envelope = ((envelope + quantum - 1) // quantum) * quantum
        return min(hard, max(4 * GIB, envelope))

    def admit(self, memory_available_bytes, active_reservations, pending_buckets, *,
              account_slots, now, dispatch_limit=6):
        """Return booked bytes for the next worker(s), preserving queue order.

        Active reservations are pairs (booked total bytes, live RssAnon bytes).
        Credit only materialized anonymous RSS already absent from MemAvailable.
        Unmaterialized workers must use zero for the latter value.
        """
        active = list(active_reservations)
        pending_buckets = tuple(pending_buckets)
        if self._stage_started is None:
            self._stage_started = now
        self._stage_filled |= len(active) >= self.stage_ceiling
        room = memory_available_bytes - HOST_RESERVE
        for booked, resident in active:
            resident = max(0, resident)
            # A worker outgrowing its estimate is treated as uncalibrated
            # while it is live; don't use the stale estimate to admit another.
            if resident > booked:
                booked = self.worker_limit_bytes
            room -= max(0, booked - resident)
        if room <= 0:
            self._stage_started = now
            self._stage_filled = False
            self._stage_completions = 0
            return []
        if dispatch_limit <= 0:
            return []

        def prefix(ceiling):
            count = len(active)
            remaining = room
            slots = min(ceiling, account_slots) - count
            selected = []
            for bucket in pending_buckets:
                if slots <= 0:
                    break
                need = self.reservation(bucket)
                if count >= BASELINE_WORKERS and need == self.worker_limit_bytes:
                    break
                if need > remaining:
                    break
                remaining -= need
                slots -= 1
                count += 1
                selected.append(need)
                if len(selected) >= dispatch_limit:
                    break
            return selected

        def book(candidates):
            # A fast shard may finish before the next admission tick. The
            # dispatcher submits this entire prefix before awaiting a result.
            self._stage_filled |= len(active) + len(candidates) >= self.stage_ceiling
            return candidates

        if (self._stage + 1 < len(self._steps) and self._stage_filled
                and self._stage_completions >= RAMP_COMPLETIONS
                and now - self._stage_started >= RAMP_SECONDS):
            candidates = prefix(self._steps[self._stage + 1])
            if len(active) + len(candidates) > self.stage_ceiling:
                self._stage += 1
                self._stage_started = now
                self._stage_filled = False
                self._stage_completions = 0
                return book(candidates)
        return book(prefix(self.stage_ceiling))


class ImmediateAdmission:
    """Admit an explicit target using the current run's live RSS envelope.

    This opt-in policy supports an explicit ceiling up to 72 workers, subject
    to actual memory and account room. Its bootstrap is a live high-water mark,
    not a completed-file peak or a hard upper bound. Above twelve workers,
    three successful full-worker RSS peaks from the exact class may replace
    that global bootstrap. Full-worker peaks include process startup. Observed
    maxima never decay; live peaks can only raise the class estimate. The
    per-worker address-space limit is unchanged. Unknown/heavy files retain
    the full hard-limit reservation. The explicitly requested 48-worker RSS
    target instead uses one complete exact-class sample with additive padding;
    it is not the default policy for other targets.
    """

    def __init__(self, max_workers, worker_limit_bytes, *, bootstrap_peak_rss_bytes,
                 target_rss=False):
        if type(max_workers) is not int or not 1 <= max_workers <= IMMEDIATE_MAX_WORKERS:
            raise ValueError('Immediate LLCR needs a 1..72 shared worker ceiling')
        if type(worker_limit_bytes) is not int or worker_limit_bytes <= 0:
            raise ValueError('Immediate LLCR needs a positive hard worker limit')
        if (type(bootstrap_peak_rss_bytes) is not int
                or not 0 < bootstrap_peak_rss_bytes <= worker_limit_bytes):
            raise ValueError('Immediate LLCR needs a positive observed live RSS peak within the hard limit')
        self.worker_limit_bytes = worker_limit_bytes
        self.bootstrap_peak_rss_bytes = bootstrap_peak_rss_bytes
        if type(target_rss) is not bool or (target_rss and max_workers != 48):
            raise ValueError('Measured RSS target admission requires the explicit 48-worker target')
        self.target_rss = target_rss
        self.required_completed_samples = 1 if target_rss else COMPLETED_REDUCTION_SAMPLES
        self.stage_ceiling = max_workers
        self._completed_reduction = max_workers > 12
        self._rss_peaks = {}
        self._vm_peaks = {}
        self._completed_buckets = set()
        self._completed_counts = {}
        self._blocked_buckets = set()

    @property
    def calibrated_buckets(self):
        # Bootstrap and live updates must not be counted as completed evidence.
        return sum(self.reservation(key) < self.worker_limit_bytes
                   and (not self._completed_reduction or self._completed_counts.get(key, 0)
                        >= self.required_completed_samples)
                   for key in self._completed_buckets)

    @property
    def blocked_buckets(self):
        return tuple(sorted(self._blocked_buckets))

    def _record_peak(self, bucket, peak_rss_bytes, peak_vm_bytes=None):
        if bucket is None or type(peak_rss_bytes) is not int or peak_rss_bytes <= 0:
            return False
        if peak_vm_bytes is not None and (
                type(peak_vm_bytes) is not int or peak_vm_bytes < peak_rss_bytes):
            return False
        self._rss_peaks[bucket] = max(self._rss_peaks.get(bucket, 0), peak_rss_bytes)
        self._vm_peaks[bucket] = max(self._vm_peaks.get(bucket, 0), peak_vm_bytes or 0)
        return True

    def observe_live(self, bucket, peak_rss_bytes, peak_vm_bytes=None):
        """Raise a live estimate without claiming a completed-file sample."""
        self._record_peak(bucket, peak_rss_bytes, peak_vm_bytes)

    def observe(self, bucket, peak_rss_bytes, peak_vm_bytes=None, *, success=True):
        """Keep observed maxima; failed files use full reservations until success."""
        if bucket is None:
            return
        if not success:
            self._blocked_buckets.add(bucket)
            self._completed_counts.pop(bucket, None)
            return
        if self._record_peak(bucket, peak_rss_bytes, peak_vm_bytes):
            self._completed_buckets.add(bucket)
            self._completed_counts[bucket] = self._completed_counts.get(bucket, 0) + 1
            self._blocked_buckets.discard(bucket)

    def reservation(self, bucket):
        """Use observed RSS and padding; the explicit target drops the 25% factor.

        The explicit target uses one complete exact-class measurement rather
        than holding a sparse class at a global bootstrap until three finish.
        It retains the 512-MiB startup allowance,
        rounding, address-space guard and full reservation for unknown files.
        """
        hard = self.worker_limit_bytes
        if (bucket is None or bucket in self._blocked_buckets
                or self._vm_peaks.get(bucket, 0) >= 7 * hard // 8):
            return hard
        completed = (self._completed_reduction and self._completed_counts.get(bucket, 0)
                     >= self.required_completed_samples)
        # A completed worker's VmHWM covers startup as well as event work.
        # Only full-worker exact-class samples can lower the live bootstrap
        # and its original 4-GiB floor. No live/instantaneous RSS can do so.
        peak = self._rss_peaks.get(bucket, 0)
        if not completed:
            peak = max(self.bootstrap_peak_rss_bytes, peak)
        quantum = 256 * MIB
        envelope = (peak if self.target_rss and completed else (5 * peak + 3) // 4) + 512 * MIB
        envelope = ((envelope + quantum - 1) // quantum) * quantum
        return min(hard, max(GIB if completed else 4 * GIB, envelope))

    def active_reservation(self, bucket, booked, resident, *, live_peak=None):
        """Refresh materialized target workers rather than pinning old bookings.

        The dispatcher records live high-water marks before calling this. A
        submitted future without materialized anonymous RSS keeps its original
        booking, and unknown/failed classes still return the full hard limit.
        A warm, measured target worker uses its own peak rather than assuming
        every file simultaneously reaches a different file's historical peak.
        """
        current = self.reservation(bucket)
        if self.target_rss and resident > 0:
            if (resident >= GIB and type(live_peak) is int and live_peak >= resident
                    and current < self.worker_limit_bytes
                    and self._completed_counts.get(bucket, 0) >= self.required_completed_samples):
                quantum = 256 * MIB
                own = ((live_peak + 256 * MIB + quantum - 1) // quantum) * quantum
                return max(resident, min(current, max(2 * GIB, own)))
            return max(resident, current)
        return max(booked, current)

    def admit(self, memory_available_bytes, active_reservations, pending_buckets, *,
              account_slots, now, dispatch_limit=6):
        """Fill the target immediately, subject to host RAM and account room.

        Active pairs are (booked total bytes, live RssAnon bytes). Only the
        latter is credited against memory already absent from MemAvailable.
        A live reservation overrun stops new dispatch until the caller refreshes
        that worker's booking from its measured high-water mark.
        """
        active = list(active_reservations)
        if dispatch_limit <= 0:
            return []
        room = memory_available_bytes - HOST_RESERVE
        for booked, resident in active:
            resident = max(0, resident)
            if resident > booked:
                return []
            room -= max(0, booked - resident)
        if room <= 0:
            return []
        slots = min(self.stage_ceiling, account_slots) - len(active)
        selected = []
        for bucket in pending_buckets:
            if slots <= 0 or len(selected) >= dispatch_limit:
                break
            need = self.reservation(bucket)
            if need > room:
                break
            room -= need
            slots -= 1
            selected.append(need)
        return selected
