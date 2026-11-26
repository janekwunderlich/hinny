"""Shared helpers: data loading, load-average gate, per-query timing, result records."""
import json
import os
import resource
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IDX = os.path.join(HERE, "indexes")
RES = os.path.join(HERE, "results")
os.makedirs(IDX, exist_ok=True)
os.makedirs(RES, exist_ok=True)

LOAD_LIMIT = float(os.environ.get("LOAD_LIMIT", "2.0"))
MAX_WAIT = float(os.environ.get("MAX_WAIT", "10800"))
PASSES = int(os.environ.get("PASSES", "3"))
BUSY_LIMIT = float(os.environ.get("BUSY_LIMIT", "1.5"))  # cores other processes may keep busy during a timed setting
RETRIES = int(os.environ.get("RETRIES", "2"))
PATIENCE = float(os.environ.get("PATIENCE", "180"))  # seconds to hold out for the strict gate
BUSY_RELAXED = float(os.environ.get("BUSY_RELAXED", "3.0"))  # cores others may use under the relaxed gate


def loadavg():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


GATE = {"mode": "strict"}  # how the last wait_quiet() let its caller through


def wait_quiet(limit=LOAD_LIMIT, what="", patience=None):
    """Block until the machine is quiet enough to time something. Returns (load, waited_s).

    Strict: the 1-minute load average is <= limit and, over a 2 s sample, other processes keep at
    most BUSY_LIMIT cores busy. If that has not happened after PATIENCE seconds (long-running
    background work that never lets the load average under the limit), a relaxed pass is taken as
    soon as other processes keep at most BUSY_RELAXED cores busy over a 5 s sample; GATE["mode"]
    then says "relaxed" and callers store it next to the result together with the load.
    """
    t0 = time.time()
    while True:
        l = loadavg()
        if l <= limit:
            a = cpu_snapshot()
            time.sleep(2)
            if other_busy_cores(a, cpu_snapshot()) <= BUSY_LIMIT:
                GATE["mode"] = "strict"
                return loadavg(), time.time() - t0
        elif time.time() - t0 > (PATIENCE if patience is None else patience):
            a = cpu_snapshot()
            time.sleep(5)
            if other_busy_cores(a, cpu_snapshot()) <= BUSY_RELAXED:
                GATE["mode"] = "relaxed"
                return loadavg(), time.time() - t0
        if time.time() - t0 > MAX_WAIT:
            print(f"[load] gave up waiting after {MAX_WAIT:.0f}s, load {l} ({what})", flush=True)
            GATE["mode"] = "gave up"
            return l, time.time() - t0
        time.sleep(5)


def docs(ds):
    return np.load(os.path.join(DATA, f"{ds}_docs.npy"))


def queries(ds):
    return np.load(os.path.join(DATA, f"{ds}_queries.npy"))


def ground_truth(ds):
    return np.load(os.path.join(DATA, f"{ds}_gt.npy"))


def path_size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def record(ds, rec):
    rec = dict(rec)
    rec["dataset"] = ds
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(RES, f"{ds}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def pct(sorted_times, num, den):
    """Same index rule as Hinny's vector_bench: sorted[len * num / den]."""
    return sorted_times[min(len(sorted_times) * num // den, len(sorted_times) - 1)]


def affinity():
    return sorted(os.sched_getaffinity(0))


def cpu_snapshot():
    """(wall time, busy jiffies of the whole machine, CPU seconds of this process)."""
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:9]]
    busy = v[0] + v[1] + v[2] + v[5] + v[6] + v[7]
    t = os.times()
    return time.time(), busy, t.user + t.system + t.children_user + t.children_system


def other_busy_cores(a, b):
    """Average number of cores kept busy by everything except this process between two snapshots."""
    dt = max(b[0] - a[0], 1e-9)
    return round(max(((b[1] - a[1]) / os.sysconf("SC_CLK_TCK") - (b[2] - a[2])) / dt, 0.0), 2)


def time_setting(search_one, qs, gtsets, passes=PASSES, extract=None):
    """Warm pass, then `passes` timed passes over the query set, one query per call.

    search_one(q) -> iterable of 10 ids. Only the call itself is inside the timer.
    Returns recall@10 (mean over queries, first timed pass), p50/p99/mean in ms over the
    pooled timed passes, and per-pass p50/p99 to show run-to-run noise.

    If other processes kept more than BUSY_LIMIT cores busy during the timed passes, or this process
    took major page faults in them (memory swapped out under it), the setting is measured again after the machine has gone quiet (up to RETRIES more times) and the least
    disturbed attempt is kept; "attempts" says how many were needed.
    """
    best = None
    for attempt in range(1 + RETRIES):
        load_before = loadavg()
        if attempt or (GATE["mode"] == "strict" and load_before > LOAD_LIMIT + 0.5):
            # (our own pinned thread adds up to 1.0 to the load average on its own)
            # Inside a sweep the strict gate was already tried at its start: do not hold out
            # another PATIENCE seconds for every setting.
            load_before, _ = wait_quiet(what="before setting", patience=20)
        allowed = BUSY_LIMIT if GATE["mode"] == "strict" else BUSY_RELAXED
        r = _measure(search_one, qs, gtsets, passes, extract)
        r["load_before"] = load_before
        r["gate"] = GATE["mode"]
        if best is None or (r["major_faults"], r["other_busy_cores"]) < (best["major_faults"], best["other_busy_cores"]):
            best = r
        if r["other_busy_cores"] <= allowed and r["major_faults"] == 0:
            break
    best["attempts"] = attempt + 1
    return best


def _measure(search_one, qs, gtsets, passes, extract):
    for q in qs:
        search_one(q)
    snap = cpu_snapshot()
    majflt0 = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
    pc = time.perf_counter
    all_times, per_pass, hits = [], [], 0
    for p in range(passes):
        times = []
        for i, q in enumerate(qs):
            t0 = pc()
            ids = search_one(q)
            t1 = pc()
            times.append((t1 - t0) * 1000.0)
            if p == 0:
                if extract is not None:
                    ids = extract(ids)
                hits += len(gtsets[i].intersection(int(x) for x in ids))
        s = sorted(times)
        per_pass.append((round(pct(s, 1, 2), 5), round(pct(s, 99, 100), 5)))
        all_times.extend(times)
    s = sorted(all_times)
    return {
        "recall": hits / (10.0 * len(qs)),
        "p50_ms": round(pct(s, 1, 2), 5),
        "p99_ms": round(pct(s, 99, 100), 5),
        "mean_ms": round(sum(s) / len(s), 5),
        "per_pass_p50_p99": per_pass,
        "load_after": loadavg(),
        "other_busy_cores": other_busy_cores(snap, cpu_snapshot()),
        # Pages this process had to fetch back from swap or disk during the timed passes: the
        # machine swaps when other work fills memory, and that lands in the tail latencies.
        "major_faults": resource.getrusage(resource.RUSAGE_SELF).ru_majflt - majflt0,
    }


def prep_queries(ds):
    Q = queries(ds)
    gt = ground_truth(ds)
    qs = [np.ascontiguousarray(Q[i:i + 1]) for i in range(len(Q))]
    gtsets = [set(int(x) for x in row) for row in gt]
    return qs, gtsets


def maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sweep(ds, engine, config, settings, apply, search_one, qs, gtsets, stop_at=0.9995, extra=None, extract=None):
    """Run settings in order; stop once recall passes stop_at (later settings only cost more) or
    has stopped rising over three settings in a row."""
    load0, waited = wait_quiet(what=f"{engine} {config}")
    print(f"[{engine} {config}] start sweep, load {load0}, waited {waited:.0f}s, affinity {affinity()}", flush=True)
    seen = []
    for s in settings:
        apply(s)
        r = time_setting(search_one, qs, gtsets, extract=extract)
        seen.append(r["recall"])
        r.update({"kind": "query", "engine": engine, "config": config, "setting": s,
                  "affinity": affinity(), "maxrss_mb": round(maxrss_mb(), 1)})
        if extra:
            r.update(extra)
        record(ds, r)
        if stop_at is not None and r["recall"] >= stop_at:
            break
        # Recall no longer moving (capped by something this knob does not control): further
        # settings would only be slower.
        if len(seen) >= 3 and seen[-1] - seen[-3] < 0.0005:
            break
