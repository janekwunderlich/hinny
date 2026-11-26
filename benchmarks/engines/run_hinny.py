"""Run the prebuilt Hinny benchmark binary unchanged and keep its output with load figures.

    .venv/bin/python run_hinny.py DATASET [live] [pin] [gate]

    (no flags)  ./hinny_vector_bench npy DOCS QUERIES, exactly as given (no CPU pinning)
    live        the same with LIVE=1 in the environment
    pin         once the build line has been printed, the main thread (the one that runs the timed
                queries) is pinned to cpu5 from outside with sched_setaffinity; the build itself is
                not pinned. The throughput section then reports one thread only (the binary sizes
                it from the affinity mask), so take throughput from an unpinned run.

    gate        once the build line has been printed, the process is paused from outside (SIGSTOP)
                until the 1-minute load average is back under the limit and other processes are
                quiet, then resumed (SIGCONT). The build raises the load average by itself, and
                the binary goes straight from building to timing, so without this the "load <= 2
                before a timed run" rule cannot be applied to its queries. The pause falls inside
                its exact-search pass (about 20 s of scanning that follows the pause and re-warms
                memory before the timed lines); the "exact ... ms per query" figure of a gated run
                includes the pause and is not usable.

TMPDIR is pointed at ./hinny_tmp so the index it writes stays inside this folder.
Each output line is stored with the 1-minute load average at that moment and with the average number
of cores kept busy by processes other than Hinny since the previous line.
"""
import os
import signal
import subprocess
import sys
import time

import common as C


def proc_cpu(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
    except OSError:
        return 0.0


def proc_majflt(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[9])
    except OSError:
        return 0


def machine_busy():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:9]]
    return (v[0] + v[1] + v[2] + v[5] + v[6] + v[7]) / os.sysconf("SC_CLK_TCK")


def main():
    ds = sys.argv[1]
    flags = set(sys.argv[2:])
    mode = "+".join(["live" if "live" in flags else "merged", "pinned5" if "pin" in flags else "unpinned"]
                    + (["gated"] if "gate" in flags else []))
    tmp = os.path.join(C.HERE, "hinny_tmp")
    os.makedirs(tmp, exist_ok=True)
    env = dict(os.environ, TMPDIR=tmp)
    env.pop("LIVE", None)
    if "live" in flags:
        env["LIVE"] = "1"
    load, waited = C.wait_quiet(what=f"hinny {mode}")
    out_path = os.path.join(C.RES, f"hinny_{ds}_{mode}_{time.strftime('%H%M%S')}.txt")
    cmd = [os.path.join(C.HERE, "hinny_vector_bench"), "npy",
           os.path.join(C.DATA, f"{ds}_docs.npy"), os.path.join(C.DATA, f"{ds}_queries.npy")]
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True, bufsize=1)
    last = (time.time(), machine_busy(), proc_cpu(p.pid))
    lines = []
    last_mf = 0
    with open(out_path, "w") as out:
        out.write(f"# {' '.join(cmd)}  mode={mode}  load_before={load} waited={waited:.0f}s\n")
        for line in p.stdout:
            line = line.rstrip("\n")
            now = (time.time(), machine_busy(), proc_cpu(p.pid))
            dt = max(now[0] - last[0], 1e-9)
            other = max(((now[1] - last[1]) - (now[2] - last[2])) / dt, 0.0)
            hinny = (now[2] - last[2]) / dt
            last = now
            mf = proc_majflt(p.pid)
            note = (f"[t+{now[0] - t0:7.1f}s load {C.loadavg():5.2f} other-busy-cores {other:4.2f} "
                    f"hinny-cores {hinny:4.2f} major-faults {mf - last_mf}]")
            last_mf = mf
            if "pin" in flags and "vectors ×" in line:
                os.sched_setaffinity(p.pid, {5})
                note += f" main thread pinned to {sorted(os.sched_getaffinity(p.pid))}"
            if "gate" in flags and "vectors ×" in line:
                os.kill(p.pid, signal.SIGSTOP)
                qload, paused = C.wait_quiet(what=f"hinny {mode} queries")
                os.kill(p.pid, signal.SIGCONT)
                last = (time.time(), machine_busy(), proc_cpu(p.pid))
                last_mf = proc_majflt(p.pid)
                note += f" paused {paused:.0f}s, resumed at load {qload} ({C.GATE['mode']} gate)"
            print(line, note, flush=True)
            out.write(f"{line}  {note}\n")
            lines.append(f"{line}  {note}")
    rc = p.wait()
    C.record(ds, {"kind": "hinny", "engine": "hinny", "config": mode, "rc": rc, "load_before": load,
                  "load_after": C.loadavg(), "wall_s": round(time.time() - t0, 1), "output": lines,
                  "file": os.path.basename(out_path)})


if __name__ == "__main__":
    main()
