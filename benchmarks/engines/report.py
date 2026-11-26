"""Summarize results/DATASET.jsonl: per engine config, build time, size, and the fastest setting
(lowest p50) that reaches recall@10 >= 0.95, 0.99, 0.997.

    .venv/bin/python report.py DATASET [--curves]
"""
import json
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = (0.95, 0.99, 0.997)


def main():
    ds = sys.argv[1]
    curves = "--curves" in sys.argv
    builds, queries, overhead, hinny, rebuilds = OrderedDict(), OrderedDict(), [], [], {}
    with open(os.path.join(HERE, "results", f"{ds}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            key = (r.get("engine"), r.get("config"))
            if r["kind"] == "build":
                builds.setdefault(key, r)  # first build of a config; rebuilds (for extra sweeps) are listed below
                rebuilds.setdefault(key, []).append(r)
                queries.setdefault(key, [])
            elif r["kind"] == "query":
                rows = queries.setdefault(key, [])
                rows[:] = [x for x in rows if x["setting"] != r["setting"]]  # a re-measured setting replaces the old one
                rows.append(r)
            elif r["kind"] == "overhead":
                overhead.append(r)
            elif r["kind"] == "hinny":
                hinny.append(r)

    def st(s):
        return ",".join(f"{k}={v}" for k, v in s.items())

    print(f"## {ds}: Python engines (single thread, pinned, one query per call; p50/p99 in ms)\n")
    hdr = ["engine", "config", "build s", "size MB", "build load"]
    for t in TARGETS:
        hdr += [f">={t}: setting", "recall", "p50", "p99", "load", "others"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for key, rows in queries.items():
        b = builds.get(key, {})
        cells = [key[0], key[1], f"{b.get('build_s', '?')}", f"{b.get('size_bytes', 0) / 1e6:.0f}",
                 f"{b.get('load_before', '?')}" + ("" if b.get("gate", "strict") == "strict" else "*")]
        for t in TARGETS:
            ok = [r for r in rows if r["recall"] >= t]
            if not ok:
                best = max(rows, key=lambda r: r["recall"]) if rows else None
                cells += [f"not reached (max {best['recall']:.4f})" if best else "no data", "", "", "", "", ""]
                continue
            r = min(ok, key=lambda r: r["p50_ms"])
            cells += [st(r["setting"]), f"{r['recall']:.4f}", f"{r['p50_ms']:.3f}", f"{r['p99_ms']:.3f}",
                      f"{r['load_before']:.2f}" + ("" if r.get("gate", "strict") == "strict" else "*"), f"{r.get('other_busy_cores', '')}"]
        print("| " + " | ".join(cells) + " |")
    print()
    for key, b in builds.items():
        extra = {k: v for k, v in b.items() if k in ("train_s", "add_s", "table_write_s", "table_bytes", "total_bytes",
                                                    "size_kind", "load_after", "waited_s", "index_stats", "ts")}
        print(f"build {key[0]} {key[1]}: {b['build_s']} s, {b['size_bytes']} bytes, load before {b['load_before']}; {extra}")
        for rb in rebuilds[key][1:]:
            print(f"   rebuilt for a further sweep: {rb['build_s']} s, load before {rb['load_before']}, others {rb.get('other_busy_cores')}")
    print("\n## per-call overhead (index too small to cost anything)\n")
    for r in overhead:
        print(f"{r['engine']:<45} n={r['n_vectors']:<5} p50 {r['p50_ms']:.4f} ms  p99 {r['p99_ms']:.4f} ms  mean {r['mean_ms']:.4f} ms  load {r['load_before']}")
    print("\n## Hinny runs\n")
    for r in hinny:
        print(f"-- {r['config']}  load before {r['load_before']}  wall {r['wall_s']} s  rc {r['rc']}  ({r['file']})")
        for line in r["output"]:
            print("   " + line)
    if curves:
        print("\n## full curves\n")
        for key, rows in queries.items():
            print(f"-- {key[0]} {key[1]}")
            for r in rows:
                print(f"   {st(r['setting']):<32} recall {r['recall']:.4f}  p50 {r['p50_ms']:.3f}  p99 {r['p99_ms']:.3f}  "
                      f"mean {r['mean_ms']:.3f}  load {r['load_before']:.2f}  others {r.get('other_busy_cores')}  majflt {r.get('major_faults')}  tries {r.get('attempts')}  gate {r.get('gate', 'strict')}  passes {r['per_pass_p50_p99']}")


if __name__ == "__main__":
    main()
