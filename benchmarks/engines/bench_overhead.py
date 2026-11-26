"""Per-call overhead of each Python engine: the same single-query call on an index too small to cost anything.

    taskset -c 5 .venv/bin/python bench_overhead.py DATASET

Uses the first 16 and the first 1,000 documents of DATASET and its 1,000 queries, k=10, one query per
call, same timer as the real runs. The 16-vector numbers are (almost) pure call overhead: Python
argument handling, the binding layer, result arrays, and for LanceDB the query planner, the async
bridge and Arrow. Hinny's Rust benchmark has none of this.
"""
import os
import shutil
import sys
import time

import faiss
import hnswlib
import lancedb
import numpy as np
import pyarrow as pa
from usearch.index import Index, MetricKind

import common as C


def measure(name, n, search_one, qs):
    for q in qs:
        search_one(q)
    pc = time.perf_counter
    times = []
    for _ in range(C.PASSES):
        for q in qs:
            t0 = pc()
            search_one(q)
            t1 = pc()
            times.append((t1 - t0) * 1000.0)
    s = sorted(times)
    return {"kind": "overhead", "engine": name, "n_vectors": n, "p50_ms": round(C.pct(s, 1, 2), 5),
            "p99_ms": round(C.pct(s, 99, 100), 5), "mean_ms": round(sum(s) / len(s), 5),
            "load_before": C.loadavg(), "affinity": C.affinity()}


def main():
    ds = sys.argv[1]
    X = np.load(os.path.join(C.DATA, f"{ds}_docs.npy"), mmap_mode="r")
    qs2, _ = C.prep_queries(ds)
    qs1 = [q[0] for q in qs2]
    d = qs1[0].shape[0]
    faiss.omp_set_num_threads(1)
    load, waited = C.wait_quiet(what="overhead")
    print(f"[overhead] load {load}, waited {waited:.0f}s", flush=True)

    # Timer alone: perf_counter back to back, and an empty Python call.
    def noop(q):
        return None
    C.record(ds, measure("python no-op call", 0, noop, qs1))

    for n in (16, 1000):
        sub = np.ascontiguousarray(X[:n])
        fl = faiss.IndexFlatIP(d)
        fl.add(sub)
        C.record(ds, measure("faiss IndexFlatIP", n, lambda q: fl.search(q, 10)[1][0], qs2))

        fh = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        fh.add(sub)
        C.record(ds, measure("faiss IndexHNSWFlat", n, lambda q: fh.search(q, 10)[1][0], qs2))

        quant = faiss.IndexFlatIP(d)
        base = faiss.IndexIVFPQFastScan(quant, d, 1, 64, 4, faiss.METRIC_INNER_PRODUCT)
        rf = faiss.IndexRefineFlat(base)
        rf.train(np.ascontiguousarray(X[:max(n, 1000)]))
        rf.add(sub)
        C.record(ds, measure("faiss IVFPQFastScan+RefineFlat (nlist 1)", n, lambda q: rf.search(q, 10)[1][0], qs2))

        h = hnswlib.Index(space="ip", dim=d)
        h.init_index(max_elements=n, ef_construction=200, M=32)
        h.add_items(sub, np.arange(n))
        h.set_num_threads(1)
        h.set_ef(10)
        C.record(ds, measure("hnswlib", n, lambda q: h.knn_query(q, k=10)[0][0], qs2))

        u = Index(ndim=d, metric=MetricKind.IP, dtype="f32")
        u.add(np.arange(n, dtype=np.uint64), sub)
        u.expansion_search = 10
        C.record(ds, measure("usearch", n, lambda q: u.search(q, 10, threads=1).keys, qs1))

        p = os.path.join(C.IDX, f"overhead_lancedb_{n}")
        shutil.rmtree(p, ignore_errors=True)
        db = lancedb.connect(p)
        vec = pa.FixedSizeListArray.from_arrays(pa.array(sub.reshape(-1)), d)
        tbl = db.create_table("t", data=pa.Table.from_arrays([pa.array(np.arange(n, dtype=np.int64)), vec],
                                                             names=["id", "vector"]))
        C.record(ds, measure("lancedb (no index, flat scan)", n,
                             lambda q: tbl.search(q, vector_column_name="vector").distance_type("dot")
                             .limit(10).select(["id", "_distance"]).to_arrow(), qs1))
        shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
