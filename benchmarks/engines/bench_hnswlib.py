"""hnswlib: M=32, ef_construction=200, inner product on normalized vectors; sweep ef.

    .venv/bin/python bench_hnswlib.py build DATASET
    taskset -c 5 .venv/bin/python bench_hnswlib.py query DATASET
"""
import os
import sys
import time

import hnswlib
import numpy as np

import common as C

EF = [10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048]
CONFIG = "M32_efc200"


def index_path(ds):
    return os.path.join(C.IDX, f"{ds}_hnswlib_{CONFIG}.bin")


def build(ds):
    if os.environ.get("REUSE_INDEX") and os.path.exists(index_path(ds)):
        print("[hnswlib] index file exists, not rebuilding", flush=True)
        return
    X = C.docs(ds)
    load, waited = C.wait_quiet(what="build hnswlib")
    gate = C.GATE["mode"]
    snap = C.cpu_snapshot()
    ix = hnswlib.Index(space="ip", dim=X.shape[1])
    t0 = time.perf_counter()
    ix.init_index(max_elements=X.shape[0], ef_construction=200, M=32, random_seed=100)
    ix.add_items(X, np.arange(X.shape[0]), num_threads=-1)
    t1 = time.perf_counter()
    p = index_path(ds)
    ix.save_index(p)
    C.record(ds, {"kind": "build", "engine": "hnswlib", "config": CONFIG, "build_s": round(t1 - t0, 2),
                  "size_bytes": os.path.getsize(p), "size_kind": "save_index file", "load_before": load,
                  "gate": gate, "load_after": C.loadavg(), "other_busy_cores": C.other_busy_cores(snap, C.cpu_snapshot()), "threads": os.cpu_count(), "waited_s": round(waited)})


def query(ds):
    qs, gtsets = C.prep_queries(ds)
    ix = hnswlib.Index(space="ip", dim=qs[0].shape[1])
    ix.load_index(index_path(ds))
    ix.set_num_threads(1)

    def search_one(q):
        return ix.knn_query(q, k=10)[0][0]

    C.sweep(ds, "hnswlib", CONFIG, [{"ef": e} for e in EF], lambda s: ix.set_ef(s["ef"]), search_one, qs, gtsets)


if __name__ == "__main__":
    mode, ds = sys.argv[1:3]
    {"build": build, "query": query}[mode](ds)
