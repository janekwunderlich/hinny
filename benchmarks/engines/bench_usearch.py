"""usearch: inner product on normalized vectors; sweep expansion_search.

    .venv/bin/python bench_usearch.py build DATASET CONFIG
    taskset -c 5 .venv/bin/python bench_usearch.py query DATASET CONFIG

CONFIG:
    default_f32 / default_f16   library defaults (connectivity 16, expansion_add 128)
    c32_f32 / c32_f16           connectivity 32, expansion_add 200 (to match the other HNSW builds)
"""
import os
import sys
import time

import numpy as np
from usearch.index import Index, MetricKind

import common as C

EF = [10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048]


def index_path(ds, config):
    return os.path.join(C.IDX, f"{ds}_usearch_{config}.usearch")


def make(config, d):
    shape, dtype = config.split("_")
    kw = {} if shape == "default" else {"connectivity": 32, "expansion_add": 200}
    return Index(ndim=d, metric=MetricKind.IP, dtype=dtype, **kw)


def build(ds, config):
    X = C.docs(ds)
    load, waited = C.wait_quiet(what=f"build usearch {config}")
    gate = C.GATE["mode"]
    snap = C.cpu_snapshot()
    ix = make(config, X.shape[1])
    t0 = time.perf_counter()
    ix.add(np.arange(X.shape[0], dtype=np.uint64), X, threads=0)
    t1 = time.perf_counter()
    p = index_path(ds, config)
    ix.save(p)
    C.record(ds, {"kind": "build", "engine": "usearch", "config": config, "build_s": round(t1 - t0, 2),
                  "size_bytes": os.path.getsize(p), "size_kind": "Index.save file", "load_before": load,
                  "gate": gate, "load_after": C.loadavg(), "other_busy_cores": C.other_busy_cores(snap, C.cpu_snapshot()), "threads": os.cpu_count(), "waited_s": round(waited),
                  "connectivity": ix.connectivity, "expansion_add": ix.expansion_add, "dtype": str(ix.dtype)})


def query(ds, config):
    qs, gtsets = C.prep_queries(ds)
    ix = make(config, qs[0].shape[1])
    ix.load(index_path(ds, config))
    qs = [q[0] for q in qs]  # usearch takes a 1-D vector for a single query

    def search_one(q):
        return ix.search(q, 10, threads=1).keys

    def apply(s):
        ix.expansion_search = s["expansion_search"]

    C.sweep(ds, "usearch", config, [{"expansion_search": e} for e in EF], apply, search_one, qs, gtsets,
            extra={"connectivity": ix.connectivity, "dtype": str(ix.dtype)})


if __name__ == "__main__":
    mode, ds, config = sys.argv[1:4]
    {"build": build, "query": query}[mode](ds, config)
