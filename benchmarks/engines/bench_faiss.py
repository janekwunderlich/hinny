"""faiss: build (all cores) and single-thread query sweep, as two separate processes.

    .venv/bin/python bench_faiss.py build DATASET CONFIG
    taskset -c 5 .venv/bin/python bench_faiss.py query DATASET CONFIG

CONFIG:
    hnsw                 IndexHNSWFlat, M=32, efConstruction=200           (sweep efSearch)
    ivfflat_NLIST        IndexIVFFlat, flat inner-product coarse quantizer (sweep nprobe)
    ivfpqfs_NLIST_M      IndexIVFPQFastScan (M sub-quantizers x 4 bits) under IndexRefineFlat
                         (sweep nprobe x k_factor)
    ivfrabitq_NLIST      IndexIVFRaBitQ (1 bit/dim) under IndexRefineFlat (sweep nprobe x k_factor)
    ivfrabitqfs_NLIST    IndexIVFRaBitQFastScan under IndexRefineFlat      (sweep nprobe x k_factor)
All use METRIC_INNER_PRODUCT on the L2-normalized vectors.
"""
import os
import sys
import time

import faiss
import numpy as np

import common as C

EF = [10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048]
NPROBE = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]
KFACTOR = [int(x) for x in os.environ.get("KFACTORS", "4,8,16,32,64,128,256").split(",")]


_KEEP = []


def index_path(ds, config):
    return os.path.join(C.IDX, f"{ds}_faiss_{config}.index")


def make(config, d):
    parts = config.split("_")
    ip = faiss.METRIC_INNER_PRODUCT
    if parts[0] == "hnsw":
        ix = faiss.IndexHNSWFlat(d, 32, ip)
        ix.hnsw.efConstruction = 200
        return ix
    nlist = int(parts[1])
    quantizer = faiss.IndexFlatIP(d)
    _KEEP.append(quantizer)
    if parts[0] == "ivfflat":
        return faiss.IndexIVFFlat(quantizer, d, nlist, ip)
    if parts[0] == "ivfpqfs":
        base = faiss.IndexIVFPQFastScan(quantizer, d, nlist, int(parts[2]), 4, ip)
    elif parts[0] == "ivfrabitq":
        base = faiss.IndexIVFRaBitQ(quantizer, d, nlist, ip)
    elif parts[0] == "ivfrabitqfs":
        base = faiss.IndexIVFRaBitQFastScan(quantizer, d, nlist, ip)
    else:
        raise SystemExit(f"unknown config {config}")
    ix = faiss.IndexRefineFlat(base)
    _KEEP.extend([quantizer, base])  # keep the Python-owned parts alive as long as the wrapper
    return ix


def build(ds, config):
    if os.environ.get("REUSE_INDEX") and os.path.exists(index_path(ds, config)):
        print(f"[faiss {config}] index file exists, not rebuilding", flush=True)
        return
    X = C.docs(ds)
    load, waited = C.wait_quiet(what=f"build faiss {config}")
    gate = C.GATE["mode"]
    snap = C.cpu_snapshot()
    ix = make(config, X.shape[1])
    t0 = time.perf_counter()
    if not ix.is_trained:
        ix.train(X)
    t1 = time.perf_counter()
    ix.add(X)
    t2 = time.perf_counter()
    p = index_path(ds, config)
    faiss.write_index(ix, p)
    C.record(ds, {"kind": "build", "engine": "faiss", "config": config, "build_s": round(t2 - t0, 2),
                  "train_s": round(t1 - t0, 2), "add_s": round(t2 - t1, 2), "size_bytes": os.path.getsize(p),
                  "size_kind": "faiss.write_index file", "load_before": load, "gate": gate, "load_after": C.loadavg(), "other_busy_cores": C.other_busy_cores(snap, C.cpu_snapshot()),
                  "threads": faiss.omp_get_max_threads(), "waited_s": round(waited)})


def query(ds, config):
    faiss.omp_set_num_threads(1)
    ix = faiss.read_index(index_path(ds, config))
    qs, gtsets = C.prep_queries(ds)
    kind = config.split("_")[0]

    def search_one(q):
        return ix.search(q, 10)[1][0]

    if kind == "hnsw":
        C.sweep(ds, "faiss", config, [{"efSearch": e} for e in EF],
                lambda s: setattr(ix.hnsw, "efSearch", s["efSearch"]), search_one, qs, gtsets)
    elif kind == "ivfflat":
        C.sweep(ds, "faiss", config, [{"nprobe": n} for n in NPROBE if n <= ix.nlist],
                lambda s: setattr(ix, "nprobe", s["nprobe"]), search_one, qs, gtsets)
    else:
        base = faiss.downcast_index(ix.base_index)
        extra = {}
        for a in ("qb", "centered"):
            if hasattr(base, a):
                extra[a] = int(getattr(base, a))

        def apply(s):
            base.nprobe = s["nprobe"]
            ix.k_factor = s["k_factor"]

        # One nprobe sweep per k_factor, each stopping once recall passes 0.9995.
        for kf in KFACTOR:
            C.sweep(ds, "faiss", config, [{"k_factor": kf, "nprobe": n} for n in NPROBE if n <= base.nlist],
                    apply, search_one, qs, gtsets, extra=extra)


if __name__ == "__main__":
    mode, ds, config = sys.argv[1:4]
    {"build": build, "query": query}[mode](ds, config)
