"""LanceDB (embedded, sync Python API): build and single-core query sweep.

    .venv/bin/python bench_lancedb.py build DATASET CONFIG
    taskset -c 5 .venv/bin/python bench_lancedb.py query DATASET CONFIG

CONFIG:
    ivfpq_P_S       IVF_PQ, P partitions, S sub-vectors of 8 bits      (sweep nprobes x refine_factor)
    ivfrq_P         IVF_RQ (RaBitQ, 1 bit per dimension), P partitions (sweep nprobes x refine_factor)
    ivfhnswsq_P     IVF_HNSW_SQ, P partitions, library m / ef_construction (sweep ef x refine_factor,
                    every partition probed when P is small)
Distance "dot" on the L2-normalized vectors. A query is
    tbl.search(q).distance_type("dot").nprobes(n).refine_factor(r).limit(10).select(["id", "_distance"]).to_arrow()
and the whole call is inside the timer: that is what a Python user of LanceDB pays per query.
"""
import os
import shutil
import sys
import time

import lancedb
import numpy as np
import pyarrow as pa

import common as C

NPROBE = [1, 2, 3, 4, 6, 8, 12, 16, 32]
# refine_factor grids (0 = no refine step). PQ codes need a large factor on these data to get
# anywhere near the recall targets; RaBitQ codes need less. Override with LANCE_REFINE=1,2,5.
REFINE = {"ivfpq": [10, 20, 50, 100, 200], "ivfrq": [0, 2, 5, 10, 20, 50]}
if os.environ.get("LANCE_REFINE"):
    REFINE = {k: [int(x) for x in os.environ["LANCE_REFINE"].split(",")] for k in REFINE}
EF = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
METRIC = os.environ.get("LANCE_METRIC", "dot")


def db_path(ds, config):
    return os.path.join(C.IDX, f"{ds}_lancedb_{config}")


def build(ds, config):
    X = C.docs(ds)
    n, d = X.shape
    p = db_path(ds, config)
    shutil.rmtree(p, ignore_errors=True)
    db = lancedb.connect(p)
    vec = pa.FixedSizeListArray.from_arrays(pa.array(X.reshape(-1)), d)
    table = pa.Table.from_arrays([pa.array(np.arange(n, dtype=np.int64)), vec], names=["id", "vector"])
    t0 = time.perf_counter()
    tbl = db.create_table("t", data=table)
    t1 = time.perf_counter()
    del table, vec
    data_bytes = C.path_size(p)
    load, waited = C.wait_quiet(what=f"build lancedb {config}")
    parts = config.split("_")
    kw = {"metric": METRIC, "vector_column_name": "vector"}
    if parts[0] == "ivfpq":
        kw.update(index_type="IVF_PQ", num_partitions=int(parts[1]), num_sub_vectors=int(parts[2]))
    elif parts[0] == "ivfrq":
        kw.update(index_type="IVF_RQ", num_partitions=int(parts[1]))
    elif parts[0] == "ivfhnswsq":
        kw.update(index_type="IVF_HNSW_SQ", num_partitions=int(parts[1]))
    else:
        raise SystemExit(f"unknown config {config}")
    gate = C.GATE["mode"]
    snap = C.cpu_snapshot()
    t2 = time.perf_counter()
    tbl.create_index(**kw)
    t3 = time.perf_counter()
    total = C.path_size(p)
    stats = None
    try:
        stats = str(tbl.index_stats(tbl.list_indices()[0].name))
    except Exception as e:  # noqa: BLE001
        stats = f"index_stats failed: {e}"
    C.record(ds, {"kind": "build", "engine": "lancedb", "config": config, "build_s": round(t3 - t2, 2),
                  "table_write_s": round(t1 - t0, 2), "size_bytes": total - data_bytes,
                  "size_kind": "index files on disk (table data separate)", "table_bytes": data_bytes,
                  "total_bytes": total, "load_before": load, "gate": gate, "load_after": C.loadavg(), "other_busy_cores": C.other_busy_cores(snap, C.cpu_snapshot()),
                  "threads": os.cpu_count(), "waited_s": round(waited), "metric": METRIC, "index_stats": stats})


def query(ds, config):
    qs, gtsets = C.prep_queries(ds)
    qs = [q[0] for q in qs]
    db = lancedb.connect(db_path(ds, config))
    tbl = db.open_table("t")
    kind = config.split("_")[0]
    state = {}

    def search_one(q):
        b = tbl.search(q, vector_column_name="vector").distance_type(METRIC)
        if "nprobes" in state:
            b = b.nprobes(state["nprobes"])
        if state.get("ef"):
            b = b.ef(state["ef"])
        if state.get("refine_factor"):
            b = b.refine_factor(state["refine_factor"])
        return b.limit(10).select(["id", "_distance"]).to_arrow()

    def extract(t):
        return t.column("id").to_pylist()

    def apply(s):
        state.clear()
        state.update(s)

    extra = {"metric": METRIC, "lance_cpu_threads": os.environ.get("LANCE_CPU_THREADS")}
    if kind == "ivfhnswsq":
        nparts = int(config.split("_")[1])
        for rf in [0, 1, 2, 5, 10]:
            C.sweep(ds, "lancedb", config, [{"refine_factor": rf, "nprobes": nparts, "ef": e} for e in EF if e >= 10 * max(rf, 1)],
                    apply, search_one, qs, gtsets, extra=extra, extract=extract)
    else:
        for rf in REFINE[kind]:
            C.sweep(ds, "lancedb", config, [{"refine_factor": rf, "nprobes": n} for n in NPROBE],
                    apply, search_one, qs, gtsets, extra=extra, extract=extract)


if __name__ == "__main__":
    mode, ds, config = sys.argv[1:4]
    {"build": build, "query": query}[mode](ds, config)
