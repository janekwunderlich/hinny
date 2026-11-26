# Hinny vs open-source vector engines: how to repeat

A million clustered vectors of 256 dimensions, searched by Hinny and by faiss, hnswlib, usearch
and LanceDB on one machine (an M1 Max under Linux), with the fastest setting of each that
reaches recall@10 of 0.95, 0.99 and 0.997. What came out is in DESIGN.md ("Against other
engines"); `results/` holds the records.

- `results/clustered.jsonl`, `results/environment.txt`: the other engines, measured on
  2026-11-13 (versions in the second file).
- `results/hinny_clustered_after.txt`: Hinny's own benchmark on the same data that evening,
  after the changes the comparison led to. The comparison itself ran an earlier build (0.136 ms
  a search at recall 0.997, where this file says 0.089 ms at 0.999).

Not run: a set of real embeddings (the GloVe script is written, not run end to end).

Everything runs from this folder. Nothing is installed system-wide. `run_hinny.py` expects the
benchmark's binary here as `hinny_vector_bench`: `cargo build --release -p hinny --example
vector_bench`, then copy `target/release/examples/vector_bench` to that name.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy faiss-cpu hnswlib usearch lancedb pyarrow h5py

.venv/bin/python gen_clustered.py          # data/clustered_docs.npy (1M x 256), data/clustered_queries.npy (1,000)
.venv/bin/python ground_truth.py clustered # data/clustered_gt.npy: exact top 10, faiss IndexFlatIP, NumPy cross-check
./run_all.sh clustered                     # every engine, one after another; appends to results/clustered.jsonl
.venv/bin/python report.py clustered --curves
```

`run_all.sh DATASET [STEP ...]` also takes single steps, e.g. `faiss:ivfflat_1000`, `usearch:c32_f16`,
`lancedb:ivfpq_1000_64`, `hinny:gate`, `hinny:live,pin,gate`, `overhead` (see its header).

## What each script does

| file | purpose |
|---|---|
| `gen_clustered.py` | dataset 1, fixed seed 20261113 (checksums in `results/clustered_sha256.txt`) |
| `ground_truth.py` | exact top 10 by inner product on normalized vectors; also reports how far a float16-rounded exact top 10 is from it |
| `common.py` | load gate, per-query timer (warm pass, then 3 timed passes, p50 = sorted[n/2], p99 = sorted[n*99/100], the same rule as Hinny's benchmark), result records |
| `bench_faiss.py`, `bench_hnswlib.py`, `bench_usearch.py`, `bench_lancedb.py` | `build` (all cores) and `query` (run under `taskset -c 5`, engine threads = 1) as separate processes |
| `bench_overhead.py` | the same single-query call on 16 and 1,000 vectors: per-call overhead of each Python engine |
| `run_hinny.py` | runs `./hinny_vector_bench npy DOCS QUERIES` unchanged; flags `live` (LIVE=1), `pin` (query thread pinned to cpu5 after the build), `gate` (paused after the build until the machine is quiet) |
| `report.py` | per engine: build time, size, fastest setting reaching recall@10 >= 0.95 / 0.99 / 0.997 |
| `prep_glove.py` | prepares the ann-benchmarks GloVe-200-angular set as a second dataset (written, download tested, never run end to end) |

## The gate

Before a build and before a query sweep the scripts wait until the 1-minute load average is <= 2.0 and
other processes use <= 1.5 cores ("strict"). If that does not happen within `PATIENCE` seconds (180 by
default) they go ahead once other processes use <= 3 cores ("relaxed"); such results carry
`"gate": "relaxed"` and are marked `*` in the report. Each query setting also records the cores other
processes kept busy during its timed passes and its major page faults, and is measured again (up to
twice) when either was too high.

Environment knobs: `LOAD_LIMIT`, `BUSY_LIMIT`, `BUSY_RELAXED`, `PATIENCE`, `MAX_WAIT`, `PASSES`,
`RETRIES`, `KFACTORS` (faiss refine factors), `LANCE_REFINE`, `REUSE_INDEX`.

Index files are written to `indexes/` and removed after each query sweep; the folder sits on a
RAM-backed filesystem, so keep an eye on free space (`df -h /tmp`).
