#!/usr/bin/env bash
# Repeat the comparison for one dataset, one engine after another, never two at once.
#
#   ./run_all.sh DATASET [STEP ...]
#
# STEP is a whole stage (overhead hinny faiss hnswlib usearch lancedb; default: all of them, in that
# order) or one item of a stage:
#   hinny:FLAGS          FLAGS = comma-separated subset of live,pin,gate ("hinny:" = exactly as given)
#   faiss:CONFIG         e.g. faiss:hnsw, faiss:ivfflat_1000, faiss:ivfpqfs_1000_128, faiss:ivfrabitqfs_4096
#   usearch:CONFIG       default_f32 c32_f32 default_f16 c32_f16
#   lancedb:CONFIG       e.g. lancedb:ivfpq_1000_64, lancedb:ivfrq_1000, lancedb:ivfhnswsq_1
#
# Needs data/DATASET_docs.npy, data/DATASET_queries.npy and data/DATASET_gt.npy
# (gen_clustered.py + ground_truth.py for "clustered"). Every build runs on all cores; every query
# sweep runs in a fresh process pinned to cpu5 (taskset -c 5) with the engine's thread count at 1.
# Each step waits until the 1-minute load average is <= LOAD_LIMIT (default 2.0) and other processes
# are quiet, and records both. Results are appended to results/DATASET.jsonl; summarize with
#   .venv/bin/python report.py DATASET [--curves]
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
DS=${1:?dataset}
shift
STEPS=${*:-overhead hinny faiss hnswlib usearch lancedb}
QCORE=${QCORE:-5}
NLIST_A=${NLIST_A:-1000}
NLIST_B=${NLIST_B:-4096}
PQ_A=${PQ_A:-64}    # 4-bit sub-quantizers for IVFPQFastScan: dims/4 and dims/2
PQ_B=${PQ_B:-128}
LANCE_PARTS=${LANCE_PARTS:-1000}
LANCE_SUBVEC=${LANCE_SUBVEC:-64}

step() { echo "=== $(date +%T) load $(cut -d' ' -f1 /proc/loadavg) :: $*"; }

hinny_run() {  # $1 = comma-separated flags
  step "hinny [${1:-as given}]"; $PY run_hinny.py "$DS" ${1//,/ }
}
faiss_cfg() {
  step "faiss $1 build"; $PY bench_faiss.py build "$DS" "$1" || { echo "FAILED build faiss $1"; return; }
  step "faiss $1 query"; taskset -c "$QCORE" $PY bench_faiss.py query "$DS" "$1" || echo "FAILED query faiss $1"
  rm -f "indexes/${DS}_faiss_$1.index"
}
hnswlib_cfg() {
  step "hnswlib build"; $PY bench_hnswlib.py build "$DS" || { echo "FAILED build hnswlib"; return; }
  step "hnswlib query"; taskset -c "$QCORE" $PY bench_hnswlib.py query "$DS" || echo "FAILED query hnswlib"
  rm -f "indexes/${DS}_hnswlib_M32_efc200.bin"
}
usearch_cfg() {
  step "usearch $1 build"; $PY bench_usearch.py build "$DS" "$1" || { echo "FAILED build usearch $1"; return; }
  step "usearch $1 query"; taskset -c "$QCORE" $PY bench_usearch.py query "$DS" "$1" || echo "FAILED query usearch $1"
  rm -f "indexes/${DS}_usearch_$1.usearch"
}
lancedb_cfg() {
  step "lancedb $1 build"; $PY bench_lancedb.py build "$DS" "$1" || { echo "FAILED build lancedb $1"; return; }
  step "lancedb $1 query"; taskset -c "$QCORE" $PY bench_lancedb.py query "$DS" "$1" || echo "FAILED query lancedb $1"
  rm -rf "indexes/${DS}_lancedb_$1"
}

for s in $STEPS; do
  case $s in
    overhead) step overhead; taskset -c "$QCORE" $PY bench_overhead.py "$DS" ;;
    hinny)
      # "gate": pause the binary after its build until the machine is quiet again (see run_hinny.py).
      for f in "" gate pin,gate live live,gate live,pin,gate; do hinny_run "$f"; done ;;
    hinny:*) hinny_run "${s#hinny:}" ;;
    faiss)
      for c in hnsw ivfflat_$NLIST_A ivfflat_$NLIST_B \
               ivfpqfs_${NLIST_A}_$PQ_A ivfpqfs_${NLIST_A}_$PQ_B ivfpqfs_${NLIST_B}_$PQ_A ivfpqfs_${NLIST_B}_$PQ_B \
               ivfrabitq_$NLIST_A ivfrabitqfs_$NLIST_A ivfrabitqfs_$NLIST_B; do
        faiss_cfg "$c"
      done ;;
    faiss:*) faiss_cfg "${s#faiss:}" ;;
    hnswlib) hnswlib_cfg ;;
    usearch) for c in default_f32 c32_f32 default_f16 c32_f16; do usearch_cfg "$c"; done ;;
    usearch:*) usearch_cfg "${s#usearch:}" ;;
    lancedb) for c in ivfpq_${LANCE_PARTS}_$LANCE_SUBVEC ivfrq_$LANCE_PARTS ivfhnswsq_1; do lancedb_cfg "$c"; done ;;
    lancedb:*) lancedb_cfg "${s#lancedb:}" ;;
    *) echo "unknown step $s" ;;
  esac
done
step done
