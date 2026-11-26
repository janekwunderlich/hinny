"""What Python adds: write and query throughput through the bindings, one thread and eight.

    python examples/bench.py
"""

import tempfile
import threading
import time

import numpy as np

import hinny

N, DIMS = 100_000, 384
rng = np.random.default_rng(7)
centres = rng.standard_normal((500, DIMS)).astype(np.float32)
vecs = (centres[rng.integers(0, 500, N)] + 0.7 * rng.standard_normal((N, DIMS))).astype(np.float32)
words = ["lease", "invoice", "flight", "refund", "contract", "meeting", "report", "travel", "budget", "photo"]

with tempfile.TemporaryDirectory() as tmp:
    ns = hinny.Database(tmp).namespace("bench", text="text", ints=["n"], vector=DIMS)
    t = time.perf_counter()
    ns.upsert_many({"key": i, "text": f"{words[i % 10]} {words[(i * 7) % 10]} item {i}", "attrs": {"n": i}, "vector": vecs[i]} for i in range(N))
    wrote = time.perf_counter() - t
    t = time.perf_counter()
    ns.optimize()
    indexed = time.perf_counter() - t
    print(f"{N:,} documents × {DIMS} dimensions: upsert_many {wrote:.1f} s ({N / wrote:,.0f}/s), optimize (one segment: clusters, codes) {indexed:.1f} s")

    queries = vecs[:500]
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    exact = np.argsort(-(unit[:500] @ unit.T), axis=1)[:, :10]
    recall = np.mean([len({h.key for h in ns.query(vector=q, limit=10)} & set(e)) / 10 for q, e in zip(queries, exact)])
    print(f"vector recall@10 against NumPy brute force: {recall:.3f}")
    for label, run in [
        ("text", lambda q: ns.query(text="lease refund", limit=10)),
        ("vector", lambda q: ns.query(vector=q, limit=10)),
        ("hybrid", lambda q: ns.query(text="lease refund", vector=q, limit=10)),
        ("hybrid + filter", lambda q: ns.query(text="lease refund", vector=q, filter=hinny.attr("n") < 10_000, limit=10)),
    ]:
        times = []
        for q in queries:
            s = time.perf_counter()
            run(q)
            times.append((time.perf_counter() - s) * 1000)
        times.sort()

        def worker(chunk):
            for q in chunk:
                run(q)

        threads = [threading.Thread(target=worker, args=(queries[i::8],)) for i in range(8)]
        s = time.perf_counter()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        qps8 = len(queries) / (time.perf_counter() - s)
        print(f"{label:<16} p50 {times[len(times) // 2]:.3f} ms  p99 {times[int(len(times) * 0.99)]:.3f} ms  one thread {1000 / (sum(times) / len(times)):,.0f} q/s  eight threads {qps8:,.0f} q/s")
