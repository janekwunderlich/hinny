"""Retrieval-only LongMemEval run on Hinny, no LLM.

For every question, its haystack is indexed in a fresh Hinny database (through the Python
binding) as sessions or as rounds, with the session's timestamp, the session id and the round
index as attributes, and the question is run against it as:

    bm25        text only, language="english" (stemming, stopwords)
    bm25-plain  text only, Hinny's default analyzer (no stemming)
    vector      Contriever vector only (cosine)
    vector-ip   Contriever vector only, ranked by inner product (see "ip" below), as the paper
    hybrid-ip   text + inner-product vectors, default fusion; hybrid-ip-rrf with fusion="rrf"
    hybrid      text + vector, Hinny's default fusion (scores scaled to 0-1, text_weight 0.5)
    hybrid-rrf  text + vector, fusion="rrf"

and, outside Hinny, the paper's own retrievers as a reference for the sanity check:

    ref-bm25    rank_bm25.BM25Okapi over whitespace-split text, as in run_retrieval.py
    ref-dot     brute-force Contriever dot product, as in run_retrieval.py
    ref-cos     brute-force Contriever cosine (what Hinny's vector search computes)

Vector rows run only for questions whose texts are all in the embedding cache (embed.py).
Each line of the output is one (question, config, mode) with metrics under both label sets
(see common.corpus); timings go to a second file.

    python run_lme.py --data S_CLEANED --configs session:user,round:user --out results/x.jsonl
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi

import hinny
import common

MODES_TEXT = ("bm25", "bm25-plain", "ref-bm25")
MODES_VEC = ("vector", "hybrid", "hybrid-rrf", "ref-dot", "ref-cos")


def iter_data(path, ids=None):
    if path.endswith("longmemeval_m.json"):
        import ijson  # the M file is 2.7 GB; stream it
        with open(path, "rb") as f:
            for x in ijson.items(f, "item"):
                if ids is None or x["question_id"] in ids:
                    yield x
    else:
        for x in common.load(path):
            if ids is None or x["question_id"] in ids:
                yield x


def rank_scores(scores):
    # As run_retrieval.py: np.argsort(scores)[::-1] (ties: later corpus items first).
    return [int(i) for i in np.argsort(np.asarray(scores))[::-1]]


def run_question(entry, unit, key, cache, workdir, use_vectors):
    items, paper_labels, all_labels = common.corpus(entry, unit, key)
    ids = [i for i, _, _ in items]
    texts = [t for _, t, _ in items]
    q = entry["question"]
    n = len(items)
    limit = min(n, 100)

    have_vec = use_vectors and q in cache and all(t in cache for t in texts)
    vecs = np.stack([cache.get(t) for t in texts]).astype(np.float32) if have_vec else None
    qv = cache.get(q).astype(np.float32) if have_vec else None

    rankings, lat = {}, {}
    timing = {"qid": entry["question_id"], "unit": unit, "key": key, "n": n,
              "chars": sum(map(len, texts)), "vectors": bool(have_vec)}

    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    db = hinny.Database(workdir)
    # "ip": the same vectors made to rank by inner product under cosine (the reduction of
    # maximum inner product search to cosine): each document gets an extra component
    # sqrt(M^2 - |x|^2), so every document has norm M, and the query gets a 0, so the cosine
    # is q.x / (|q| M), which orders documents exactly as q.x does. M is the haystack's
    # largest norm. Contriever is trained for dot product; Hinny offers cosine only.
    spaces = {"en": ("english", 768), "plain": (None, None), "ip": ("english", 769)}
    if not have_vec:
        del spaces["ip"]
    else:
        norms = np.linalg.norm(vecs, axis=1)
        big = float(norms.max())
        ip_vecs = np.hstack([vecs, np.sqrt(np.maximum(big ** 2 - norms ** 2, 0))[:, None]]).astype(np.float32)
        ip_q = np.append(qv, 0.0).astype(np.float32)
    nss = {}
    for name, (lang, dims) in spaces.items():
        nss[name] = db.namespace(
            name, text="text", ints=["ts", "round"], strings=["session"],
            vector=dims if have_vec else None, language=lang)
    for name, ns in nss.items():
        docs = []
        for k, (_, t, a) in enumerate(items):
            d = {"key": k, "text": t, "attrs": a}
            if have_vec and name == "en":
                d["vector"] = vecs[k]
            if name == "ip":
                d["vector"] = ip_vecs[k]
            docs.append(d)
        t0 = time.perf_counter()
        ns.upsert_many(docs)
        timing[f"ingest_{name}_s"] = time.perf_counter() - t0

    def timed(mode, ns, **kw):
        # The first query on a fresh haystack also builds per-segment scan caches: kept
        # apart as the cold time; the reported time is the median of five after it.
        t0 = time.perf_counter()
        ns.query(limit=limit, snippet=0, with_text=False, **kw)
        lat[mode + ":cold"] = time.perf_counter() - t0
        reps = []
        for _ in range(5):
            t0 = time.perf_counter()
            hits = ns.query(limit=limit, snippet=0, with_text=False, **kw)
            reps.append(time.perf_counter() - t0)
        lat[mode] = float(np.median(reps))
        rankings[mode] = [h.key for h in hits]

    timed("bm25", nss["en"], text=q)
    timed("bm25-plain", nss["plain"], text=q)
    if have_vec:
        timed("vector", nss["en"], vector=qv)
        timed("hybrid", nss["en"], text=q, vector=qv)
        timed("hybrid-rrf", nss["en"], text=q, vector=qv, fusion="rrf")
        timed("vector-ip", nss["ip"], vector=ip_q)
        timed("hybrid-ip", nss["ip"], text=q, vector=ip_q)
        timed("hybrid-ip-rrf", nss["ip"], text=q, vector=ip_q, fusion="rrf")
    del nss, db
    gc.collect()
    shutil.rmtree(workdir)

    # The paper's retrievers, exactly as run_retrieval.py calls them.
    bm = BM25Okapi([t.split(" ") for t in texts])
    rankings["ref-bm25"] = rank_scores(bm.get_scores(q.split(" ")))[:limit]
    if have_vec:
        rankings["ref-dot"] = rank_scores(vecs @ qv)[:limit]
        nv = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        rankings["ref-cos"] = rank_scores(nv @ (qv / np.linalg.norm(qv)))[:limit]

    timing["query_s"] = lat
    rows = []
    for mode, rank in rankings.items():
        ranked_ids = [ids[i] for i in rank]
        row = {"qid": entry["question_id"], "qtype": entry["question_type"], "unit": unit,
               "key": key, "mode": mode, "n": n}
        for lname, labels in (("paper", paper_labels), ("all", all_labels)):
            m = common.metrics(ranked_ids, labels, ids)
            if unit == "round":
                m.update(common.metrics_as_sessions(ranked_ids, labels, ids))
            m["n_labels"] = len(labels)
            row[lname] = m
        rows.append(row)
    return rows, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="S_CLEANED", help="S_CLEANED, S_ORIG, M_ORIG or a path")
    ap.add_argument("--configs", default="session:user,session:full,round:user,round:full")
    ap.add_argument("--questions", type=int, default=0, help="first N of the fixed random order (0 = all)")
    ap.add_argument("--no-vectors", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = getattr(common, args.data, args.data)
    configs = [tuple(c.split(":")) for c in args.configs.split(",")]

    cache = common.EmbeddingCache() if not args.no_vectors else {}
    ids = None
    if args.questions:
        order = common.subset_order(common.load(common.S_CLEANED))
        ids = set(order[:args.questions])
    workdir = os.path.join(common.WORK, "tmp", f"db_{os.getpid()}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t_start = time.time()
    count = 0
    with open(args.out, "w") as out, open(args.out.replace(".jsonl", ".timing.jsonl"), "w") as tout:
        for entry in iter_data(path, ids):
            if "_abs" in entry["question_id"]:
                continue
            for unit, key in configs:
                rows, timing = run_question(entry, unit, key, cache, workdir, not args.no_vectors)
                for r in rows:
                    out.write(json.dumps(r) + "\n")
                tout.write(json.dumps(timing) + "\n")
            count += 1
            if count % 25 == 0:
                print(f"{count} questions, {time.time() - t_start:.0f} s", flush=True)
    print(f"done: {count} questions in {time.time() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
