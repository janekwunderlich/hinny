"""Retrieval-only LoCoMo run on Hinny, no LLM.

LoCoMo (github.com/snap-research/locomo, data/locomo10.json): 10 long conversations between
two speakers, each with its own QA set; every answerable question lists evidence dialog ids
("D3:12" = session 3, turn 12). Each conversation is one haystack, indexed at turn
granularity (one dialog turn: "Speaker: text", plus the image caption the dataset gives for
shared photos, as the LoCoMo paper's own retrieval does) and at session granularity, with
the session date as an attribute. Category 5 (adversarial, unanswerable) is skipped, as is
usual for retrieval; questions without parseable evidence are skipped and counted.

Metrics are common.metrics against the evidence ids (turn level), and session level for
both units. Modes as in run_lme.py; missing embeddings are computed here with Contriever.

    python run_locomo.py --out results/locomo.jsonl
"""

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi

import hinny
import common

LOCOMO = os.path.join(common.WORK, "src", "locomo", "data", "locomo10.json")
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}  # the LoCoMo paper's names


def parse_date(s):
    # "1:56 pm on 8 May, 2023"
    return int(datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y").timestamp())


def haystack(sample, unit):
    conv = sample["conversation"]
    items = []
    n = 1
    while f"session_{n}" in conv:
        ts = parse_date(conv[f"session_{n}_date_time"])
        turns = conv[f"session_{n}"]
        lines = []
        for t in turns:
            text = f"{t['speaker']}: {t['text']}"
            if t.get("blip_caption"):
                text += f" [shares a photo: {t['blip_caption']}]"
            lines.append((t["dia_id"], text))
        if unit == "turn":
            items += [(d, text, {"ts": ts, "session": f"D{n}"}) for d, text in lines]
        else:
            items.append((f"D{n}", "\n".join(x for _, x in lines), {"ts": ts, "session": f"D{n}"}))
        n += 1
    return items


def evidence(qa):
    ev = set()
    for e in qa.get("evidence", []):
        ev |= set(re.findall(r"D\d+:\d+", e))
    return ev


def rank_scores(scores):
    # As run_retrieval.py: np.argsort(scores)[::-1] (ties: later corpus items first).
    return [int(i) for i in np.argsort(np.asarray(scores))[::-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-vectors", action="store_true")
    args = ap.parse_args()
    data = common.load(LOCOMO)
    cache = common.EmbeddingCache()
    use_vec = not args.no_vectors

    if use_vec:
        texts = set()
        for s in data:
            for unit in ("turn", "session"):
                texts |= {t for _, t, _ in haystack(s, unit)}
            texts |= {q["question"] for q in s["qa"]}
        todo = sorted((t for t in texts if t not in cache), key=len)
        if todo:
            import embed
            tok, model = embed.load_model()
            t0 = time.time()
            for i in range(0, len(todo), 2000):
                chunk = todo[i:i + 2000]
                cache.add(chunk, embed.encode(tok, model, chunk))
                print(f"embedded {i + len(chunk)}/{len(todo)} in {time.time() - t0:.0f} s", flush=True)

    workdir = os.path.join(common.WORK, "tmp", f"db_locomo_{os.getpid()}")
    skipped = 0
    with open(args.out, "w") as out, open(args.out.replace(".jsonl", ".timing.jsonl"), "w") as tout:
        for s in data:
            for unit in ("turn", "session"):
                items = haystack(s, unit)
                ids = [i for i, _, _ in items]
                vecs = np.stack([cache.get(t) for _, t, _ in items]).astype(np.float32) if use_vec else None
                if os.path.exists(workdir):
                    shutil.rmtree(workdir)
                db = hinny.Database(workdir)
                ns = db.namespace("en", text="text", ints=["ts"], strings=["session"],
                                  vector=768 if use_vec else None, language="english")
                ns_plain = db.namespace("plain", text="text", ints=["ts"], strings=["session"])
                if use_vec:  # inner product under cosine, as in run_lme.py
                    norms = np.linalg.norm(vecs, axis=1)
                    big = float(norms.max())
                    ip_vecs = np.hstack([vecs, np.sqrt(np.maximum(big ** 2 - norms ** 2, 0))[:, None]]).astype(np.float32)
                    ns_ip = db.namespace("ip", text="text", ints=["ts"], strings=["session"],
                                         vector=769, language="english")
                docs = [{"key": k, "text": t, "attrs": a, **({"vector": vecs[k]} if use_vec else {})}
                        for k, (_, t, a) in enumerate(items)]
                t0 = time.perf_counter()
                ns.upsert_many(docs)
                ingest = time.perf_counter() - t0
                ns_plain.upsert_many([{"key": d["key"], "text": d["text"], "attrs": d["attrs"]} for d in docs])
                if use_vec:
                    ns_ip.upsert_many([{"key": d["key"], "text": d["text"], "attrs": d["attrs"],
                                        "vector": ip_vecs[d["key"]]} for d in docs])
                bm = BM25Okapi([t.split(" ") for _, t, _ in items])
                limit = min(len(items), 100)
                lat = {m: [] for m in ("bm25", "vector", "hybrid", "hybrid-rrf", "vector-ip", "hybrid-ip")}
                nq = 0
                for qi, qa in enumerate(s["qa"]):
                    if qa["category"] == 5:
                        continue
                    ev = evidence(qa)
                    if not ev:
                        skipped += unit == "turn"
                        continue
                    labels = ev if unit == "turn" else {e.split(":")[0] for e in ev}
                    q = qa["question"]
                    qv = cache.get(q).astype(np.float32) if use_vec else None
                    runs = {"bm25": dict(ns=ns, text=q), "bm25-plain": dict(ns=ns_plain, text=q)}
                    if use_vec:
                        runs.update({"vector": dict(ns=ns, vector=qv),
                                     "hybrid": dict(ns=ns, text=q, vector=qv),
                                     "hybrid-rrf": dict(ns=ns, text=q, vector=qv, fusion="rrf")})
                        ipq = np.append(qv, 0.0).astype(np.float32)
                        runs.update({"vector-ip": dict(ns=ns_ip, vector=ipq),
                                     "hybrid-ip": dict(ns=ns_ip, text=q, vector=ipq),
                                     "hybrid-ip-rrf": dict(ns=ns_ip, text=q, vector=ipq, fusion="rrf")})
                    rankings = {}
                    for mode, kw in runs.items():
                        kw = dict(kw)
                        space = kw.pop("ns")
                        t0 = time.perf_counter()
                        hits = space.query(limit=limit, snippet=0, with_text=False, **kw)
                        if mode in lat:
                            lat[mode].append(time.perf_counter() - t0)
                        rankings[mode] = [h.key for h in hits]
                    rankings["ref-bm25"] = rank_scores(bm.get_scores(q.split(" ")))[:limit]
                    if use_vec:
                        rankings["ref-dot"] = rank_scores(vecs @ qv)[:limit]
                    for mode, rank in rankings.items():
                        ranked = [ids[i] for i in rank]
                        m = common.metrics(ranked, labels, ids)
                        if unit == "turn":
                            sess = []
                            for r in ranked:
                                d = r.split(":")[0]
                                if d not in sess:
                                    sess.append(d)
                            sl = {e.split(":")[0] for e in ev}
                            sm = common.metrics(sess, sl, sorted({i.split(":")[0] for i in ids}))
                            m.update({"s_" + k: v for k, v in sm.items()})
                        m["n_labels"] = len(labels)
                        out.write(json.dumps({"qid": f"{s['sample_id']}#{qi}", "qtype": CATS[qa["category"]],
                                              "unit": unit, "key": "full", "mode": mode, "n": len(items),
                                              "all": m}) + "\n")
                    nq += 1
                tout.write(json.dumps({"sample": s["sample_id"], "unit": unit, "n": len(items),
                                       "chars": sum(len(t) for _, t, _ in items), "ingest_en_s": ingest,
                                       "questions": nq,
                                       "query_s_all": {k: v for k, v in lat.items() if v}}) + "\n")
                del ns, ns_plain, db
                ns_ip = None
                gc.collect()
                shutil.rmtree(workdir)
                print(s["sample_id"], unit, len(items), "items", nq, "questions", flush=True)
    print("questions skipped for lack of evidence ids:", skipped)


if __name__ == "__main__":
    sys.exit(main())
