"""How a hybrid query's candidate budgets trade quality for time, measured on the queries of a
split other than test (dev, or train where there's no dev), so the defaults chosen from it
aren't tuned on the test queries. Uses run_bench's Hinny build and the beir-v2 protocol: request
11, remove the query's own document, keep 10; time the same request over the first 300 queries.

  candidates.py DATA EMB OUT fiqa:dev nfcorpus:dev scifact:train ...
  candidates.py DATA EMB OUT fiqa:dev:sparse ...   # a sparse field and vectors

Needs emb/<name>/queries-<split>.npy (embed_beir.py with SPLIT=<split>), and for `:sparse`
sparse_queries-<split>.jsonl (embed_sparse.py with SPLIT=<split>). Prints one JSON line per
dataset and budget: nDCG@10 and p50/p99 ms."""
import json, statistics, sys, time
from pathlib import Path

# Run with `python -I`, which leaves the script's folder off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bench as rb  # noqa: E402

TEXT = [20, 40, 100, 200, 400]
VECTOR = [50, 100, 200, 400]
SPARSE = [20, 40, 100, 200, 400]
SPARSE_VECTOR = [50, 100, 200]


def load_split(name, split):
    import numpy as np

    ds = rb.load(name)
    d = rb.DATA / name / name
    qrels = {}
    for line in list(open(d / "qrels" / f"{split}.tsv"))[1:]:
        q, doc, rel = line.rstrip("\n").split("\t")
        qrels.setdefault(q, {})[doc] = int(rel)
    qtext = {q["_id"]: q["text"] for q in (json.loads(l) for l in open(d / "queries.jsonl"))}
    e = rb.EMB / name
    qids = json.load(open(e / f"query_ids-{split}.json"))
    ds.update(qids=qids, qtexts=[qtext[q] for q in qids], qvectors=np.load(e / f"queries-{split}.npy"), qrels=qrels)
    return ds


def search(ns, ds, i, k, text, vector):
    hits = ns.query(text=ds["qtexts"][i], vector=ds["qvectors"][i], limit=k, snippet=0, with_text=False, text_candidates=text, vector_candidates=vector)
    return [h.key for h in hits]


def search_sparse(ns, ds, i, k, sparse, vector):
    hits = ns.query(sparse={"sp": ds["sparse_queries"][i]}, vector=ds["qvectors"][i], limit=k, snippet=0, with_text=False, sparse_candidates=sparse, vector_candidates=vector)
    return [h.key for h in hits]


def main():
    rb.DATA, rb.EMB, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    for spec in sys.argv[4:]:
        name, split, *mode = spec.split(":")
        sparse = mode == ["sparse"]
        ds = load_split(name, split)
        path = out / f"{name}-candidates"
        if path.exists():
            import shutil
            shutil.rmtree(path)
        if sparse:
            ds["sparse_queries"] = [json.loads(l) for l in open(rb.EMB / name / f"sparse_queries-{split}.jsonl")]
            rb.sparse_of(ds, "corpus")
            ns, run, grid = rb.hinny_build_sparse(ds, path), search_sparse, [(a, v) for a in SPARSE for v in SPARSE_VECTOR]
        else:
            ns, run, grid = rb.hinny_build_english(ds, path), search, [(a, v) for a in TEXT for v in VECTOR]
        n = min(300, len(ds["qids"]))
        for lexical, vector in grid:
            nd = [rb.ndcg10(rb.exclude_self(ds, i, run(ns, ds, i, 11, lexical, vector), 10), ds["qrels"][q]) for i, q in enumerate(ds["qids"])]
            for i in range(min(20, n)):
                run(ns, ds, i, 11, lexical, vector)
            ms = []
            for i in range(n):
                s = time.perf_counter()
                run(ns, ds, i, 11, lexical, vector)
                ms.append((time.perf_counter() - s) * 1000)
            ms.sort()
            print(json.dumps({"dataset": name, "split": split, "queries": len(ds["qids"]), **({"sparse": lexical} if sparse else {"text": lexical}), "vector": vector,
                              "ndcg10": round(statistics.mean(nd), 4), "p50_ms": round(ms[len(ms) // 2], 3), "p99_ms": round(ms[int(len(ms) * 0.99)], 3)}), flush=True)


if __name__ == "__main__":
    main()
