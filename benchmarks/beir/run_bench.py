"""BEIR retrieval with embedded databases: the same documents (title + text), the same
EmbeddingGemma 2 vectors (256-d) and the same queries for every system, scored by one function.

    python -I run_bench.py DATA EMB OUT dataset [dataset ...]

Protocol beir-v2: nDCG@10 and warm latency both request 11 results, remove the query ID,
then retain 10. Recall@100 is a separate, untimed request for 101 results, trimmed to 100.
Build (load and index, ready to query) and index size are measured in a fresh directory.
Each row records its provenance: the commit, package versions, the machine, hashes of the
dataset and embedding files, and the conditions it was timed under.
"""
import hashlib, json, math, os, platform, re, shutil, statistics, subprocess, sys, tempfile, time, traceback
from functools import cache
from pathlib import Path

PROTOCOL = "beir-v2"
DATA = EMB = OUT = None
WORDS = re.compile(r"\w+", re.UNICODE)


PACKAGES = ["hinny", "lancedb", "chromadb", "qdrant-client", "tantivy", "sqlite-vec", "numpy", "pyarrow"]
CONDITIONS = {"cache": "warm: each system built in a fresh directory, timed after its quality passes and 20 warm-up queries",
              "threads": "one query at a time from Python; engines' own internal parallelism unchanged",
              "seed": "none in the harness; engines use their own defaults"}


@cache
def environment():
    """Where this run happens: the commit (and whether the tree had uncommitted changes), the
    packages, Python, and the machine."""
    import importlib.metadata as metadata
    import sqlite3

    repo = Path(__file__).resolve().parents[2]

    def run(*command):
        try:
            return subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    packages = {"sqlite": sqlite3.sqlite_version}
    for p in PACKAGES:
        try:
            packages[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    if sys.platform == "darwin":
        cpu, memory = run("sysctl", "-n", "machdep.cpu.brand_string"), run("sysctl", "-n", "hw.memsize")
    else:
        cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else None
        memory = str(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) if hasattr(os, "sysconf") else None
    return {"commit": run("git", "rev-parse", "HEAD"), "uncommitted_changes": bool(run("git", "status", "--porcelain", "--untracked-files=no")),
            "packages": packages, "python": platform.python_version(),
            "machine": {"cpu": cpu, "cores": os.cpu_count(), "memory_gb": round(int(memory) / 2**30) if memory else None, "os": platform.platform()}}


def file_hashes(paths):
    """SHA-256 (first 16 hex digits) of each file that exists, by name."""
    out = {}
    for label, path in paths.items():
        if path.exists():
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            out[label] = h.hexdigest()[:16]
    return out


def dataset_files(name):
    d, e = DATA / name / name, EMB / name
    return {"corpus.jsonl": d / "corpus.jsonl", "queries.jsonl": d / "queries.jsonl", "qrels/test.tsv": d / "qrels" / "test.tsv",
            "corpus.npy": e / "corpus.npy", "queries.npy": e / "queries.npy", "corpus_ids.json": e / "corpus_ids.json", "query_ids.json": e / "query_ids.json",
            "sparse_corpus.jsonl": e / "sparse_corpus.jsonl", "sparse_queries.jsonl": e / "sparse_queries.jsonl"}


def provenance(ds):
    return {**environment(), "data": ds.get("hashes", {}), "conditions": CONDITIONS}


def load(name):
    import numpy as np

    d = DATA / name / name
    corpus = [json.loads(l) for l in open(d / "corpus.jsonl")]
    qrels = {}
    for line in list(open(d / "qrels" / "test.tsv"))[1:]:
        q, doc, rel = line.rstrip("\n").split("\t")
        qrels.setdefault(q, {})[doc] = int(rel)
    qtext = {q["_id"]: q["text"] for q in (json.loads(l) for l in open(d / "queries.jsonl"))}
    e = EMB / name
    ids = json.load(open(e / "corpus_ids.json"))
    assert ids == [c["_id"] for c in corpus]
    qids = json.load(open(e / "query_ids.json"))
    return {
        "name": name,
        "ids": ids,
        "texts": [f"{(c.get('title') or '').strip()} {c.get('text', '')}".strip() for c in corpus],
        "vectors": np.load(e / "corpus.npy"),
        "qids": qids,
        "qtexts": [qtext[q] for q in qids],
        "qvectors": np.load(e / "queries.npy"),
        "qrels": qrels,
        "hashes": file_hashes(dataset_files(name)),
    }


def ndcg10(ranked, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall100(ranked, rel):
    want = {d for d, r in rel.items() if r > 0}
    return len(want & set(ranked[:100])) / len(want) if want else 0.0


def dir_size(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def fts_words(q):
    return WORDS.findall(q.lower())


def rrf(lists, k=60):
    s = {}
    for lst in lists:
        for rank, d in enumerate(lst):
            s[d] = s.get(d, 0.0) + 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(s.items(), key=lambda x: (-x[1], x[0]))]


def query_config(system, mode):
    """Record adapter settings, including defaults delegated to each engine.

    Change PROTOCOL when changing the retrieval or measurement rules below.
    Engine defaults are not a claim of identical candidate generation across systems.
    """
    def profile(cutoff):
        limit = cutoff + 1
        parameters = {"engine_options": "defaults"}
        if system.startswith("hinny"):
            parameters.update(snippet=0, with_text=False)
            if mode in ("hybrid", "sparse+vector", "all three"):
                # hinny's default budgets (TEXT_CANDIDATES, VECTOR_CANDIDATES), raised to the limit.
                parameters.update(fusion="default_minmax_convex", text_weight=0.5,
                                  lexical_candidate_depth=max(200, limit),
                                  vector_candidate_depth=max(100, limit))
        elif system == "sqlite fts5 + sqlite-vec" and mode == "hybrid":
            parameters.update(fusion="rrf", rrf_k=60, candidate_depth=max(4 * limit, 40))
        elif system == "lancedb":
            parameters.update(distance_type="cosine")
            if mode == "hybrid":
                parameters["fusion"] = "RRFReranker_default"
        return {"mode": mode, "metric_cutoff": cutoff, "request_limit": limit,
                "self_exclusion": "remove_query_id_then_trim",
                "parameters": parameters}

    return {"quality_and_latency": profile(10), "recall100": profile(100)}


def exclude_self(ds, i, indices, cutoff):
    qid = ds["qids"][i]
    return [ds["ids"][j] for j in indices if ds["ids"][j] != qid][:cutoff]


def evaluate(state, ds, search, mode, config):
    """Quality and latency share one query profile; deep recall has its own ranking."""
    if not ds["qids"]:
        raise ValueError("benchmark requires at least one query")
    quality = config["quality_and_latency"]
    recall = config["recall100"]
    nd, rc = [], []
    for i, qid in enumerate(ds["qids"]):
        got = search(state, ds, i, mode, quality["request_limit"])
        ranked = exclude_self(ds, i, got, quality["metric_cutoff"])
        nd.append(ndcg10(ranked, ds["qrels"][qid]))
        got = search(state, ds, i, mode, recall["request_limit"])
        ranked = exclude_self(ds, i, got, recall["metric_cutoff"])
        rc.append(recall100(ranked, ds["qrels"][qid]))

    n = min(300, len(ds["qids"]))
    warmup = min(20, n)
    for i in range(warmup):
        search(state, ds, i, mode, quality["request_limit"])
    ms = []
    for i in range(n):
        start = time.perf_counter()
        search(state, ds, i, mode, quality["request_limit"])
        ms.append((time.perf_counter() - start) * 1000)
    ms.sort()
    return {"ndcg10": round(statistics.mean(nd), 4), "recall100": round(statistics.mean(rc), 4),
            "p50_ms": round(ms[len(ms) // 2], 3),
            "p99_ms": round(ms[min(len(ms) - 1, int(len(ms) * 0.99))], 3),
            "timing": {"warmup_queries": warmup, "measured_queries": n,
                       "query_selection": "first_query_ids", "scope": "search_adapter_only"}}


# MARK: Systems. Each: build(data, dir) -> state; search(state, i, mode, k) -> [corpus index].

def hinny_build(ds, path, language=None):
    import hinny
    ns = hinny.Database(path).namespace("docs", text="text", vector=256, language=language)
    ns.upsert_many({"key": i, "text": t, "vector": v} for i, (t, v) in enumerate(zip(ds["texts"], ds["vectors"])))
    ns.optimize()
    return ns


def hinny_build_english(ds, path):
    return hinny_build(ds, path, "english")


def sparse_of(ds, kind):
    """The learned sparse encodings (embed_sparse.py), loaded on first use."""
    key = "sparse_" + kind
    if key not in ds:
        e = EMB / ds["name"]
        ds[key] = [json.loads(l) for l in open(e / f"sparse_{kind}.jsonl")]
    return ds[key]


def hinny_build_sparse(ds, path):
    import hinny
    ns = hinny.Database(path).namespace("docs", text="text", vector=256, language="english", sparse=["sp"])
    sp = sparse_of(ds, "corpus")
    ns.upsert_many({"key": i, "text": t, "vector": v, "sparse": {"sp": sp[i]}} for i, (t, v) in enumerate(zip(ds["texts"], ds["vectors"])))
    ns.optimize()
    return ns


def hinny_sparse_search(ns, ds, i, mode, k):
    kw = {"limit": k, "snippet": 0, "with_text": False, "sparse": {"sp": sparse_of(ds, "queries")[i]}}
    if mode in ("sparse+vector", "all three"):
        kw["vector"] = ds["qvectors"][i]
    if mode == "all three":
        kw["text"] = ds["qtexts"][i]
    return [h.key for h in ns.query(**kw)]


def hinny_search(ns, ds, i, mode, k):
    kw = {"limit": k, "snippet": 0, "with_text": False}
    if mode in ("text", "hybrid"):
        kw["text"] = ds["qtexts"][i]
    if mode in ("vector", "hybrid"):
        kw["vector"] = ds["qvectors"][i]
    return [h.key for h in ns.query(**kw)]


def tantivy_build(ds, path):
    import tantivy
    sb = tantivy.SchemaBuilder()
    sb.add_integer_field("i", stored=True, indexed=False)
    sb.add_text_field("body", stored=False)
    index = tantivy.Index(sb.build(), path=str(path))
    w = index.writer(heap_size=512_000_000)
    for i, t in enumerate(ds["texts"]):
        w.add_document(tantivy.Document(i=i, body=t))
    w.commit()
    w.wait_merging_threads()
    index.reload()
    return index


def tantivy_search(index, ds, i, mode, k):
    words = fts_words(ds["qtexts"][i])
    if not words:
        return []
    s = index.searcher()
    q = index.parse_query(" ".join(words), ["body"])
    return [s.doc(addr)["i"][0] for _, addr in s.search(q, k).hits]


def sqlite_build(ds, path):
    import sqlite3, sqlite_vec
    c = sqlite3.connect(str(Path(path) / "db.sqlite"), check_same_thread=False)
    c.enable_load_extension(True)
    sqlite_vec.load(c)
    c.execute("CREATE VIRTUAL TABLE fts USING fts5(body)")
    c.execute("CREATE VIRTUAL TABLE vec USING vec0(embedding float[256] distance_metric=cosine)")
    c.executemany("INSERT INTO fts(rowid, body) VALUES (?, ?)", ((i, t) for i, t in enumerate(ds["texts"])))
    c.executemany("INSERT INTO vec(rowid, embedding) VALUES (?, ?)", ((i, v.astype("float32").tobytes()) for i, v in enumerate(ds["vectors"])))
    c.commit()
    c.execute("INSERT INTO fts(fts) VALUES ('optimize')")
    c.commit()
    return c


def sqlite_search(c, ds, i, mode, k):
    def text(n):
        words = fts_words(ds["qtexts"][i])
        if not words:
            return []
        q = " OR ".join(f'"{w}"' for w in words)
        return [r[0] for r in c.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?", (q, n))]

    def vector(n):
        v = ds["qvectors"][i].astype("float32").tobytes()
        return [r[0] for r in c.execute("SELECT rowid FROM vec WHERE embedding MATCH ? AND k = ? ORDER BY distance", (v, n))]

    if mode == "text":
        return text(k)
    if mode == "vector":
        return vector(k)
    depth = max(4 * k, 40)
    return rrf([text(depth), vector(depth)])[:k]


def lance_build(ds, path):
    import lancedb, pyarrow as pa
    db = lancedb.connect(str(path))
    table = pa.table({
        "i": pa.array(range(len(ds["ids"])), pa.int64()),
        "text": pa.array(ds["texts"]),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(ds["vectors"].astype("float32").ravel()), 256),
    })
    tbl = db.create_table("docs", data=table)
    tbl.create_fts_index("text")
    return tbl


def lance_search(tbl, ds, i, mode, k):
    # Words only, as the other engines with a query syntax get them (quotes would be phrases).
    words = " ".join(fts_words(ds["qtexts"][i]))
    if mode == "text":
        q = tbl.search(words, query_type="fts")
    elif mode == "vector":
        q = tbl.search(ds["qvectors"][i], query_type="vector").distance_type("cosine")
    else:
        from lancedb.rerankers import RRFReranker
        q = tbl.search(query_type="hybrid").vector(ds["qvectors"][i]).text(words).distance_type("cosine").rerank(RRFReranker())
    return [r["i"] for r in q.limit(k).select(["i"]).to_list()]


def chroma_build(ds, path):
    import chromadb
    client = chromadb.PersistentClient(path=str(path))
    col = client.create_collection("docs", configuration={"hnsw": {"space": "cosine"}})
    n = client.get_max_batch_size()
    for s in range(0, len(ds["ids"]), n):
        e = min(s + n, len(ds["ids"]))
        col.add(ids=[str(i) for i in range(s, e)], embeddings=ds["vectors"][s:e])
    return col


def chroma_search(col, ds, i, mode, k):
    return [int(x) for x in col.query(query_embeddings=[ds["qvectors"][i]], n_results=k, include=[])["ids"][0]]


def qdrant_build(ds, path):
    from qdrant_client import QdrantClient, models
    c = QdrantClient(path=str(path))
    c.create_collection("docs", vectors_config=models.VectorParams(size=256, distance=models.Distance.COSINE))
    c.upload_collection("docs", vectors=ds["vectors"], ids=list(range(len(ds["ids"]))), batch_size=1024)
    return c


def qdrant_search(c, ds, i, mode, k):
    return [p.id for p in c.query_points("docs", query=ds["qvectors"][i], limit=k).points]


SYSTEMS = [
    ("hinny", hinny_build, hinny_search, ["text", "vector", "hybrid"]),
    ("hinny (english)", hinny_build_english, hinny_search, ["text", "hybrid"]),
    ("hinny (sparse)", hinny_build_sparse, hinny_sparse_search, ["sparse", "sparse+vector", "all three"]),
    ("tantivy-py", tantivy_build, tantivy_search, ["text"]),
    ("sqlite fts5 + sqlite-vec", sqlite_build, sqlite_search, ["text", "vector", "hybrid"]),
    ("lancedb", lance_build, lance_search, ["text", "vector", "hybrid"]),
    ("chroma", chroma_build, chroma_search, ["vector"]),
    ("qdrant (local mode)", qdrant_build, qdrant_search, ["vector"]),
]


def run(name, only=None):
    ds = load(name)
    rows = []
    for sys_name, build, search, modes in SYSTEMS:
        if only and sys_name not in only:
            continue
        path = Path(tempfile.mkdtemp(prefix="beir-", dir=OUT))
        try:
            if build is hinny_build_sparse:
                # Inputs are in memory before the clock starts, as the texts and vectors are.
                sparse_of(ds, "corpus"), sparse_of(ds, "queries")
            t = time.perf_counter()
            state = build(ds, path)
            built = time.perf_counter() - t
            size = dir_size(path)
            for mode in modes:
                config = query_config(sys_name, mode)
                row = {"protocol": PROTOCOL, "query_config": config,
                       "dataset": name, "system": sys_name, "mode": mode,
                       **evaluate(state, ds, search, mode, config),
                       "build_s": round(built, 2), "size_mb": round(size / 1e6, 1), "docs": len(ds["ids"]), "queries": len(ds["qids"]),
                       "provenance": provenance(ds)}
                rows.append(row)
                print(json.dumps(row), flush=True)
        except Exception as e:
            row = {"protocol": PROTOCOL, "dataset": name, "system": sys_name, "error": f"{type(e).__name__}: {e}"}
            rows.append(row)
            print(json.dumps(row), flush=True)
            traceback.print_exc()
        finally:
            state = None
            shutil.rmtree(path, ignore_errors=True)
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise SystemExit("usage: run_bench.py DATA EMB OUT dataset [dataset ...]")
    DATA, EMB, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    OUT.mkdir(parents=True, exist_ok=True)
    only = os.environ.get("ONLY", "").split(",") if os.environ.get("ONLY") else None
    with open(OUT / "results.jsonl", "a") as f:
        for name in sys.argv[4:]:
            for row in run(name, only):
                f.write(json.dumps(row) + "\n")
