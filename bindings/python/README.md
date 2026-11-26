# hinny for Python

**An embedded hybrid search database for Python.** Combines full-text search (BM25), dense vectors, learned sparse terms, and typed attribute filters in-process on local files. Text and vectors run together under one filter and fuse into a single ranking—no server, no network overhead. A hybrid query over 100,000 documents takes 0.17 ms from Python.

```python
import hinny
from hinny import attr

db = hinny.Database("data")
docs = db.namespace(
    "docs",
    text={"title": 2.0, "body": 1.0},  # BM25 fields with boosts
    ints=["year"],
    strings=["kind"],                  # typed attributes for filters and results
    vector=384,                        # embeddings (cosine by default)
)

docs.upsert(
    1,
    text={"title": "Lease agreement", "body": "The apartment on Elm Street"},
    attrs={"year": 2025, "kind": "pdf"},
    vector=embed("Lease agreement …"),
)

hits = docs.query(
    text="apartment lease",
    vector=embed("where's my lease?"),
    filter=(attr("year") >= 2025) & (attr("kind") == "pdf"),
    limit=10,
)
for h in hits:
    print(h.key, h.score, h.attrs, h.snippet)  # also h.text, h.highlights, h.text_score, h.vector_score
```

## For retrieval-augmented generation

Store chunks as documents with source metadata as attributes, and query with the user's question as both text and vector:

```python
chunks = db.namespace("chunks", text="text", ints=["page"], strings=["source"], vector=768)
chunks.upsert_many(
    {"key": hinny.key(f"{path}#{i}"), "text": chunk, "attrs": {"source": path, "page": page}, "vector": vec}
    for i, (chunk, page, vec) in enumerate(split_and_embed(path))
)

hits = chunks.query(text=question, vector=embed(question), filter=attr("source").prefix("handbook/"), limit=8)
context = "\n\n".join(h.text["text"] for h in hits)
```

Across five BEIR datasets with identical embeddings, Hinny's hybrid search (`language="english"`) ranked first on all five (nDCG@10: 0.817 on SciFact, 0.377 on NFCorpus, 0.603 on ArguAna, 0.183 on SciDocs, 0.336 on FiQA vs. 0.807, 0.369, 0.579, 0.177, and 0.319 for LanceDB) in 0.10–0.29 ms—11–21× faster than LanceDB, with quality and latency measured from the same queries. Adding an inference-free sparse field lifted FiQA further to 0.370. On a private mailbox benchmark (23,000 emails, 119 questions), hybrid search reached 48.7% recall@1 vs. 35.3% for BM25 alone. Full benchmarks are in [`benchmarks/beir`](../../benchmarks/beir).

## API overview

### Databases, namespaces, and schemas

- `Database(path)` manages namespaces in a directory. `db.namespace(name, text=…, ints=…, floats=…, strings=…, tags=…, sparse=…, vector=dims, metric="cosine", language=…, stopwords=…)` opens or creates a namespace. Schemas can grow over time by adding fields or attributes.
- `db.open(name)` opens an existing namespace with its stored schema; `db.namespaces()` lists namespace names; `db.delete(name)` deletes a closed namespace (`del` any `Namespace` and `Snapshot` handles first, or it raises `OSError`).
- `language="english"` (or any of 18 Snowball languages) stems text fields and drops stopwords by default for English and German (`stopwords=False` keeps them). Analyzers are fixed once the namespace holds data.
- `metric` selects `"cosine"` (default), `"euclidean"`, or `"dot"`, fixed once set (`ns.schema["metric"]` reports it; reopening with a different metric raises `ValueError`). Higher scores always mean closer matches: cosine similarity, negated squared distance (`0` for an exact match), or dot product. Under `"euclidean"` and `"dot"`, vectors are stored as given without normalization.
- Attributes support integers (`ints`), floats (`floats`), strings (`strings`), and multi-string tags (`tags`, passed as a list/tuple/set and returned as a sorted list). Integers passed to float attributes are converted automatically.

### Writes, mutations, and durability

- `ns.upsert(key, text=…, attrs=…, vector=…, sparse=…, version=…)`: `text` is a string (single text field) or `dict[field, str]`; `vector` is a NumPy array (`float32` is read in place) or sequence of numbers. Writes are immediately searchable and survive a process crash on return; the WAL is OS-buffered until `ns.sync()`.
- `ns.upsert_many(docs)`: commits an iterable of document dicts as one atomic batch.
- `ns.apply(upsert=[...], delete=[keys])`: executes upserts and deletes together as one atomic batch (`ns.delete(key)` and `ns.delete_many(keys)` for deletes alone).
- `ns.delete_where(filter)`: deletes all documents matching a filter in one atomic batch and returns the deleted count.
- `ns.patch(key, text=…, attrs=…, vector=…, sparse=…, version=…, remove=[...], condition=…)`: replaces specified fields/attributes and drops names in `remove` while keeping the rest. Because sparse term maps are not stored raw, patching a namespace with sparse fields must supply `sparse=` or list the field in `remove` (otherwise raises `ValueError`). Returns `False` if the document does not exist or `condition` fails. `ns.patch_where(filter, ...)` patches all matching documents and returns how many were updated. Patching reads and rewrites the document (requiring stored text).
- Conditional writes: `ns.upsert_if(key, condition, ...)` writes if the key does not exist or the existing document passes `condition`; `ns.delete_if(key, condition)` deletes only if it passes. The check and write execute atomically under the writer lock and return whether the write occurred.
- `ns.sync()` flushes all prior WAL writes to storage without building segments; `ns.flush()` also materializes memtables into segments. Check I/O errors explicitly—cleanup on drop is best-effort. Writes after the last successful sync may be lost on OS crash or power loss. File and Unix directory syncs depend on filesystem/device support; non-Unix directory entries are not synced.
- `ns.fork("sandbox")`: creates an independent copy-on-write namespace using hard links—takes milliseconds regardless of size and uses no extra disk until the two diverge.

### Queries, filters, and aggregations

- `ns.query(text=…, vector=…, sparse=…, filter=…, limit=10, order_by="-year")`: runs text, vector, sparse, or any hybrid combination. With only a filter, returns matching documents ordered by `order_by`.
- Search-as-you-type: `ns.query(text="tax ret", typing=32)` expands the last word as a prefix across the 32 most frequent matching terms.
- Hybrid fusion: scores from each retriever are min-max scaled to `0–1` and blended with `text_weight=0.5` by default (lower for strong domain embeddings, higher when exact words matter). Pass `fusion="rrf"` for Reciprocal Rank Fusion.
- Candidate budgets & diagnostics: `text_candidates=200`, `vector_candidates=100`, `sparse_candidates=200` control how many candidates each retriever gathers before fusion (automatically raised if `limit` exceeds them). `ns.explain(...)` returns `{"hits": [...], "fusion": {...}}` with per-retriever depths, counts, score ranges, and overlap.
- `nprobe`: overrides how many vector clusters each segment probes (`0`, the default, uses each segment's self-calibrated budget for 0.997 recall@10).
- Learned sparse fields: declare `sparse=["terms"]`, pass `sparse={"terms": {"lease": 1.4, "rent": 0.6}}` at upsert, and query with `sparse={"terms": {...}}`. Scored by dot product on the same block-max posting lists as BM25; weights are stored to hundredths.
- Filters: `attr("kind") == "pdf"`, `!=`, `>=`, `>`, `<=`, `<`, `.between(a, b)`, `.one_of([...])`, `.prefix("…")`, `.contains("…")`, `.glob("/home/*/taxes/*.pdf")`, `.iglob("…")`, `.exists()`, combined with `&`, `|`, `~`. Globs match full strings (`*` any chars, `?` one char, `\` escape; `iglob` is case-insensitive). Tag filters add `.has("tax")`, `.has_any([...])`, and `.has_all([...])`.
- Aggregations: `ns.count(filter)` counts matching documents from attribute columns; `ns.count_by("kind", filter, limit)` returns `[(value, count), ...]` sorted by frequency descending (documents with tags count once per tag).
- Snapshots & utilities: `ns.snapshot()` pins a point-in-time read view (`snap.seq`, `query`, `get`, `text`, `versions`); `ns.get(key)`, `ns.text(key)`, `ns.versions()`, `len(ns)`, `ns.optimize()`, `ns.stats()`, `ns.verify()`, and `hinny.key("any string")` for stable 64-bit integer keys.
- Resource bounds: pass `timeout=` (seconds) on `query`/`explain` to raise `TimeoutError` instead of returning partial results. `ns.set_limits(max_batch_bytes=…, flush_bytes=…)` configures batch limits (default 1 GiB) and memtable flush thresholds (default 256 MiB).

### Threading, keys, and stored text

Writes and queries release the GIL: queries run in parallel across threads while writes serialize cleanly. Keys are unsigned integers (`0` to `2^63`); use `hinny.key()` to hash string IDs.

Document text is stored by default (`store_text=True`). Each hit includes `hit.text` (field → text), a ~240-character `hit.snippet` from the best-matching field, and `hit.highlights` character spans (`hit.snippet[a:b]`). Pass `snippet=0` or `with_text=False` to skip them per query.

## Performance

From `examples/bench.py` on an Apple M5 Pro: 100,000 documents with 384-dimensional vectors, queried from Python after `optimize()`.

| Mode | p50 | p99 | 1 thread | 8 threads |
| :--- | ---: | ---: | ---: | ---: |
| Text | 0.13 ms | 0.15 ms | 7,800 q/s | 42,400 q/s |
| Vector | 0.10 ms | 0.13 ms | 10,500 q/s | 60,700 q/s |
| Hybrid | 0.17 ms | 0.21 ms | 5,700 q/s | 29,000 q/s |
| Hybrid + filter (10% pass) | 0.24 ms | 0.28 ms | 4,200 q/s | 17,300 q/s |

Includes snippet extraction from zstd-compressed blob files. Vector recall@10 against NumPy brute force is 0.999. `upsert_many` ingests 150,000–180,000 documents/s; `optimize()` completes in 0.4 s.

## Examples

- `examples/rag.py`: end-to-end RAG retrieval over this repository's docs (uses `sentence-transformers` if installed, or a hashing fallback).
- `examples/bench.py`: reproduces the throughput and latency table above.

## Install

The `hinny` package on PyPI is currently an empty `0.0.1` placeholder. Build from source (requires **Python 3.11+** and **Rust 1.89+**):

```bash
pip install ./bindings/python
```

Or build a reusable wheel with [maturin](https://www.maturin.rs):

```bash
cd bindings/python && maturin build --out dist && pip install dist/hinny-*.whl
```

For local development, `maturin develop` installs into your active virtual environment. See [RELEASING.md](RELEASING.md) for wheel builds and release details.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
