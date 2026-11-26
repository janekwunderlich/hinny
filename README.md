<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/hinny-mark-dark.svg">
    <img src="brand/hinny-mark-light.svg" alt="Hinny" width="88">
  </picture>
</p>

<h1 align="center">Hinny</h1>

<p align="center">
  <strong>An embedded hybrid search database that travels with your application.</strong>
</p>

<p align="center">
  <a href="https://hinny.dev">Website</a> ·
  <a href="https://hinny.dev/docs">Docs</a> ·
  <a href="https://hinny.dev/benchmarks">Benchmarks</a> ·
  <a href="DESIGN.md">Design</a>
</p>

---

Hinny runs inside your process on memory-mapped local files—no server to deploy or manage. A single query combines full-text search (BM25), dense vectors, learned sparse terms, and typed attribute filters, fused into one ranking. Every write is searchable the moment the call returns.

Built for local RAG, AI agents, desktop apps, and CLI tools where search belongs in-process rather than behind a network call.

> **Alpha (`0.3.0`)** — The API and file format are still evolving. The `hinny` packages on crates.io, PyPI, and npm are empty `0.0.1` placeholders; build from source below.

## Why Hinny

* **True hybrid in one pass** — Runs BM25, dense vectors, and learned sparse terms together under one filter and fuses scores in-engine (50/50 min-max scaling by default, or RRF).
* **Sub-millisecond on one machine** — Immutable memory-mapped segments, SIMD block-max BM25, and 1-bit RaBitQ vectors that rerank <0.5% of candidates at half precision.
* **Live and crash-safe** — Atomic batches through a checksummed WAL, lock-free MVCC reads, explicit `sync()`/`flush()` boundaries, and background merges.
* **Millisecond namespace forks** — Copy-on-write forks via hard links with zero extra disk until they diverge—built for agent sandboxes and safe migrations.
* **Embedded first, server optional** — Native Rust, Python, Node.js, and C interfaces, plus an optional HTTP server (`hinny-server`) with read replicas.

## Quick start

```python
import hinny
from hinny import attr

docs = hinny.Database("data").namespace(
    "docs",
    text={"title": 2.0, "body": 1.0},
    ints=["year"], strings=["kind"],
    vector=384,
)

docs.upsert(
    1,
    text={"title": "Lease agreement", "body": "The apartment on Elm Street"},
    attrs={"year": 2025, "kind": "pdf"},
    vector=embedding,
)

hits = docs.query(
    text="apartment lease",
    vector=query_embedding,
    filter=(attr("year") >= 2025) & (attr("kind") == "pdf"),
    limit=10,
)
```

<details>
<summary>Rust, Node.js, and C examples</summary>

**Rust** (`cargo run --release -p hinny --example quickstart`)

```rust
use hinny::{Database, Document, Schema, attr};

let db = Database::open("data")?;
let docs = db.namespace("docs", Schema::new()
    .text("title", 2.0).text("body", 1.0)
    .int("year").string("kind")
    .vector(384))?;

docs.upsert(Document::new(1)
    .text("title", "Lease agreement").text("body", "The apartment on Elm Street")
    .int("year", 2025).string("kind", "pdf")
    .vector(embedding))?;
docs.sync()?;

let hits = docs.query()
    .text("apartment lease")
    .vector(query_embedding)
    .filter(attr("year").gte(2025) & attr("kind").eq("pdf"))
    .limit(10)
    .run()?;
```

**Node.js** ([`bindings/node`](bindings/node))

```js
const { Database, attr } = require('hinny')

const docs = new Database('data').namespace('docs', {
  text: { title: 2, body: 1 }, ints: ['year'], strings: ['kind'], vector: 384,
})
docs.upsert({
  key: 1,
  text: { title: 'Lease agreement', body: 'The apartment on Elm Street' },
  attrs: { year: 2025, kind: 'pdf' },
  vector: embedding,
})
const hits = docs.query({
  text: 'apartment lease',
  vector: queryEmbedding,
  filter: attr('year').gte(2025).and(attr('kind').eq('pdf')),
  limit: 10,
})
```

**C** ([`crates/hinny-c`](crates/hinny-c) — single header `hinny.h`, JSON in/out + `float*` vectors)

```c
hinny_db *db = hinny_open("data", &err);
hinny_namespace *docs = hinny_namespace_open(db, "docs",
    "{\"text\": [{\"name\": \"body\"}], \"ints\": [\"year\"], \"vector\": 384}", &err);
hinny_upsert(docs, "{\"key\": 1, \"text\": {\"body\": \"The apartment on Elm Street\"}, "
                   "\"ints\": {\"year\": 2025}}", embedding, 384, &err);
char *hits = hinny_query(docs, "{\"text\": \"apartment lease\", "
                   "\"filter\": {\"range\": [\"year\", 2025, null]}}", query, 384, &err);
```

</details>

## Benchmarks

*Apple M5 Pro, single-threaded, warm cache. Full methodology and charts at [hinny.dev/benchmarks](https://hinny.dev/benchmarks) and [`benchmarks/beir`](benchmarks/beir).*

| Benchmark | Dataset | Hinny | Comparison |
| :--- | :--- | :--- | :--- |
| **Hybrid latency** | BEIR FiQA (57,638 docs) | **0.29 ms** p50 | 11–21× faster than LanceDB across all 5 BEIR datasets |
| **Hybrid quality** | 5 BEIR datasets (English) | **#1 on all 5** | Tested vs. LanceDB, SQLite FTS5 + sqlite-vec, Chroma, Qdrant local |
| **Vector latency** | 1M × 256-d (synthetic) | **0.13 ms** p50 | **0.997** recall@10 (0.04% reranked at half precision) |
| **Full-text p99** | ~1.1M passages | **7.8 ms** | Tantivy: 34.5 ms p99 (identical BM25 scores) |
| **Private mailbox**\* | 23,000 emails, 119 queries | **48.7%** R@1 | BM25 alone: 35.3%; RRF: 38.7% |

*\*Private dataset; not externally reproducible.*

<details>
<summary>Detailed benchmark tables (BEIR, vectors, full-text, file names)</summary>

### BEIR (EmbeddingGemma 2, 256-d, protocol `beir-v2` from Python)

| Hybrid (nDCG@10, p50) | SciFact | NFCorpus | ArguAna | SciDocs | FiQA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hinny, sparse + vectors** | **0.824**, 0.12 ms | 0.373, **0.10 ms** | **0.611**, 0.28 ms | 0.186, **0.20 ms** | **0.370**, 0.35 ms |
| **Hinny, English + sparse + vectors** | **0.824**, 0.12 ms | **0.381**, **0.10 ms** | 0.606, 0.46 ms | **0.188**, 0.23 ms | 0.365, 0.39 ms |
| Hinny, English + vectors | 0.817, **0.11 ms** | 0.377, **0.10 ms** | 0.603, **0.24 ms** | 0.183, **0.20 ms** | 0.336, **0.29 ms** |
| Hinny, plain text + vectors | 0.815, **0.11 ms** | 0.364, **0.10 ms** | 0.595, 0.42 ms | 0.182, **0.20 ms** | 0.337, 0.39 ms |
| LanceDB | 0.807, 1.9 ms | 0.369, 1.6 ms | 0.579, 2.7 ms | 0.177, 3.7 ms | 0.319, 6.2 ms |
| SQLite (FTS5 + sqlite-vec) | 0.768, 5.3 ms | 0.364, 1.1 ms | 0.573, 248 ms | 0.173, 22 ms | 0.322, 57 ms |

| Vector (nDCG@10, p50) | SciFact | NFCorpus | ArguAna | SciDocs | FiQA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hinny** (approximate) | **0.833**, **0.03 ms** | **0.352**, **0.05 ms** | **0.574**, **0.03 ms** | **0.167**, **0.08 ms** | 0.290, **0.12 ms** |
| Chroma (HNSW) | **0.833**, 0.31 ms | 0.348, 0.31 ms | **0.574**, 0.39 ms | **0.167**, 0.42 ms | 0.286, 0.58 ms |
| Exact (LanceDB, sqlite-vec, Qdrant local) | **0.833**, 0.9–1.4 ms | 0.351–**0.352**, 0.6–1.1 ms | 0.572–**0.574**, 1.5–2.1 ms | **0.167**, 3.3–6.2 ms | **0.292**, 5.5–14 ms |

### Vector recall (calibrated default vs. exact scan)

| Collection | Default | Exact | Reranked | Index |
| :--- | :--- | :--- | ---: | :--- |
| FiQA (57,638 docs, 648 queries) | **0.992**, p50 0.10 ms, p99 0.22 ms | 0.78 ms | 0.39% | 34 MB |
| Private mailbox (23,037 emails) | **0.998**, p50 0.04 ms, p99 0.07 ms | 0.31 ms | 0.34% | 14 MB |
| 1M × 256-d (synthetic clustered) | **0.997**, p50 0.13 ms, p99 0.25 ms | 13 ms | 0.04% | 596 MB (512 MB f16) |

### Full text (~1.1M passages, identical BM25 scores)

| Metric | Hinny | Tantivy 0.27 | SQLite FTS5 |
| :--- | :--- | :--- | :--- |
| Agent question, OR, top 200 (p99) | **7.8 ms** | 34.5 ms | 683 ms |
| Agent question, filler dropped (p99) | **1.4 ms** | 3.1 ms | 127 ms |
| 1–3 words, AND (p99) | **0.33 ms** | 0.55 ms | 18–20 ms |
| Prefix, 32 completions (p99) | **1.9 ms** | 6.8 ms | 68 ms |
| Index size / Build time | 175 MB / **2.3 s** | 175 MB / 7.1 s | 2.58 GB / 59 s |

### File names (macOS, 9.2M paths)

| Metric | Hinny (`hinny-names`) | SQLite |
| :--- | :--- | :--- |
| Search-as-you-type | **0.08 ms** p50, ~2 ms p99 | FTS5 trigram 0.5–320 ms; `LIKE` 0.4–1.9 s |
| Single letter | **0.003–0.006 ms** | 250–320 ms |
| Freshness / Index size | **~50 ms** / 722 MB | — / 1.49 GB |

</details>

## Features

| Area | Capabilities |
| :--- | :--- |
| **Full text** | BM25 with block-max MaxScore, Snowball stemming in 18 languages (stopwords dropped by default for English & German), search-as-you-type (`typing`), stored text & highlighted snippets |
| **Vectors** | 1-bit RaBitQ + FastScan SIMD, self-calibrated probe budgets (~0.997 recall@10), **cosine**, **Euclidean**, and **dot product** metrics |
| **Learned sparse** | SPLADE / inference-free sparse term weights scored by dot product on shared posting lists |
| **Filters & counts** | Equality, ranges, sets, prefix, contains, `glob`/`iglob`, tag membership (`has`, `has_any`, `has_all`), boolean logic evaluated inside search; columnar `count()` and `count_by()` |
| **Mutations & MVCC** | Atomic batches, `patch`/`patch_where`, conditional writes (`upsert_if`, `patch_if`, `delete_if`), `delete_where`, point-in-time `snapshot()`, millisecond `fork()` |
| **Server & extras** | Optional HTTP server ([`hinny-server`](crates/hinny-server)) with read replicas ([`hinny-replica`](crates/hinny-replica)); macOS file-name index ([`hinny-names`](crates/hinny-names)) |

## Building from source

Requires **Rust 1.89+**:

```bash
git clone https://github.com/janekwunderlich/hinny.git
cd hinny
cargo build --release
```

| Interface | How to use from source | Docs |
| :--- | :--- | :--- |
| **Rust** | `hinny = { git = "https://github.com/janekwunderlich/hinny", tag = "v0.3.0" }` | [`crates/hinny`](crates/hinny) |
| **Python** (3.11+) | `pip install ./bindings/python` (zero-copy NumPy, releases GIL) | [`bindings/python`](bindings/python) |
| **Node.js** | `cd bindings/node && npm ci && npm run build` (TypeScript types included) | [`bindings/node`](bindings/node) |
| **C** | `cargo build --release -p hinny-c` (`hinny.h` for Swift/Kotlin/C FFI) | [`crates/hinny-c`](crates/hinny-c) |
| **HTTP** | `cargo run --profile server -p hinny-server -- --dir data` | [`crates/hinny-server`](crates/hinny-server) |

## Not yet

Not yet implemented: multiple vectors per document, phrase matching, regex filters, and native boolean, datetime, or number-array attribute types. See [DESIGN.md](DESIGN.md) for architecture details, durability guarantees, and roadmap notes.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
