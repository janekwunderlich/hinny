# hinny

Core Rust library for Hinny—an embedded hybrid search database combining full text (BM25), 1-bit RaBitQ vectors reranked at half precision, learned sparse terms, and typed attribute filters over a write-ahead log, immutable memory-mapped segments, and background merges.

See the root [README.md](../../README.md) for quick start, API examples, and benchmarks, and [DESIGN.md](../../DESIGN.md) for architecture details.

## Module layout

| Module | Responsibility |
| :--- | :--- |
| `api` | Public API: `Database`, `Namespace`, `Schema`, `Document`, `Query`, `Patch`, filters (`attr`) |
| `index` | Live LSM index: WAL, memtables, checkpoints, background merges, MVCC views, queries |
| `segment` | Immutable segment format: posting lists with impacts, dictionaries, columns, vectors |
| `search` | BM25 block-max MaxScore and term-at-a-time scoring |
| `vector` | Randomized Hadamard rotation, RaBitQ codes, FastScan kernels, cluster budgets |
| `attrs` | Columnar attribute storage, zone maps, and SIMD filter evaluation |

## Common commands

```bash
cargo test --release
cargo run --release --example quickstart
cargo run --release --example vector_bench -- synthetic 1000000 256
cargo run --release --features bench --bin text-bench -- build passages.tsv 1 DIR   # against Tantivy
```
