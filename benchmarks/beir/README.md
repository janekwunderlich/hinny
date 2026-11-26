# BEIR: hinny and other embedded databases

Five [BEIR](https://github.com/beir-cellar/beir) datasets, searched by six embedded databases with
the same documents, the same vectors and the same queries, and scored by the same code.

| | Documents | Test queries |
|---|---|---|
| SciFact | 5,183 | 300 |
| NFCorpus | 3,633 | 323 |
| ArguAna | 8,674 | 1,406 |
| SciDocs | 25,657 | 1,000 |
| FiQA | 57,638 | 648 |

## Current protocol: `beir-v2`

Run on 2026-10-24 on an Apple M5 Pro, every system on all five datasets at commit `f9cc7f1`, then Hinny's rows again at `cc059d7` (its term-at-a-time fix) and on 2026-10-25 at `03bb9f0` (an exact top k of items, depth-scaled vector probing)
(`results.jsonl`, the rows with `"protocol": "beir-v2"`). Hinny's hybrid ranks first of the
databases tested on all five, and answers in 0.10–0.29 ms; LanceDB's takes 1.6–6.2 ms and
SQLite's 1.1–248 ms. Best per dataset and kind of search in bold.

| nDCG@10 (request 11) | SciFact | NFCorpus | ArguAna | SciDocs | FiQA |
|---|---|---|---|---|---|
| Text: hinny | 0.662 | 0.307 | 0.458 | 0.149 | 0.238 |
| Text: hinny, English analyzer | **0.688** | **0.328** | **0.491** | **0.157** | **0.256** |
| Text: Tantivy (tantivy-py) | 0.661 | 0.308 | 0.404 | 0.151 | 0.237 |
| Text: SQLite (FTS5, sqlite-vec) | 0.668 | 0.307 | 0.472 | 0.151 | 0.234 |
| Text: LanceDB | 0.684 | 0.327 | **0.491** | **0.157** | 0.253 |
| Learned sparse: hinny, a sparse field | 0.708 | 0.345 | 0.521 | 0.163 | 0.358 |
| Vector: hinny | **0.833** | **0.352** | **0.574** | **0.167** | 0.290 |
| Vector: SQLite (FTS5, sqlite-vec) | **0.833** | 0.351 | 0.572 | **0.167** | **0.292** |
| Vector: LanceDB | **0.833** | **0.352** | **0.574** | **0.167** | **0.292** |
| Vector: Chroma | **0.833** | 0.348 | **0.574** | **0.167** | 0.286 |
| Vector: Qdrant (local mode) | **0.833** | 0.351 | 0.572 | **0.167** | **0.292** |
| Hybrid: hinny | 0.815 | 0.364 | 0.595 | 0.182 | 0.337 |
| Hybrid: hinny, English analyzer | 0.817 | 0.377 | 0.603 | 0.183 | 0.336 |
| Hybrid: hinny, sparse field + vectors | **0.824** | 0.373 | **0.611** | 0.186 | **0.370** |
| Hybrid: hinny, English + sparse + vectors | **0.824** | **0.381** | 0.606 | **0.188** | 0.365 |
| Hybrid: SQLite (FTS5, sqlite-vec) | 0.768 | 0.364 | 0.573 | 0.173 | 0.322 |
| Hybrid: LanceDB | 0.807 | 0.369 | 0.579 | 0.177 | 0.319 |

| p50, request 11, matching nDCG@10 | SciFact | NFCorpus | ArguAna | SciDocs | FiQA |
|---|---|---|---|---|---|
| Text: hinny | **0.04 ms** | **0.01 ms** | 0.34 ms | 0.10 ms | 0.26 ms |
| Text: hinny, English analyzer | **0.04 ms** | **0.01 ms** | **0.17 ms** | **0.07 ms** | **0.10 ms** |
| Text: Tantivy (tantivy-py) | 0.07 ms | 0.02 ms | 0.54 ms | 0.19 ms | 0.47 ms |
| Text: SQLite (FTS5, sqlite-vec) | 3.85 ms | 0.16 ms | 252 ms | 15 ms | 42 ms |
| Text: LanceDB | 0.86 ms | 0.80 ms | 1.66 ms | 1.00 ms | 1.18 ms |
| Learned sparse: hinny, a sparse field | 0.04 ms | 0.02 ms | 0.22 ms | 0.12 ms | 0.25 ms |
| Vector: hinny | **0.03 ms** | **0.05 ms** | **0.03 ms** | **0.08 ms** | **0.12 ms** |
| Vector: SQLite (FTS5, sqlite-vec) | 1.24 ms | 0.83 ms | 2.04 ms | 6.25 ms | 14 ms |
| Vector: LanceDB | 1.38 ms | 1.13 ms | 2.06 ms | 3.29 ms | 5.47 ms |
| Vector: Chroma | 0.31 ms | 0.31 ms | 0.39 ms | 0.42 ms | 0.58 ms |
| Vector: Qdrant (local mode) | 0.87 ms | 0.61 ms | 1.53 ms | 4.34 ms | 9.77 ms |
| Hybrid: hinny | **0.11 ms** | **0.10 ms** | 0.42 ms | **0.20 ms** | 0.39 ms |
| Hybrid: hinny, English analyzer | **0.11 ms** | **0.10 ms** | **0.24 ms** | **0.20 ms** | **0.29 ms** |
| Hybrid: hinny, sparse field + vectors | 0.12 ms | **0.10 ms** | 0.28 ms | **0.20 ms** | 0.35 ms |
| Hybrid: hinny, English + sparse + vectors | 0.12 ms | **0.10 ms** | 0.46 ms | 0.23 ms | 0.39 ms |
| Hybrid: SQLite (FTS5, sqlite-vec) | 5.25 ms | 1.12 ms | 248 ms | 22 ms | 57 ms |
| Hybrid: LanceDB | 1.89 ms | 1.64 ms | 2.71 ms | 3.67 ms | 6.20 ms |

| Recall@100, separate request 101 | SciFact | NFCorpus | ArguAna | SciDocs | FiQA |
|---|---|---|---|---|---|
| Text: hinny | 0.886 | 0.236 | 0.945 | 0.348 | 0.510 |
| Text: hinny, English analyzer | **0.921** | **0.247** | 0.964 | 0.363 | 0.552 |
| Text: Tantivy (tantivy-py) | 0.889 | 0.237 | 0.907 | 0.348 | 0.510 |
| Text: SQLite (FTS5, sqlite-vec) | 0.886 | 0.233 | 0.958 | 0.349 | 0.517 |
| Text: LanceDB | **0.921** | **0.247** | **0.967** | **0.365** | **0.554** |
| Learned sparse: hinny, a sparse field | 0.938 | 0.278 | 0.982 | 0.374 | 0.640 |
| Vector: hinny | **0.967** | **0.352** | **0.979** | **0.385** | **0.623** |
| Vector: SQLite (FTS5, sqlite-vec) | **0.967** | **0.352** | **0.979** | **0.385** | **0.623** |
| Vector: LanceDB | **0.967** | **0.352** | **0.979** | **0.385** | **0.623** |
| Vector: Chroma | **0.967** | 0.347 | **0.979** | 0.383 | 0.617 |
| Vector: Qdrant (local mode) | **0.967** | **0.352** | **0.979** | **0.385** | **0.623** |
| Hybrid: hinny | 0.977 | 0.350 | 0.985 | 0.405 | 0.623 |
| Hybrid: hinny, English analyzer | 0.977 | 0.351 | 0.989 | 0.410 | 0.644 |
| Hybrid: hinny, sparse field + vectors | 0.970 | 0.336 | **0.991** | 0.410 | **0.678** |
| Hybrid: hinny, English + sparse + vectors | 0.980 | 0.343 | 0.989 | **0.419** | 0.668 |
| Hybrid: SQLite (FTS5, sqlite-vec) | 0.980 | **0.354** | 0.989 | **0.419** | 0.654 |
| Hybrid: LanceDB | **0.987** | 0.353 | 0.989 | 0.411 | 0.651 |

| Build, size | SciFact | NFCorpus | ArguAna | SciDocs | FiQA |
|---|---|---|---|---|---|
| hinny | 0.4 s, 8 MB | 0.1 s, 6 MB | 0.2 s, 12 MB | 0.4 s, 36 MB | 0.7 s, 67 MB |
| hinny, English analyzer | 0.2 s, 8 MB | 0.2 s, 6 MB | 0.2 s, 11 MB | 0.6 s, 34 MB | 1.1 s, 64 MB |
| hinny, English + sparse field | 0.4 s, 11 MB | 0.3 s, 8 MB | 0.5 s, 16 MB | 1.4 s, 47 MB | 2.6 s, 90 MB |
| Tantivy (tantivy-py) | 0.5 s, 3 MB | 0.5 s, 2 MB | 0.5 s, 4 MB | 0.7 s, 11 MB | 0.7 s, 18 MB |
| SQLite (FTS5, sqlite-vec) | 0.2 s, 23 MB | 0.1 s, 17 MB | 0.2 s, 29 MB | 0.7 s, 83 MB | 1.4 s, 141 MB |
| LanceDB | 1.0 s, 12 MB | 0.2 s, 9 MB | 0.4 s, 16 MB | 0.5 s, 50 MB | 0.7 s, 93 MB |
| Chroma | 0.6 s, 14 MB | 0.2 s, 10 MB | 0.5 s, 19 MB | 1.4 s, 41 MB | 4.8 s, 84 MB |
| Qdrant (local mode) | 2.2 s, 21 MB | 1.1 s, 15 MB | 3.8 s, 36 MB | 9.2 s, 106 MB | 18.3 s, 239 MB |

The runner uses the same request for nDCG@10 and warm latency: request **11** results, remove a document whose ID equals the query
ID, then keep the first **10**. Requesting the extra result avoids losing the tenth result when
the query is itself in the corpus. The same request limit is used even when no self ID exists.
Recall@100 is measured separately by requesting **101**, excluding the self ID, and keeping
**100**. Its deeper ranking is never used for nDCG@10 and its latency is not reported.

Each new row records `protocol`, both profiles in `query_config`, the timing sample in
`timing`, and its `provenance`: the commit (and whether the tree had uncommitted changes),
package versions, Python, the machine, SHA-256 prefixes of the dataset and embedding files,
and the conditions it was timed under. The 2026-10-24 rows had theirs added after the run, from
the same machine, environment and files. Hinny's hybrid candidates are its defaults, 200 lexical / 100 vector, for the request
of 11, and 200 / 101 for the request of 101 (fixed budgets, raised to the limit; chosen on dev
splits with `candidates.py`); SQLite's hybrid uses 44 / 44 and 404 / 404. This protocol
aligns quality with latency within each configuration; it does not make the engines' candidate
generation identical. Unknown engine defaults remain explicitly labeled as defaults.

Quality covers all supplied test queries. Latency covers the first at most 300 query IDs, with
the first at most 20 used for warm-up, after the quality passes. Calls are sequential from
Python; engine-internal parallelism is unchanged. Timing includes the search adapter and result
ID extraction, but excludes self-ID filtering and metric calculation. Small warm datasets and
test-set-selected settings do not establish production performance or held-out quality.

## Historical results: unversioned legacy protocol

These numbers are preserved from the original runs. **nDCG@10 came from requests for 100
results, while latency came from requests for 10.** Candidate depth and hybrid normalization
can change with that limit. These tables therefore cannot establish a paired quality/latency
tradeoff or a performance frontier. Recall@100 also requested only 100 before self exclusion,
so it could retain only 99 results when a self ID was returned. Rerun before making current
comparative claims. Bold marks historical best values within each individual metric only.

| Legacy nDCG@10, request 100 | SciFact | NFCorpus | ArguAna | SciDocs | FiQA |
|---|---|---|---|---|---|
| Text: hinny | 0.662 | 0.307 | 0.458 | 0.149 | 0.238 |
| Text: hinny, English analyzer | **0.688** | **0.328** | **0.491** | **0.157** | **0.256** |
| Text: Tantivy (tantivy-py) | 0.661 | 0.307 | 0.404 | 0.151 | 0.237 |
| Text: SQLite (FTS5, sqlite-vec) | 0.668 | 0.307 | 0.472 | 0.151 | 0.234 |
| Text: LanceDB | 0.684 | 0.327 | 0.490 | **0.157** | 0.253 |
| Learned sparse: hinny, a sparse field | 0.708 | 0.345 | 0.521 | 0.163 | 0.358 |
| Vector: hinny | **0.833** | **0.352** | **0.574** | **0.167** | 0.290 |
| Vector: SQLite (FTS5, sqlite-vec) | **0.833** | 0.351 | 0.572 | **0.167** | **0.292** |
| Vector: LanceDB | **0.833** | **0.352** | **0.574** | **0.167** | **0.292** |
| Vector: Chroma | 0.831 | **0.352** | **0.574** | **0.167** | 0.288 |
| Vector: Qdrant (local mode) | **0.833** | 0.351 | 0.572 | **0.167** | **0.292** |
| Hybrid: hinny | 0.801 | 0.364 | 0.597 | 0.182 | 0.339 |
| Hybrid: hinny, English analyzer | 0.803 | 0.375 | 0.604 | 0.183 | 0.341 |
| Hybrid: hinny, sparse field + vectors | **0.819** | 0.372 | **0.614** | 0.187 | **0.371** |
| Hybrid: hinny, English + sparse + vectors | 0.814 | **0.380** | 0.608 | **0.188** | 0.367 |
| Hybrid: SQLite (FTS5, sqlite-vec) | 0.755 | 0.363 | 0.573 | 0.174 | 0.324 |
| Hybrid: LanceDB | 0.780 | 0.371 | 0.579 | 0.178 | 0.327 |

| Legacy p50, request 10 | SciFact | NFCorpus | ArguAna | SciDocs | FiQA |
|---|---|---|---|---|---|
| Text: hinny | **0.04 ms** | **0.01 ms** | 0.36 ms | 0.11 ms | 0.28 ms |
| Text: hinny, English analyzer | **0.04 ms** | **0.01 ms** | **0.18 ms** | **0.08 ms** | **0.13 ms** |
| Text: Tantivy (tantivy-py) | 0.07 ms | **0.01 ms** | 0.53 ms | 0.18 ms | 0.44 ms |
| Text: SQLite (FTS5, sqlite-vec) | 3.67 ms | 0.16 ms | 222 ms | 15 ms | 40 ms |
| Text: LanceDB | 0.97 ms | 1.62 ms | 1.47 ms | 0.88 ms | 0.98 ms |
| Learned sparse: hinny, a sparse field | 0.05 ms | 0.02 ms | 0.23 ms | 0.12 ms | 0.25 ms |
| Vector: hinny | **0.03 ms** | **0.05 ms** | **0.03 ms** | **0.07 ms** | **0.10 ms** |
| Vector: SQLite (FTS5, sqlite-vec) | 1.16 ms | 0.79 ms | 1.83 ms | 5.50 ms | 13 ms |
| Vector: LanceDB | 2.46 ms | 2.28 ms | 1.71 ms | 2.84 ms | 4.64 ms |
| Vector: Chroma | 0.31 ms | 0.31 ms | 0.36 ms | 0.37 ms | 0.61 ms |
| Vector: Qdrant (local mode) | 0.82 ms | 0.59 ms | 1.38 ms | 4.04 ms | 9.31 ms |
| Hybrid: hinny | **0.09 ms** | **0.09 ms** | 0.41 ms | 0.18 ms | 0.38 ms |
| Hybrid: hinny, English analyzer | **0.09 ms** | 0.10 ms | **0.22 ms** | **0.17 ms** | **0.26 ms** |
| Hybrid: hinny, sparse field + vectors | 0.10 ms | 0.10 ms | 0.27 ms | 0.18 ms | 0.31 ms |
| Hybrid: hinny, English + sparse + vectors | 0.10 ms | 0.10 ms | 0.46 ms | 0.21 ms | 0.35 ms |
| Hybrid: SQLite (FTS5, sqlite-vec) | 4.86 ms | 1.02 ms | 226 ms | 21 ms | 54 ms |
| Hybrid: LanceDB | 3.66 ms | 1.88 ms | 2.41 ms | 3.31 ms | 5.10 ms |

| Legacy Recall@100, request 100 | SciFact | NFCorpus | ArguAna | SciDocs | FiQA |
|---|---|---|---|---|---|
| Text: hinny | 0.886 | 0.236 | 0.945 | 0.348 | 0.510 |
| Text: hinny, English analyzer | **0.921** | **0.247** | 0.964 | 0.363 | 0.552 |
| Text: Tantivy (tantivy-py) | 0.889 | 0.237 | 0.907 | 0.348 | 0.510 |
| Text: SQLite (FTS5, sqlite-vec) | 0.886 | 0.233 | 0.958 | 0.349 | 0.517 |
| Text: LanceDB | **0.921** | **0.247** | **0.967** | **0.365** | **0.554** |
| Learned sparse: hinny, a sparse field | 0.938 | 0.278 | 0.982 | 0.374 | 0.640 |
| Vector: hinny | **0.967** | 0.352 | **0.979** | 0.384 | 0.620 |
| Vector: SQLite (FTS5, sqlite-vec) | **0.967** | 0.352 | **0.979** | **0.385** | **0.623** |
| Vector: LanceDB | **0.967** | 0.352 | **0.979** | **0.385** | **0.623** |
| Vector: Chroma | 0.963 | **0.353** | **0.979** | 0.381 | 0.620 |
| Vector: Qdrant (local mode) | **0.967** | 0.352 | **0.979** | **0.385** | **0.623** |
| Hybrid: hinny | 0.980 | 0.355 | 0.982 | 0.416 | 0.650 |
| Hybrid: hinny, English analyzer | 0.983 | 0.357 | 0.984 | 0.414 | 0.662 |
| Hybrid: hinny, sparse field + vectors | 0.973 | 0.349 | 0.986 | 0.415 | **0.690** |
| Hybrid: hinny, English + sparse + vectors | 0.980 | **0.358** | 0.986 | 0.418 | 0.685 |
| Hybrid: SQLite (FTS5, sqlite-vec) | 0.980 | 0.354 | **0.989** | **0.419** | 0.655 |
| Hybrid: LanceDB | **0.987** | 0.353 | **0.989** | 0.411 | 0.651 |

| Build, size | SciFact | NFCorpus | ArguAna | SciDocs | FiQA |
|---|---|---|---|---|---|
| hinny | 0.2 s, 8 MB | 0.1 s, 6 MB | 0.1 s, 12 MB | 0.4 s, 36 MB | 0.7 s, 67 MB |
| hinny, English analyzer | 0.2 s, 8 MB | 0.1 s, 6 MB | 0.2 s, 11 MB | 0.7 s, 34 MB | 1.1 s, 64 MB |
| hinny, English + sparse field | 0.4 s, 11 MB | 0.2 s, 8 MB | 0.5 s, 16 MB | 1.5 s, 47 MB | 3.0 s, 90 MB |
| Tantivy (tantivy-py) | 0.5 s, 3 MB | 0.5 s, 2 MB | 0.5 s, 4 MB | 0.7 s, 11 MB | 0.7 s, 18 MB |
| SQLite (FTS5, sqlite-vec) | 0.1 s, 23 MB | 0.1 s, 17 MB | 0.2 s, 29 MB | 0.7 s, 83 MB | 1.1 s, 141 MB |
| LanceDB | 0.9 s, 12 MB | 0.2 s, 9 MB | 0.3 s, 16 MB | 1.1 s, 50 MB | 0.6 s, 93 MB |
| Chroma | 0.5 s, 14 MB | 0.2 s, 10 MB | 0.5 s, 19 MB | 1.8 s, 41 MB | 4.1 s, 84 MB |
| Qdrant (local mode) | 2.6 s, 21 MB | 1.3 s, 15 MB | 2.7 s, 36 MB | 8.8 s, 106 MB | 16.9 s, 239 MB |

## Interpretation limits

The historical tables motivated the English analyzer, learned sparse fields, and hybrid fusion.
They do not validate the same quality at the reported latency; the `beir-v2` tables do, for each
configuration. Hinny's hybrid candidate budgets were chosen on dev and train queries
(`candidates.py`), but its min–max-scaled convex combination at weight 0.5 was selected using
these test sets and the mailbox benchmark (see `../../DESIGN.md`), so the hybrid rows are still
development results, not held-out evidence of a general advantage.

## Historical method

- **Vectors:** EmbeddingGemma 2, the first 256 dimensions (Matryoshka), normalized. Documents
  as `title: {title} | text: {text}`, queries with its search prompt, at most 512 tokens.
  Embedded once (`embed_beir.py`); every system gets the same arrays.
- **Text:** title and text as one field. Engines with a query syntax (Tantivy, FTS5, LanceDB)
  get the query's words, so quotes and operators in a query don't change the search.
- **Quality:** nDCG@10 (linear gains, as `pytrec_eval`) and Recall@100 from requests for 100
  results on the test split, ignoring a result identical to the query, as BEIR does.
- **Speed:** sequential queries from Python, warm, request 10, p50 over up to 300 queries. Build
  is from documents in memory to ready to query; size is the directory on disk.
- **Learned sparse:** OpenSearch's inference-free `opensearch-neural-sparse-encoding-doc-v3-distill`
  (Apache-2.0): documents through the model (title and text, at most 512 tokens, about 240
  weighted terms each), queries without a model, as its tokenizer's tokens with its static
  weights, so a query costs no inference. Encoded once (`embed_sparse.py`, sentence-transformers
  5.2 on MPS: 33 minutes for all five, most of it FiQA), into a hinny sparse field; the
  English analyzer's text field and the vectors are the same as the other rows'.
- **Hybrid:** each system's own: hinny's default, LanceDB's `RRFReranker`, and reciprocal rank
  fusion written by hand for SQLite (depth 400 for quality, 40 for latency).
- **Configurations:** defaults throughout. Chroma: HNSW, cosine. Qdrant: the client's local
  mode, a Python brute-force search, not Qdrant's server. LanceDB: no vector index (an exact
  scan, LanceDB's advice below 100,000 rows), its native full-text index. SQLite: FTS5 and a
  sqlite-vec `vec0` table (exact).
- **Versions:** hinny 0.3.0 (the other systems' rows on the first four datasets are from hinny
  0.1.0's run; SciDocs was added with 0.3.0, all systems run then), lancedb 0.40.0 (pyarrow 26.0.0), chromadb 1.5.9, qdrant-client
  1.20.0, sqlite-vec 0.1.9 on SQLite 3.53.1, tantivy-py 0.27.0, numpy 2.5.4, Python 3.12.
- **Machine:** Apple M5 Pro, 48 GB, macOS 27.2. Another app used about half a core during the
  runs. LanceDB's latencies varied about 2× between two runs; these are from the second.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.in ../../bindings/python
```

Download the datasets from `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/NAME.zip`
into `data/NAME/`, then embed them with a Python that has sentence-transformers and access to
`google/embeddinggemma-2` (on a Mac with Apple silicon: about 50 minutes for all five), and run:

```bash
python -I embed_beir.py data emb scifact nfcorpus arguana scidocs fiqa
```

For the sparse rows, encode them too (another 33 minutes, with the same kind of Python):

```bash
python -I embed_sparse.py data emb scifact nfcorpus arguana scidocs fiqa
```

```bash
.venv/bin/python -I run_bench.py data emb out scifact nfcorpus arguana scidocs fiqa
```

`python3 tables.py out/results.jsonl` renders only `beir-v2` rows, validates their query profiles,
and reports excluded rows from other protocols. It fails if no current rows exist.
`python3 tables.py --legacy results.jsonl` explicitly renders the historical tables.

`results.jsonl` preserves every historical row. New runs append metadata without rewriting old
rows. The website data generator preserves protocol metadata and rejects mixed input unless
`BEIR_PROTOCOL=beir-v2` or `BEIR_PROTOCOL=legacy-unversioned` selects one protocol explicitly.

The fast protocol regression tests need only Python; the website generator check also uses Node:

```bash
python3 -m unittest discover -s benchmarks/beir -p 'test_*.py'
```
