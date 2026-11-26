# hinny for Node.js

**An embedded hybrid search database for Node.js.** Combines full-text search (BM25), dense vectors, learned sparse terms, and typed attribute filters in-process on local files. Text and vectors run together under one filter and fuse into a single ranking. Built as a native Node-API addon with TypeScript definitions and zero runtime dependencies.

```js
const { Database, attr } = require('hinny')

const db = new Database('data')
const docs = db.namespace('docs', {
  text: { title: 2, body: 1 },        // BM25 fields with boosts
  ints: ['year'], strings: ['kind'],  // typed attributes for filters and results
  vector: 384,                        // embeddings (cosine by default)
})

docs.upsert({
  key: 1,
  text: { title: 'Lease agreement', body: 'The apartment on Elm Street' },
  attrs: { year: 2025, kind: 'pdf' },
  vector: embed('Lease agreement …'),  // Float32Array, Float64Array, or number[]
})

const hits = docs.query({
  text: 'apartment lease',
  vector: embed("where's my lease?"),
  filter: attr('year').gte(2025).and(attr('kind').eq('pdf')),
  limit: 10,
})
for (const h of hits) {
  console.log(h.key, h.score, h.attrs, h.snippet)  // also h.text, h.highlights, h.textScore, h.vectorScore
}
```

## For retrieval-augmented generation

Store chunks as documents with source metadata as attributes, and query with the user's question as both text and vector:

```js
const { key } = require('hinny')

const chunks = db.namespace('chunks', { text: 'text', ints: ['page'], strings: ['source'], vector: 768 })
chunks.upsertMany(splitAndEmbed(file).map(({ chunk, page, vec }, i) => (
  { key: key(`${file}#${i}`), text: chunk, attrs: { source: file, page }, vector: vec }
)))

const hits = chunks.query({ text: question, vector: embed(question), filter: attr('source').prefix('handbook/'), limit: 8 })
const context = hits.map((h) => h.text.text).join('\n\n')
```

## API overview

Mirrors the Python binding using JavaScript conventions (options objects, camelCase, `null` for missing values). See [`index.d.ts`](index.d.ts) for full TypeScript signatures.

### Databases, namespaces, and schemas

- `new Database(path)` manages namespaces in a directory. `db.namespace(name, schema)` opens or creates one, where `schema` accepts `{ text, ints, floats, strings, tags, sparse, vector, metric, storeText, language, stopwords }`. Schemas can grow over time.
- `metric` selects `'cosine'` (default), `'euclidean'`, or `'dot'`, fixed once set (`ns.schema.metric` reports it). Higher scores always indicate closer vectors: cosine similarity, negated squared distance (`0` for an exact match), or dot product. Under `'euclidean'` and `'dot'`, vectors are stored unnormalized.
- `language: 'english'` (or any of 18 Snowball languages) stems text fields and drops stopwords by default for English and German (`stopwords: false` keeps them). Analyzers are fixed once the namespace holds data.
- `db.open('docs')` opens an existing namespace with its stored schema; `db.namespaces()` lists namespace names; `db.delete('sandbox')` deletes a closed namespace (call `close()` first or it throws `ERR_HINNY_LOCKED`).
- `ns.fork('sandbox')` creates an independent copy-on-write namespace via hard links in milliseconds, using no extra disk until the two diverge.

### Writes, mutations, and durability

- `ns.upsert({ key, text, attrs, vector, version, sparse })`: immediately searchable and survives a process crash on return; call `ns.sync()` for an OS-crash/power-loss durability boundary.
- `ns.upsertMany(docs)`, `ns.apply({ upsert, delete })`, `ns.delete(key)`, `ns.deleteMany(keys)`, `ns.deleteWhere(filter)`: atomic batches; invalid inputs reject the entire batch before writing.
- `ns.patch(key, { text, attrs, vector, sparse, version, remove }, condition?)` and `ns.patchWhere(filter, patch)`: update specific fields and drop names in `remove`. In namespaces with sparse fields, supply `sparse` or list the field in `remove` (otherwise throws `ERR_HINNY_INVALID`). Patching reads and rewrites the document (requiring stored text).
- Conditional writes: `ns.upsertIf(doc, condition)` and `ns.deleteIf(key, condition)` evaluate `condition` and write atomically under the writer lock, returning whether the write ran.
- `ns.sync()`, `ns.flush()`, `ns.optimize()`, `ns.verify()`, and `ns.setLimits({ maxBatchBytes, flushBytes })` manage durability, compaction, integrity checks, and memory bounds.

### Queries, filters, and aggregations

- `ns.query({ text, vector, sparse, filter, limit, orderBy, typing, textWeight, fusion, textCandidates, vectorCandidates, sparseCandidates, nprobe, timeout, snippet, withText })`: text, vector, sparse, hybrid, or filtered attribute queries (`orderBy: '-year'` for descending).
- `ns.explain(options)` returns `{ hits, fusion }` with per-retriever candidate counts, score ranges, and overlap.
- `typing: 32` enables search-as-you-type prefix expansion on the final word.
- `timeout` (milliseconds) throws `ERR_HINNY_TIMEOUT` instead of returning partial results.
- Filters: `attr('kind').eq('pdf')`, `.ne`, `.gt`, `.gte`, `.lt`, `.lte`, `.between(a, b)`, `.oneOf([...])`, `.prefix('…')`, `.contains('…')` (case-insensitive), `.glob('…')`, `.iglob('…')`, `.exists()`, combined with `.and(f)`, `.or(f)`, `.not()`. Tag attributes (`tags: ['labels']`) add `.has('tax')`, `.hasAny([...])`, and `.hasAll([...])`.
- Aggregations & reads: `ns.count(filter)`, `ns.size`, `ns.countBy('kind', filter, limit)`, `ns.get(key)`, `ns.text(key)`, `ns.versions()`, `ns.schema`, `ns.stats()`, `ns.snapshot()`, and `key('any string')` for stable 64-bit keys.

### Synchronous by default, async for maintenance

Queries and writes are synchronous because sub-millisecond in-process operations are simpler and faster without event-loop round trips. The three maintenance operations that can take longer on large namespaces also have Promise variants running on libuv's thread pool so your event loop stays responsive: `ns.optimizeAsync()`, `ns.flushAsync()`, and `ns.verifyAsync()`. Queries keep running concurrently during all three.

### Keys are `bigint`s

Keys are unsigned 64-bit integers. Pass either a safe `number` (up to `2^53 − 1`) or a `bigint` (up to `2^64 − 1`). Hits, `versions()`, and `snap.seq` always return `bigint`s to avoid silent precision loss; numbers above `2^53 − 1` are rejected. Compare with `h.key === 3n` or convert with `Number(h.key)` when keys are small. Integer attributes return `number` unless they exceed `2^53`.

### Closing handles

A namespace holds an exclusive writer lock until closed via `ns.close()`, `db.close()`, or `using ns = db.namespace(...)`. Closing a namespace closes its snapshots. Open a namespace only once per process at a time. Unclosed handles are cleaned up on GC or normal exit, but always call `ns.sync()` or `ns.flush()` explicitly to handle durability errors.

### Stored text & snippets

Text is stored by default (`storeText: true`). Each hit carries `text`, a ~240-character `snippet`, and `highlights` spans in UTF-16 code units (`h.snippet.slice(a, b)`). Pass `snippet: 0` or `withText: false` to skip them per query.

### Error codes

Errors are standard `Error` instances with a `code` property:

| `code` | Meaning |
| :--- | :--- |
| `ERR_HINNY_INVALID` | Invalid input (unknown field, mismatched vector dimensions, `NaN` float, unknown option, batch over limit) |
| `ERR_HINNY_TIMEOUT` | Query exceeded its `timeout` |
| `ERR_HINNY_LOCKED` | Namespace is already open for writing in another process or unclosed handle |
| `ERR_HINNY_CLOSED` | Namespace, snapshot, or database handle was already closed |
| `ERR_HINNY_IO` | Filesystem read/write failure (message includes the path) |

## Build

The `hinny` package on npm is currently an empty `0.0.1` placeholder. Build from source with **Rust 1.89+** and **Node.js**:

```bash
cd bindings/node
npm ci           # installs @napi-rs/cli (no runtime dependencies)
npm run build    # builds hinny.<platform>.node and native.js
npm test         # runs node --test
```

Use `npm run build:debug` for faster unoptimized builds. Cargo outputs to the workspace `target/` directory (or `CARGO_TARGET_DIR`). Currently tested on Linux arm64 (glibc) with Node.js 26; other platforms are not yet tested.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
