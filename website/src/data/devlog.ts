import history from './history.json'

/** One measured version of benchmarks/beir/results.jsonl: FiQA's median times and nDCG@10 for
 * Hinny's main configurations, at the commit that changed them (generated from git history). */
export type Point = (typeof history)[number]
export const points: Point[] = history

/** The configurations the chart follows, all on BEIR FiQA (57,638 documents). */
export const series = [
  { key: 'hybrid', name: 'Hybrid, English analyzer' },
  { key: 'vector', name: 'Vector' },
  { key: 'text', name: 'Full text, English analyzer' },
  { key: 'textPlain', name: 'Full text, words as written' },
] as const
export type SeriesKey = (typeof series)[number]['key']

export type Entry = {
  time: string
  commit: string
  /** Speed, quality and method changes get a numbered marker on the chart. */
  kind: 'speed' | 'quality' | 'method' | 'reliability'
  title: string
  text: string
}

/** What changed, newest first, with the numbers each commit measured. */
export const entries: Entry[] = [
  {
    time: '2026-10-25T21:43:56-07:00',
    commit: '03bb9f0',
    kind: 'quality',
    title: 'Deeper searches probe deeper',
    text: 'Each segment calibrates its probe budget for the 10 nearest of all its vectors. Measured on real queries, choosing on BEIR’s dev and train queries and checking once on the test queries, a top 1000 or a filtered top k reached into clusters it never probed: Quora, k=1000 with a 10% filter against the query’s direction, recall 0.970. The budget now grows with how deep a search must look, (k / 10s)^0.1 for a share s of vectors passing, and leaves the calibrated case alone. Held out, no cell got worse and the worst rose to 0.997; k=1000 unfiltered 0.983–0.994 → 0.998–0.999, for 13% more time at the median cell. A hybrid’s vector side asks for 100, so it probes a little deeper too: FiQA vector recall@100 0.620 → 0.623.',
  },
  {
    time: '2026-10-25T21:05:12-07:00',
    commit: 'e697efe',
    kind: 'speed',
    title: 'An exact top k of items',
    text: 'A text search kept four times k passages and then each item’s best, so one item’s strong passages could hide the next (asked for 2, it returned 1 of 2 matches), and all-words queries past 63 words let wrong documents through in memory. It now keeps the best k items, each at its best passage, and prunes against the k-th item: a higher bar, sooner. Same results on BEIR, and the hybrids 20–35% faster: FiQA English 0.378 → 0.294 ms, words as written 0.510 → 0.385, English full text 0.133 → 0.102. Attribute-only queries keep only their limit (1M documents by date 31.8 → 3.2 ms, 38 MB → 0.1 MB), and exact filtered vector searches stopped building scan caches they don’t use.',
  },
  {
    time: '2026-10-25T20:21:37-07:00',
    commit: '8b2d497',
    kind: 'reliability',
    title: 'Memory and work budgets',
    text: 'Batches are checked against the widths the log writes them in and a size limit before anything is written; memory flushes by bytes as well as passages; stats show what snapshots pin; queries take a deadline or a cancel handle and fail rather than return part of their results.',
  },
  {
    time: '2026-10-24T22:32:51-07:00',
    commit: 'cc059d7',
    kind: 'speed',
    title: 'Term at a time only where pruning can’t skip',
    text: 'Scoring term at a time fired whenever a query’s posting lists added up to the segment, so short queries with stopwords chose it and lost: their long lists have tiny bounds that MaxScore skips. Up to 32 results it now counts only the lists MaxScore can’t skip, against a share of the segment that grows with its size; a deeper top k keeps the old rule. Full text with words as written: FiQA 0.280 → 0.268 ms, SciDocs 0.116 → 0.109, Quora at 11 results 0.65 → 0.47 ms. Most of FiQA’s earlier 0.22 ms is back; the rest would cost Quora’s tail. (Correction, 25 Oct: from Python the new rule didn’t apply yet. The search asked its collector for four times 11 passages, past 32, so the Python numbers here moved by noise; the Quora figure is the Rust harness’s. The rule has applied since e697efe.)',
  },
  {
    time: '2026-10-24T21:53:35-07:00',
    commit: '2795c20',
    kind: 'method',
    title: 'A sparse budget, and explain()',
    text: 'A sparse field searched without text gets a candidate budget of its own, chosen on dev queries: at 200 against 100, SciFact nDCG@10 0.939 against 0.928, FiQA 0.391 against 0.392 for 0.47 against 0.38 ms. Query::explain shows what fusion saw: each retriever’s depth, how many it found and their score range, and how many both found.',
  },
  {
    time: '2026-10-24T20:22:07-07:00',
    commit: '750e5b0',
    kind: 'reliability',
    title: 'Killing the writer, 40 times',
    text: 'A child process writes batches while flushes and merges run, and is killed with SIGKILL at a random moment. A process crash keeps what reached the OS, so every acknowledged batch must survive, whole and in order. With the log flush on each write removed, the test fails at once.',
  },
  {
    time: '2026-10-24T19:39:21-07:00',
    commit: '5d7903c',
    kind: 'method',
    title: 'BEIR rerun: quality and speed from the same queries',
    text: 'The earlier runs measured nDCG@10 with requests for 100 results and speed with requests for 10, which fused at different depths, so FiQA’s 0.341 at 0.26 ms described two settings. Measured together: 0.334 at 0.38 ms. Hinny’s hybrid still ranks first on all five datasets, in 9 to 16 times less time than LanceDB’s.',
  },
  {
    time: '2026-10-24T19:14:44-07:00',
    commit: '66a66d9',
    kind: 'reliability',
    title: 'Every I/O failure, one at a time',
    text: 'Each place the engine writes or syncs can be made to fail. A sweep fails each one at every point a workload reaches it (296 cases, 344 once torn log appends were added), crashes and reopens. Reintroducing the deletion bug fixed an hour earlier makes it fail.',
  },
  {
    time: '2026-10-24T19:04:30-07:00',
    commit: '6558e5f',
    kind: 'reliability',
    title: 'Checksummed metadata (format 4)',
    text: 'A manifest damaged in a way that still parsed used to be believed, and could open an index empty. The manifest and deletion files now carry CRC32s and are refused when they fail; indexes from 0.3 open unchanged.',
  },
  {
    time: '2026-10-24T18:52:12-07:00',
    commit: 'f9cc7f1',
    kind: 'quality',
    title: 'Candidate budgets, chosen on dev splits',
    text: 'Hybrid retrievers looked four times the limit deep, so asking for 10 or 11 results changed the ranking. Now fixed budgets of 200 text and 100 vector candidates, chosen on dev and train queries, never test ones: FiQA nDCG@10 0.358 → 0.370 for 0.28 → 0.39 ms, SciFact 0.908 → 0.927, Quora 0.877 → 0.884.',
  },
  {
    time: '2026-10-24T18:42:02-07:00',
    commit: '4c224d5',
    kind: 'reliability',
    title: 'Recovery fails safe, and sync()',
    text: 'An audit found that an unreadable manifest opened as an empty index and deleted its segments, and that a failed deletion save could be retried as if saved. Both fixed; writes gain an explicit sync(), and the benchmark protocol now measures quality and speed at one setting.',
  },
  {
    time: '2026-10-24T16:50:33-07:00',
    commit: '77b3774',
    kind: 'speed',
    title: 'Calibration stops where recall stops',
    text: 'At a recall target of 0.997, a million synthetic vectors never reached it and probed every cluster: 2.6 ms. Calibration now also stops where more probes stop helping: 0.13 ms. The higher target is the one deliberate slowdown on this page: SciDocs vector search 0.056 → 0.073 ms, for nDCG@10 0.166 → 0.167.',
  },
  {
    time: '2026-10-24T16:47:03-07:00',
    commit: '37c7ed1',
    kind: 'speed',
    title: 'Vector search 1.6× faster: FastScan',
    text: 'Codes regrouped in blocks of 32 and scored with 4-bit lookup tables, as Faiss’s FastScan does: FiQA vector p50 0.150 → 0.094 ms at the same recall, hybrid 0.30 → 0.26 ms.',
  },
  {
    time: '2026-10-24T16:34:07-07:00',
    commit: '39c4c1a',
    kind: 'quality',
    title: 'Snowball’s stopwords',
    text: 'The English analyzer drops Snowball’s 127-word stopword list instead of the earlier 33-word list: FiQA full text nDCG@10 0.250 → 0.256 and p50 0.20 → 0.13 ms. Full text now matches or beats LanceDB’s on all five datasets.',
  },
  {
    time: '2026-10-24T04:43:59-07:00',
    commit: '6acaf09',
    kind: 'speed',
    title: 'Term at a time where pruning can’t win',
    text: 'When a query’s posting lists are about as long as the segment, block-max pruning skips nothing; such queries are scored term at a time instead: ArguAna full text 0.40 → 0.21 ms, FiQA hybrid p99 0.68 → 0.42 ms.',
  },
  {
    time: '2026-10-24T04:33:23-07:00',
    commit: '9606dad',
    kind: 'quality',
    title: 'Fusing vectors deeper',
    text: 'Vectors fused at least 100 deep, text 40: the mailbox benchmark’s right-email-first rate 45.4% → 50.4%, FiQA hybrid at 0.32 ms.',
  },
  {
    time: '2026-10-24T03:44:00-07:00',
    commit: 'a514312',
    kind: 'speed',
    title: 'Full-precision vectors in half precision',
    text: 'They’re only read to rerank, so f16 costs no recall (FiQA 0.993 either way) and halves a segment: FiQA 64 → 34 MB.',
  },
  {
    time: '2026-10-24T03:40:17-07:00',
    commit: '059a1fc',
    kind: 'speed',
    title: 'Segments searched at once',
    text: 'A query’s segments are searched in parallel on the query threads: FiQA in five segments, vector p50 0.31 → 0.18 ms.',
  },
  {
    time: '2026-10-24T03:35:33-07:00',
    commit: '8fc96e8',
    kind: 'speed',
    title: 'Both retrievers at once',
    text: 'A hybrid query’s text and vector searches run on threads of their own: FiQA hybrid p50 0.53 → 0.33 ms (1.0 ms before that night’s vector work).',
  },
  {
    time: '2026-10-24T03:29:45-07:00',
    commit: '23e5fcf',
    kind: 'speed',
    title: 'Indexing 2–3× faster',
    text: 'k-means as matrix products, seeding from a sample, cached stems and Fx hashing: FiQA’s upserts 1.7 → 1.1 s.',
  },
  {
    time: '2026-10-24T03:17:39-07:00',
    commit: 'ad33e6d',
    kind: 'speed',
    title: 'Vector search 3× faster',
    text: 'Requantizing the query for each probed cluster had been 70% of the time; now it’s done once, without allocations or branches, with NEON bit counts: FiQA vector p50 0.53 → 0.16 ms at recall 0.993, exact search 4.4 → 1.0 ms.',
  },
  {
    time: '2026-10-24T02:55:33-07:00',
    commit: '7df3bc0',
    kind: 'quality',
    title: 'Query term frequency in BM25',
    text: 'Repeated query words are now weighted by how often they appear (query term frequency): ArguAna nDCG@10 0.425 → 0.482 with the English analyzer.',
  },
  {
    time: '2026-10-24T00:45:16-07:00',
    commit: '24aa570',
    kind: 'method',
    title: 'First comparison with other embedded databases',
    text: 'Four BEIR datasets against LanceDB, Chroma, Qdrant, SQLite and Tantivy, with the same vectors for all. Convex fusion beat reciprocal rank fusion everywhere (the mailbox’s right-email-first rate 38.7% → 48.7%), and calibrated probing lifted vector recall@10 from 0.913 on FiQA with a fixed budget to 0.990–0.999 on real queries.',
  },
  {
    time: '2026-10-23T21:10:50-07:00',
    commit: '2a21677',
    kind: 'method',
    title: 'The first commit',
    text: 'A library-first hybrid search database: BM25 and 1-bit RaBitQ vectors over per-segment clusters in one index, one query builder for text, vectors or both (then fused by reciprocal rank), and filters checked during the search.',
  },
]
