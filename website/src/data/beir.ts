import rows from './beir.json'

export type Row = (typeof rows)[number]
export const results: Row[] = rows

export const datasets = [
  { id: 'scifact', name: 'SciFact', docs: 5183, queries: 300 },
  { id: 'nfcorpus', name: 'NFCorpus', docs: 3633, queries: 323 },
  { id: 'arguana', name: 'ArguAna', docs: 8674, queries: 1406 },
  { id: 'scidocs', name: 'SciDocs', docs: 25657, queries: 1000 },
  { id: 'fiqa', name: 'FiQA', docs: 57638, queries: 648 },
] as const
export type DatasetId = (typeof datasets)[number]['id']

export type Kind = 'hybrid' | 'vector' | 'text'
export const kinds: { id: Kind; name: string }[] = [
  { id: 'hybrid', name: 'Hybrid' },
  { id: 'vector', name: 'Vector' },
  { id: 'text', name: 'Full text' },
]

/** The engine's ids in the benchmark rows. benchmarks/beir/results.jsonl was written under the
 * project's earlier name and is read as it is; only the labels shown on the site say Hinny. */
export const ENGINE = {
  plain: 'hinny',
  english: 'hinny (english)',
  sparse: 'hinny (sparse)',
} as const

/** Each configuration the site shows: which kind of search it is, and what to call it. */
export const configurations: { system: string; mode: string; kind: Kind; name: string; ours: boolean }[] = [
  { system: ENGINE.sparse, mode: 'sparse+vector', kind: 'hybrid', name: 'Hinny, sparse + vectors', ours: true },
  { system: ENGINE.sparse, mode: 'all three', kind: 'hybrid', name: 'Hinny, BM25 + sparse + vectors', ours: true },
  { system: ENGINE.english, mode: 'hybrid', kind: 'hybrid', name: 'Hinny, BM25 + vectors', ours: true },
  { system: 'lancedb', mode: 'hybrid', kind: 'hybrid', name: 'LanceDB', ours: false },
  { system: 'sqlite fts5 + sqlite-vec', mode: 'hybrid', kind: 'hybrid', name: 'SQLite', ours: false },
  { system: ENGINE.plain, mode: 'vector', kind: 'vector', name: 'Hinny', ours: true },
  { system: 'chroma', mode: 'vector', kind: 'vector', name: 'Chroma (HNSW)', ours: false },
  { system: 'lancedb', mode: 'vector', kind: 'vector', name: 'LanceDB (exact)', ours: false },
  { system: 'qdrant (local mode)', mode: 'vector', kind: 'vector', name: 'Qdrant (local mode)', ours: false },
  { system: 'sqlite fts5 + sqlite-vec', mode: 'vector', kind: 'vector', name: 'SQLite (sqlite-vec)', ours: false },
  { system: ENGINE.english, mode: 'text', kind: 'text', name: 'Hinny, English analyzer', ours: true },
  { system: ENGINE.sparse, mode: 'sparse', kind: 'text', name: 'Hinny, sparse field', ours: true },
  { system: 'lancedb', mode: 'text', kind: 'text', name: 'LanceDB', ours: false },
  { system: 'tantivy-py', mode: 'text', kind: 'text', name: 'Tantivy', ours: false },
  { system: 'sqlite fts5 + sqlite-vec', mode: 'text', kind: 'text', name: 'SQLite (FTS5)', ours: false },
]

export type Point = { name: string; ours: boolean; ndcg: number; recall: number; p50: number; p99: number }

/** The configurations of one kind on one dataset. */
export function points(kind: Kind, dataset: DatasetId): Point[] {
  return configurations
    .filter((c) => c.kind === kind)
    .flatMap((c) => {
      const r = results.find((r) => r.dataset === dataset && r.system === c.system && r.mode === c.mode)
      return r ? [{ name: c.name, ours: c.ours, ndcg: r.ndcg10, recall: r.recall100, p50: r.p50_ms, p99: r.p99_ms }] : []
    })
}

/** The three systems that fuse BM25 with vectors, in the order the site lists them. */
export const hybrids = [
  { system: ENGINE.english, name: 'Hinny', ours: true },
  { system: 'lancedb', name: 'LanceDB', ours: false },
  { system: 'sqlite fts5 + sqlite-vec', name: 'SQLite', ours: false },
] as const

const hybridRow = (dataset: string, system: string) => results.find((r) => r.dataset === dataset && r.system === system && r.mode === 'hybrid')

/** How many times longer than Hinny the other two took for the median hybrid query, over the
 * datasets: the range against LanceDB, and the largest against SQLite to one significant figure.
 * Computed from the rows, so the sentence that quotes them can't drift from the chart beside it. */
export const hybridSpeed = (() => {
  const ratios = (system: string) =>
    datasets.flatMap((d) => {
      const ours = hybridRow(d.id, ENGINE.english)
      const theirs = hybridRow(d.id, system)
      return ours && theirs ? [theirs.p50_ms / ours.p50_ms] : []
    })
  const lance = ratios('lancedb')
  const sqlite = Math.max(...ratios('sqlite fts5 + sqlite-vec'))
  const magnitude = 10 ** Math.floor(Math.log10(sqlite))
  return {
    lanceLow: Math.round(Math.min(...lance)),
    lanceHigh: Math.round(Math.max(...lance)),
    sqliteMax: (Math.round(sqlite / magnitude) * magnitude).toLocaleString('en-US'),
  }
})()

/** Hinny's median hybrid query (English analyzer and vectors) over the datasets: the fastest, the
 * slowest, and FiQA's, the largest. Computed from the rows, like the ratios above. */
export const hybridTimes = (() => {
  const times = datasets.flatMap((d) => hybridRow(d.id, ENGINE.english)?.p50_ms ?? [])
  return { low: Math.min(...times), high: Math.max(...times), fiqa: hybridRow('fiqa', ENGINE.english)!.p50_ms }
})()

/** Hybrid nDCG@10 per dataset for those three, and Hinny's lead over the better of the other two. */
export const hybridQuality = datasets.map((d) => {
  const scores = hybrids.map((h) => hybridRow(d.id, h.system)?.ndcg10)
  const [ours, ...others] = scores
  const best = Math.max(...others.map((v) => v ?? 0))
  return { name: d.name, scores, lead: ours === undefined ? undefined : ours - best }
})

export function ms(v: number): string {
  return v >= 10 ? `${v.toFixed(0)} ms` : v >= 1 ? `${v.toFixed(1)} ms` : `${v.toFixed(2)} ms`
}
