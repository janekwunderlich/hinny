import * as stylex from '@stylexjs/stylex'
import { announceOnly } from '#/announce'
import { createFileRoute } from '@tanstack/react-router'
import { pageMeta } from '#/seo'
import { useState } from 'react'
import { BarChart } from '#/components/BarChart'
import { BeirTable } from '#/components/BeirTable'
import { Chart } from '#/components/Chart'
import { InlineCode } from '#/components/CodeBlock'
import { LatencyStrip } from '#/components/LatencyStrip'
import { ExternalLink } from '#/components/Link'
import { PageHead, Section } from '#/components/Page'
import { Item, List, Note, Strong } from '#/components/Prose'
import { Segmented } from '#/components/Segmented'
import { DataTable } from '#/components/Table'
import { focus, link, text } from '#/components/styles'
import { ENGINE, configurations, datasets, hybridQuality, hybridSpeed, hybridTimes, hybrids, results, type DatasetId } from '#/data/beir'
import { GITHUB_URL, NAME } from '#/data/site'
import { colors, media, sizes, space, type } from '../tokens.stylex'

export const Route = createFileRoute('/benchmarks')({
  beforeLoad: announceOnly,
  head: () => ({ meta: pageMeta(`Benchmarks · ${NAME}`, `${NAME} against LanceDB, SQLite, Chroma, Qdrant and Tantivy on five BEIR datasets: nDCG@10, Recall@100, latency, build time and size.`, '/benchmarks') }),
  component: Benchmarks,
})

const builds = [
  { system: ENGINE.english, name: 'Hinny, English analyzer', ours: true },
  { system: ENGINE.sparse, name: 'Hinny, English + sparse field', ours: true },
  { system: 'tantivy-py', name: 'Tantivy (text only)', ours: false },
  { system: 'sqlite fts5 + sqlite-vec', name: 'SQLite', ours: false },
  { system: 'lancedb', name: 'LanceDB', ours: false },
  { system: 'chroma', name: 'Chroma (vectors only)', ours: false },
  { system: 'qdrant (local mode)', name: 'Qdrant (vectors only)', ours: false },
]

const styles = stylex.create({
  facts: { display: 'grid', gridTemplateColumns: { default: 'repeat(2, minmax(0, 1fr))', [media.phone]: 'minmax(0, 1fr)' }, columnGap: space.x8, rowGap: space.x5, paddingTop: space.x5, borderTopWidth: sizes.hair, borderTopStyle: 'solid', borderTopColor: colors.line },
  fact: { display: 'grid', gap: space.x1, alignContent: 'start' },
  figure: { fontSize: type.h3, lineHeight: 1.2, fontWeight: 500, letterSpacing: '-0.02em', color: colors.fg, fontVariantNumeric: 'tabular-nums' },
})

/** Hybrid nDCG@10 of every hybrid configuration on one dataset, as bars from zero; the tabs
 * choose the dataset. The scale ends at the next round value above the best score. */
function RankingBars() {
  const [dataset, setDataset] = useState<DatasetId>('fiqa')
  const current = datasets.find((d) => d.id === dataset)!
  const bars = configurations
    .filter((c) => c.kind === 'hybrid')
    .flatMap((c) => {
      const r = results.find((r) => r.dataset === dataset && r.system === c.system && r.mode === c.mode)
      const [name, note] = c.name.split(', ')
      return r ? [{ name: name!, note: note ?? (name === 'SQLite' ? 'FTS5 + sqlite-vec' : undefined), value: r.ndcg10, label: r.ndcg10.toFixed(3), ours: c.ours }] : []
    })
  const best = Math.max(...bars.map((b) => b.value))
  const step = [0.05, 0.1, 0.2, 0.25].find((s) => best / s <= 5) ?? 0.25
  const max = Math.ceil(best / step - 1e-9) * step
  const ticks = Array.from({ length: Math.round(max / step) + 1 }, (_, i) => +(i * step).toFixed(2))
  return (
    <BarChart
      title={`Hybrid ranking on ${current.name}`}
      subtitle={`nDCG@10, ${current.docs.toLocaleString('en-US')} documents, ${current.queries.toLocaleString('en-US')} queries`}
      bars={bars}
      max={max}
      ticks={ticks}
      axis="nDCG@10, higher is better"
      format={(t) => t.toFixed(2)}
      controls={<Segmented label="Dataset" options={datasets} value={dataset} onChange={setDataset} />}
    />
  )
}

/** A time as the tables show it: two decimals under 1 ms. */
const time = (v: number) => `${v.toFixed(2)}`

function Benchmarks() {
  return (
    <>
      <PageHead label="Benchmarks" title="Five BEIR datasets, six embedded databases">
        Same documents, same 256-dim EmbeddingGemma 2 vectors, same queries, scored by the same harness with quality and latency timed on the exact same call. {NAME}'s hybrid search ranks first on all five datasets in {time(hybridTimes.low)}–{time(hybridTimes.high)} ms.{' '}
        {GITHUB_URL && (
          <ExternalLink href={`${GITHUB_URL}/tree/main/benchmarks/beir`} {...stylex.props(link.inline, focus.ring)}>
            Benchmark scripts on GitHub
          </ExternalLink>
        )}
      </PageHead>

      <Section
        id="speed"
        title="Hybrid query latency"
        lead={
          <>
            Median hybrid query latency across all five datasets (top 10): {NAME} is {hybridSpeed.lanceLow}–{hybridSpeed.lanceHigh}× faster than LanceDB and up to ~{hybridSpeed.sqliteMax}× faster than SQLite.
          </>
        }
      >
        <LatencyStrip />
        <dl {...stylex.props(styles.facts)}>
          <div {...stylex.props(styles.fact)}>
            <dt {...stylex.props(styles.figure)}>0.10 ms</dt>
            <dd {...stylex.props(text.small)}>median vector search on FiQA at 0.99 recall@10</dd>
          </div>
          <div {...stylex.props(styles.fact)}>
            <dt {...stylex.props(styles.figure)}>150–180k/s</dt>
            <dd {...stylex.props(text.small)}>documents ingested from Python, searchable immediately (OS-buffered WAL)</dd>
          </div>
        </dl>
      </Section>

      <Section id="rank" title="Hybrid ranking" lead={<>Hybrid nDCG@10 using identical documents, vectors, and queries: {NAME} ranks first on all five datasets.</>}>
        <RankingBars />
        <DataTable
          label="Hybrid nDCG@10 on five BEIR datasets"
          tintColumn={1}
          head={['Dataset', ...hybrids.map((h) => h.name), 'Lead']}
          rows={hybridQuality.map((q) => [q.name, ...q.scores.map((v, i) => ({ value: v?.toFixed(3) ?? '–', strong: i === 0 })), { value: q.lead === undefined ? '–' : `+${q.lead.toFixed(3)}`, strong: true }])}
        />
        <Note>
          Quality and latency come from the same queries at the same settings for every system. Because {NAME}'s default fusion weight was evaluated on these test sets, treat its hybrid rows as development results. See{' '}
          <a href="#method" {...stylex.props(link.inline, focus.ring)}>
            methodology
          </a>{' '}
          below.
        </Note>
      </Section>

      <Section id="chart" title="Quality vs. latency" lead="Every configuration for one search mode on one dataset: higher is better ranking, left is faster.">
        <Chart />
      </Section>

      <Section id="table" title="All results" lead="Every configuration across all five datasets, grouped by search mode and metric.">
        <BeirTable />
      </Section>

      <Section id="mail" title="On a private mailbox" lead="119 questions over 23,000 emails from a private archive (not externally reproducible). A hit is the target email or another message in its thread.">
        <DataTable
          label="Recall on the mailbox benchmark, in percent"
          ours={[2]}
          head={['Retrieval', 'R@1', 'R@3', 'R@5', 'R@10']}
          rows={[
            ['Text (BM25)', '35.3', '47.9', '52.9', '62.2'],
            ['Hybrid, Reciprocal Rank Fusion', '38.7', '63.9', '72.3', '78.2'],
            ["Hybrid, Hinny's default", '48.7', '64.7', '74.8', '78.2'],
          ]}
        />
        <Note>Percentage of questions with a hit in the top 1, 3, 5, and 10 results.</Note>
      </Section>

      <Section id="vec" title="Vector recall" lead="Recall@10 against exact search at each segment's calibrated probe budget. Full vectors are stored in half precision and read only to rerank.">
        <DataTable
          wide
          label="Vector recall and time on three collections"
          head={['Collection', 'Recall@10', 'p50', 'Exact scan', 'Index']}
          rows={[
            ['FiQA, 57,638 docs, 648 questions', '0.992', '0.10 ms', '0.78 ms', '34 MB'],
            ['Private mailbox, 23,037 emails', '0.998', '0.04 ms', '0.31 ms', '14 MB'],
            ['1 M vectors, synthetic', '0.997', '0.13 ms', '13 ms', '596 MB'],
          ]}
        />
      </Section>

      <Section id="build" title="Build time and index size" lead="From documents in memory to ready to query; size is total directory size on disk.">
        <DataTable
          wide
          label="Build time and size on disk for every system and dataset"
          ours={builds.flatMap((b, i) => (b.ours ? [i] : []))}
          head={['System', ...datasets.map((d) => d.name)]}
          rows={builds.map((b) => [
            b.name,
            ...datasets.map((d) => {
              const r = results.find((r) => r.system === b.system && r.dataset === d.id)
              return r ? `${r.build_s.toFixed(1)} s, ${r.size_mb.toFixed(0)} MB` : ''
            }),
          ])}
        />
        <Note>{NAME} stores compressed document text, half-precision vectors, and 1-bit codes. Its build time includes merging into one segment and calibrating probe budgets.</Note>
      </Section>

      <Section id="method" title="Methodology">
        <List>
          <Item>
            <Strong>Vectors:</Strong> EmbeddingGemma 2, first 256 dimensions, L2-normalized. Documents formatted as <InlineCode>title: … | text: …</InlineCode>, queries with its search prompt, capped at 512 tokens.
          </Item>
          <Item>
            <Strong>Learned sparse:</Strong> OpenSearch's inference-free <InlineCode>doc-v3-distill</InlineCode>. Documents are encoded at ingest; queries require no model inference.
          </Item>
          <Item>
            <Strong>Paired quality and latency:</Strong> Each query requests 11 results, drops self-matches (per BEIR convention), and scores the top 10; the exact same request is timed. Recall@100 uses a separate request for 101.
          </Item>
          <Item>
            <Strong>Environment:</Strong> Single-threaded Python queries on an Apple M5 Pro, warm cache, p50 over up to 300 queries. Measured 2026-10-24 at commit <InlineCode>f9cc7f1</InlineCode>.
          </Item>
          <Item>
            <Strong>Configurations:</Strong> Defaults throughout. LanceDB without a vector index (per its guidance below 100,000 rows) plus its native full-text index; Chroma with HNSW; Qdrant local mode (Python brute-force scan); SQLite with FTS5 and sqlite-vec. Hybrid uses each system's built-in fusion.
          </Item>
          <Item>
            <Strong>Caveats:</Strong> {NAME}'s hybrid candidate budgets were chosen on dev/train splits, while its default fusion weight was evaluated on these test sets, so hybrid rows should be read as development results. Earlier runs that measured quality and latency at different request sizes are archived in the repository.
          </Item>
        </List>
      </Section>
    </>
  )
}
