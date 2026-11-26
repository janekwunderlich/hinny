import * as stylex from '@stylexjs/stylex'
import { announceOnly } from '#/announce'
import { createFileRoute } from '@tanstack/react-router'
import { pageMeta } from '#/seo'
import { Chevron } from '#/components/Button'
import { InlineCode } from '#/components/CodeBlock'
import { Link } from '#/components/Link'
import { LogChart, marked } from '#/components/LogChart'
import { PageHead, Section } from '#/components/Page'
import { Note } from '#/components/Prose'
import { DataTable } from '#/components/Table'
import { focus, link, text } from '#/components/styles'
import { ms } from '#/data/beir'
import { entries, points, series } from '#/data/devlog'
import { NAME } from '#/data/site'
import { colors, fonts, sizes, space, type } from '../tokens.stylex'

export const Route = createFileRoute('/log')({
  beforeLoad: announceOnly,
  head: () => ({ meta: pageMeta(`Dev log · ${NAME}`, `How ${NAME}’s benchmarks moved, commit by commit: what changed, and what each change measured.`, '/log') }),
  component: Log,
})

const when = (iso: string) => new Date(iso).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', timeZone: 'America/Los_Angeles' })

/** Each configuration's first and latest FiQA numbers. */
function changes(metric: 'p50' | 'ndcg') {
  return series.map((s) => {
    const measured = points.filter((p) => (p[metric] as Record<string, number>)[s.key] !== undefined)
    const first = measured[0]!
    const last = measured[measured.length - 1]!
    return { name: s.name, from: (first[metric] as Record<string, number>)[s.key]!, since: first.time, to: (last[metric] as Record<string, number>)[s.key]! }
  })
}

const styles = stylex.create({
  entries: { display: 'grid' },
  entry: {
    display: 'grid',
    gap: space.x2,
    paddingBlock: space.x5,
    borderBottomWidth: sizes.hair,
    borderBottomStyle: 'solid',
    borderBottomColor: colors.line,
    scrollMarginTop: '88px',
    // The entry a chart marker or a link points at.
    backgroundColor: { default: 'transparent', ':target': colors.surface },
    paddingInline: { default: 0, ':target': space.x4 },
    marginInline: { default: 0, ':target': `calc(-1 * ${space.x4})` },
  },
  entryFirst: { paddingTop: 0 },
  meta: { display: 'flex', flexWrap: 'wrap', alignItems: 'center', columnGap: space.x4, rowGap: space.x1 },
  commit: { fontFamily: fonts.mono, fontSize: type.micro, color: colors.muted },
  number: { display: 'inline-grid', placeItems: 'center', minWidth: '20px', height: '20px', borderWidth: sizes.hair, borderStyle: 'solid', borderColor: colors.fg, borderRadius: '50%', color: colors.fg, fontSize: '11px', lineHeight: 1, fontVariantNumeric: 'tabular-nums' },
  body: { fontSize: '15px', lineHeight: 1.7 },
  more: { display: 'flex' },
})

function Log() {
  const number = new Map(marked.map((m) => [m.commit, m.n]))
  return (
    <>
      <PageHead label="Dev log" title="How the numbers moved">
        {NAME}’s benchmarks commit by commit, from the initial comparison against other embedded databases through the paired-protocol rerun: what changed and what each change measured. Every figure comes directly from the repository’s benchmark results and commits.
      </PageHead>

      <Section
        id="chart"
        title="Query time against the first measurement"
        lead="BEIR FiQA (57,638 documents), median time of a top-10 query from Python. The numbered markers are the changes below; the rise at the end is the corrected protocol, which fuses deeper to measure quality and speed at one setting, not a slowdown."
      >
        <LogChart />
      </Section>

      <Section id="table" title="First and now" lead="Each configuration's first BEIR row and the corrected run, on FiQA.">
        <DataTable
          label="FiQA median query time, first and now"
          head={['FiQA, median time', 'First', 'Now', 'Change']}
          rows={changes('p50').map((c) => [c.name, ms(c.from), ms(c.to), { value: `${(c.to / c.from).toFixed(2)}×`, strong: c.to < c.from }])}
        />
        <DataTable
          label="FiQA nDCG@10, first and now"
          head={['FiQA, nDCG@10', 'First', 'Now', 'Change']}
          rows={changes('ndcg').map((c) => [c.name, c.from.toFixed(3), c.to.toFixed(3), `${c.to >= c.from ? '+' : '−'}${Math.abs(c.to - c.from).toFixed(3)}`])}
        />
        <Note>
          “First” is each configuration’s first BEIR row ({when(points[0]!.time)} for vectors and words as written, two hours later for the English analyzer); “now” is the corrected run. The hybrid’s quality fell because the earlier rows scored it fused 400 deep while timing it 40 to 100 deep; measured at one setting it fuses 200 and 100. Full text with words as written is the one that got slower: scoring term at a time chose short queries with stopwords and lost, which the term-at-a-time fix mostly undid. The hybrid rows are slower than their best because they now fuse 200 text candidates for quality; <InlineCode>text_candidates=40</InlineCode> buys the time back for FiQA nDCG@10 0.370 → 0.358 on dev queries.
        </Note>
        <div {...stylex.props(styles.more)}>
          <Link to="/benchmarks" {...stylex.props(link.more, focus.ring)}>
            Every current result
            <Chevron />
          </Link>
        </div>
      </Section>

      <Section id="changes" title="Changes" lead="Newest first. A number is the change's marker on the chart; the others are reliability work, which the chart does not mark.">
        <ol {...stylex.props(styles.entries)}>
          {entries.map((e, i) => (
            <li key={e.commit} id={e.commit} {...stylex.props(styles.entry, i === 0 && styles.entryFirst)}>
              <div {...stylex.props(text.meta, styles.meta)}>
                <span>{when(e.time)}</span>
                <code {...stylex.props(styles.commit)}>{e.commit}</code>
                {number.has(e.commit) ? (
                  <span {...stylex.props(styles.number)} aria-label={`Marker ${number.get(e.commit)} on the chart`}>
                    {number.get(e.commit)}
                  </span>
                ) : (
                  <span>{e.kind}</span>
                )}
              </div>
              <h3 {...stylex.props(text.title)}>{e.title}</h3>
              <p {...stylex.props(text.body, styles.body)}>{e.text}</p>
            </li>
          ))}
        </ol>
      </Section>
    </>
  )
}
