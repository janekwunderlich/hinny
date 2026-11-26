import * as stylex from '@stylexjs/stylex'
import { Fragment, type ReactNode } from 'react'
import { coats, colors, fonts, media, radii, sizes, space, type } from '../tokens.stylex'
import { reveal, stagger, useReveal, type Phase } from './ChartParts'

/* How Hinny is put together, as a diagram: everything inside the application's own process, a
   write going left to right through the log into segments, a query going through a snapshot, the
   retrievers and fusion, and under both the files they stand on. The retrievers wear the coats
   of the home page's cards. The steps come up one after another as the diagram is first seen. */

const styles = stylex.create({
  figure: { display: 'grid', gap: space.x3, margin: 0, minWidth: 0 },
  process: {
    display: 'grid',
    gap: space.x6,
    padding: { default: space.x8, [media.phone]: space.x4 },
    borderWidth: sizes.hair,
    borderStyle: 'solid',
    borderColor: colors.lineStrong,
    borderRadius: radii.card,
  },
  top: { display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: space.x3 },
  label: { fontSize: type.meta, lineHeight: '20px', color: colors.muted },
  langs: { display: 'flex', gap: space.x2 },
  lang: { paddingBlock: '4px', paddingInline: space.x3, borderRadius: radii.pill, backgroundColor: colors.raised, fontFamily: fonts.mono, fontSize: type.micro, lineHeight: '16px', color: colors.fg },
  lane: { display: 'grid', gap: space.x3 },
  row: { display: 'flex', flexDirection: { default: 'row', [media.tablet]: 'column' }, alignItems: 'stretch', gap: space.x2 },
  step: { display: 'grid', gap: space.x1, alignContent: 'start', flexGrow: 1, flexBasis: 0, minWidth: 0, padding: space.x4, borderRadius: '10px', backgroundColor: colors.surface },
  wide: { flexGrow: 2.4 },
  name: { fontSize: type.small, lineHeight: '20px', fontWeight: 500, color: colors.fg },
  note: { fontSize: type.meta, lineHeight: '18px', color: colors.muted },
  arrow: { alignSelf: 'center', flexShrink: 0, color: colors.faint, transform: { default: 'none', [media.tablet]: 'rotate(90deg)' } },
  retrievers: { display: 'grid', gridTemplateColumns: { default: 'repeat(3, minmax(0, 1fr))', [media.phone]: 'minmax(0, 1fr)' }, gap: space.x2, marginTop: space.x1 },
  retriever: { display: 'grid', gap: '2px', padding: space.x3, borderRadius: '8px', backgroundColor: colors.bg },
  dot: { display: 'inline-block', width: '8px', height: '8px', marginRight: space.x2, borderRadius: '50%' },
  sun: { backgroundColor: coats.sun },
  iris: { backgroundColor: coats.iris },
  clover: { backgroundColor: coats.clover },
  rose: { backgroundColor: coats.rose },
  filter: { marginTop: space.x1 },
  disk: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    columnGap: space.x6,
    rowGap: space.x2,
    paddingBlock: space.x4,
    paddingInline: { default: space.x8, [media.phone]: space.x4 },
    borderRadius: radii.card,
    backgroundColor: colors.surface,
  },
  files: { display: 'flex', flexWrap: 'wrap', columnGap: space.x5, rowGap: space.x1, fontFamily: fonts.mono, fontSize: type.micro, lineHeight: '18px', color: colors.fg },
})

function Arrow() {
  return (
    <svg {...stylex.props(styles.arrow)} width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M2 7h9.5M8 3.5 11.5 7 8 10.5" />
    </svg>
  )
}

function Step({ name, note, phase, at, wide = false, children }: { name: string; note?: string; phase: Phase; at: number; wide?: boolean; children?: ReactNode }) {
  return (
    <li {...stylex.props(styles.step, wide && styles.wide, reveal(phase, 'rise'))} style={stagger(at, 90)}>
      <span {...stylex.props(styles.name)}>{name}</span>
      {note && <span {...stylex.props(styles.note)}>{note}</span>}
      {children}
    </li>
  )
}

const WRITE: [string, string][] = [
  ['Batch', 'Atomic upserts & deletes, one seq'],
  ['Write-ahead log', 'CRC32 record, all or nothing'],
  ['Memtable', 'Searchable immediately'],
  ['Segments', 'Immutable mmap files, background merges'],
]

const RETRIEVERS: [keyof typeof DOTS, string, string][] = [
  ['sun', 'Full text', 'BM25, block-max pruning'],
  ['iris', 'Vectors', '1-bit RaBitQ, f16 rerank'],
  ['clover', 'Learned sparse', 'Shared posting lists'],
]
const DOTS = { sun: styles.sun, iris: styles.iris, clover: styles.clover, rose: styles.rose } as const

export function Architecture() {
  const [ref, phase] = useReveal<HTMLElement>()
  return (
    <figure ref={ref} aria-label="Hinny's architecture: a write and a query inside the application's process, and the files under them" {...stylex.props(styles.figure)}>
      <div {...stylex.props(styles.process)}>
        <div {...stylex.props(styles.top)}>
          <span {...stylex.props(styles.label)}>Your application's process</span>
          <span {...stylex.props(styles.langs)}>
            {['Rust', 'Python', 'Node.js', 'C'].map((l) => (
              <span key={l} {...stylex.props(styles.lang)}>
                {l}
              </span>
            ))}
          </span>
        </div>

        <div {...stylex.props(styles.lane)}>
          <span {...stylex.props(styles.label)}>Write path</span>
          <ol {...stylex.props(styles.row)}>
            {WRITE.map(([name, note], i) => (
              <Fragment key={name}>
                {i > 0 && <Arrow />}
                <Step name={name} note={note} phase={phase} at={i} />
              </Fragment>
            ))}
          </ol>
        </div>

        <div {...stylex.props(styles.lane)}>
          <span {...stylex.props(styles.label)}>Query path</span>
          <ol {...stylex.props(styles.row)}>
            <Step name="Snapshot" note="Segments and memtable as of now, lock-free" phase={phase} at={4} />
            <Arrow />
            <Step name="Retrievers, in parallel" phase={phase} at={5} wide>
              <span {...stylex.props(styles.retrievers)}>
                {RETRIEVERS.map(([coat, name, note]) => (
                  <span key={name} {...stylex.props(styles.retriever)}>
                    <span {...stylex.props(styles.name)}>
                      <i {...stylex.props(styles.dot, DOTS[coat])} />
                      {name}
                    </span>
                    <span {...stylex.props(styles.note)}>{note}</span>
                  </span>
                ))}
              </span>
              <span {...stylex.props(styles.note, styles.filter)}>
                <i {...stylex.props(styles.dot, DOTS.rose)} />
                one filter, evaluated inside each
              </span>
            </Step>
            <Arrow />
            <Step name="Fusion" note="Scores scaled to 0–1, then combined" phase={phase} at={6} />
            <Arrow />
            <Step name="Hits" note="One ranking, with stored text and snippets" phase={phase} at={7} />
          </ol>
        </div>
      </div>

      <div {...stylex.props(styles.disk, reveal(phase, 'rise'))} style={stagger(8, 90)}>
        <span {...stylex.props(styles.label)}>On local disk</span>
        <span {...stylex.props(styles.files)}>
          <span>write-ahead log</span>
          <span>segments</span>
          <span>stored text</span>
          <span>manifest</span>
        </span>
      </div>
    </figure>
  )
}
