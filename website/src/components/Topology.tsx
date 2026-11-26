import * as stylex from '@stylexjs/stylex'
import type { CSSProperties, ReactNode } from 'react'
import { coats, colors, fonts, media, radii, space, type } from '../tokens.stylex'
import { reveal, stagger, useReveal } from './ChartParts'

/* Small pictures of the engine, each in a card like the home page's others: a heading, a line
   of text, and a quiet diagram: chips joined by dashed lines, rows of cells, or two bars. The
   dashes march along the lines a query or a write travels (not where motion is reduced). A
   diagram with lines is one SVG on a stage of 360 by 220 units, so it scales with its card and
   nothing can leave it. The first card is the lead: it takes a whole row, words beside the
   picture. Every figure in them is from DESIGN.md or the test suite it describes. */

type Tone = 'sun' | 'iris' | 'clover' | 'rose'

const march = stylex.keyframes({ to: { strokeDashoffset: -20 } })
const lit = stylex.keyframes({
  '0%, 20%': { backgroundColor: 'var(--lit)' },
  '32%, 100%': { backgroundColor: colors.lineStrong },
})

const s = stylex.create({
  cards: { display: 'grid', gridTemplateColumns: { default: 'repeat(3, minmax(0, 1fr))', '@media (max-width: 1179px)': 'repeat(2, minmax(0, 1fr))', '@media (max-width: 759px)': 'minmax(0, 1fr)' }, gap: space.x3 },
  card: { display: 'grid', gridTemplateRows: 'auto minmax(0, 1fr)', gap: space.x6, padding: { default: space.x8, [media.phone]: space.x6 }, borderRadius: radii.card, backgroundColor: colors.surface, minWidth: 0 },
  // The lead card: a row to itself, the words beside the picture where there is room.
  cardWide: {
    gridColumn: '1 / -1',
    gridTemplateRows: { default: 'auto', '@media (max-width: 759px)': 'auto minmax(0, 1fr)' },
    gridTemplateColumns: { default: 'minmax(0, 5fr) minmax(0, 7fr)', '@media (max-width: 759px)': 'minmax(0, 1fr)' },
    alignItems: 'center',
    columnGap: space.x12,
    padding: { default: space.x12, '@media (max-width: 1179px)': space.x8, [media.phone]: space.x6 },
  },
  words: { display: 'grid', gap: space.x2 },
  wordsWide: { gap: space.x3 },
  titleWide: { fontSize: type.h2, lineHeight: 1.2, letterSpacing: '-0.022em' },
  textWide: { fontSize: type.body, lineHeight: 1.65, maxWidth: '34em' },
  stageWide: { maxWidth: '560px' },
  title: { fontSize: '20px', lineHeight: '28px', fontWeight: 500, letterSpacing: '-0.015em', color: colors.fg },
  text: { fontSize: type.small, lineHeight: 1.6, color: colors.muted, maxWidth: '40ch' },

  stage: { display: 'block', width: '100%', maxWidth: '400px', height: 'auto', justifySelf: 'center', alignSelf: 'center' },
  label: { fontFamily: fonts.mono, fontSize: '11.5px' },
  small: { fontSize: '11px' },
  line: { fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'square', strokeDasharray: '5 5', animationName: { default: 'none', [media.motion]: march }, animationDuration: '1.4s', animationTimingFunction: 'linear', animationIterationCount: 'infinite' },
  dot: { width: '6px', height: '6px', borderRadius: '50%', flexShrink: 0 },
  sun: { backgroundColor: coats.sun },
  iris: { backgroundColor: coats.iris },
  clover: { backgroundColor: coats.clover },
  rose: { backgroundColor: coats.rose },

  track: { height: '22px', borderRadius: '6px', backgroundColor: colors.raised, overflow: 'hidden' },
  fill: { display: 'block', height: '100%', borderRadius: '6px', backgroundColor: colors.lineStrong },
  fillOurs: { backgroundColor: colors.accent },
  kills: { display: 'grid', gridTemplateColumns: 'repeat(20, minmax(0, 1fr))', gap: '3px' },
  kill: { height: '16px', animationDuration: '6s' },
  // The second card: rows of cells that light a group at a time.
  rows: { display: 'grid', gap: space.x5, alignContent: 'center', width: '100%', maxWidth: '400px', justifySelf: 'center' },
  rowHead: { display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: space.x2, fontFamily: fonts.mono, fontSize: '11.5px', lineHeight: '16px' },
  rowName: { display: 'inline-flex', alignItems: 'center', gap: '7px', color: colors.fg },
  rowValue: { color: colors.muted },
  cells: { display: 'grid', gridTemplateColumns: 'repeat(32, minmax(0, 1fr))', gap: '2px' },
  cell: { height: '14px', borderRadius: '2px', backgroundColor: colors.lineStrong, animationName: { default: 'none', [media.motion]: lit }, animationDuration: '4s', animationTimingFunction: 'ease-out', animationIterationCount: 'infinite' },
})

const DOT = { sun: s.sun, iris: s.iris, clover: s.clover, rose: s.rose } as const
const VAR = { sun: coats.sun, iris: coats.iris, clover: coats.clover, rose: coats.rose } as const

/** A chip centred on a place of the stage, `w` units wide. Drawn in the SVG, so it scales with
 * the lines that meet it and can't leave the card. */
function Chip({ x, y, w, tone, strong = false, children }: { x: number; y: number; w: number; tone?: Tone; strong?: boolean; children: string }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      <rect x={-w / 2} y={-15} width={w} height={30} rx={10} fill={strong ? colors.fg : colors.raised} />
      {tone && <circle cx={-w / 2 + 14} cy={0} r={3} fill={VAR[tone]} />}
      <text x={tone ? 7 : 0} y={0.5} textAnchor="middle" dominantBaseline="central" fill={strong ? colors.bg : colors.fg} {...stylex.props(s.label)}>
        {children}
      </text>
    </g>
  )
}

function Source({ x, y, d }: { x: number; y: number; d: string }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      <circle r={20} fill={colors.raised} />
      <path d={d} transform="translate(-8 -8)" fill="none" stroke={colors.fg} strokeWidth={1.4} strokeLinecap="round" strokeLinejoin="round" />
    </g>
  )
}

/** The stage: dashed lines first, then what is drawn on them, then a line of small print. */
function Stage({ paths, note, noteTone, wide = false, children }: { paths: string[]; note: string; noteTone?: Tone; wide?: boolean; children: ReactNode }) {
  return (
    <svg {...stylex.props(s.stage, wide && s.stageWide)} viewBox="0 0 360 220" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      <g color={colors.faint}>
        {paths.map((d, i) => (
          <path key={d} d={d} {...stylex.props(s.line)} style={{ animationDelay: `${-i * 0.23}s` }} />
        ))}
      </g>
      {children}
      {noteTone && <circle cx={180 - note.length * 3.3 - 9} cy={210} r={3} fill={VAR[noteTone]} />}
      <text x={180} y={210.5} textAnchor="middle" dominantBaseline="central" fill={colors.muted} {...stylex.props(s.label, s.small)}>
        {note}
      </text>
    </svg>
  )
}

function Card({ title, text, label, at, wide = false, children }: { title: string; text: string; label: string; at: number; wide?: boolean; children: ReactNode }) {
  const [ref, phase] = useReveal<HTMLDivElement>()
  return (
    <div ref={ref} {...stylex.props(s.card, wide && s.cardWide, reveal(phase, 'rise'))} style={stagger(at, 90)}>
      <div {...stylex.props(s.words, wide && s.wordsWide)}>
        <h3 {...stylex.props(s.title, wide && s.titleWide)}>{title}</h3>
        <p {...stylex.props(s.text, wide && s.textWide)}>{text}</p>
      </div>
      <div role="img" aria-label={label} style={{ display: 'grid' }}>
        {children}
      </div>
    </div>
  )
}

const SEARCH = 'M7.2 12.4a5.2 5.2 0 1 0 0-10.4 5.2 5.2 0 0 0 0 10.4ZM11 11l3.5 3.5'
const PEN = 'M3 13l.6-2.8 7.2-7.2 2.2 2.2-7.2 7.2ZM9.6 4.2l2.2 2.2'

/** One call runs every retriever, with one filter inside each, and comes back as one ranking. */
export function HybridCard({ at = 0, wide = false }: { at?: number; wide?: boolean }) {
  return (
    <Card at={at} wide={wide} title="Truly hybrid" text="Retrievers aren't separate queries for your app to stitch together. One call runs them in parallel, evaluates the same filter inside each, and fuses their scores in the engine." label="A query fans out to full text, vectors and learned sparse terms, each with the same filter, and comes back as one ranking">
      <Stage wide={wide} note="one filter, inside each" noteTone="rose" paths={['M 52 96 C 82 96, 82 30, 108 30', 'M 52 96 H 108', 'M 52 96 C 82 96, 82 162, 108 162', 'M 226 30 C 252 30, 250 96, 270 96', 'M 226 96 H 270', 'M 226 162 C 252 162, 250 96, 270 96']}>
        <Source x={32} y={96} d={SEARCH} />
        <Chip x={167} y={30} w={118} tone="sun">
          full text
        </Chip>
        <Chip x={167} y={96} w={118} tone="iris">
          vectors
        </Chip>
        <Chip x={167} y={162} w={118} tone="clover">
          sparse terms
        </Chip>
        <Chip x={314} y={96} w={88} strong>
          1 ranking
        </Chip>
      </Stage>
    </Card>
  )
}

const STAGES: { name: string; tone: Tone; per: number; value: string }[] = [
  { name: 'filter', tone: 'rose', per: 16, value: '64 codes to a word of bits' },
  { name: 'postings', tone: 'sun', per: 8, value: '128 documents a block' },
  { name: 'vector codes', tone: 'iris', per: 8, value: '32 codes a table lookup' },
]

/** No stage handles one document at a time: each row lights the group it takes in one step. */
export function ExecutionCard({ at = 0 }: { at?: number }) {
  return (
    <Card at={at} title="Block-at-a-time execution" text="No stage of a query handles one document at a time. Filters, posting lists, and vector codes are all processed in SIMD blocks." label="Filters are evaluated 64 codes to a word, postings 128 documents a block, vector codes 32 at a time">
      <div {...stylex.props(s.rows)}>
        {STAGES.map((stage) => (
          <div key={stage.name}>
            <div {...stylex.props(s.rowHead)}>
              <span {...stylex.props(s.rowName)}>
                <i {...stylex.props(s.dot, DOT[stage.tone])} />
                {stage.name}
              </span>
              <span {...stylex.props(s.rowValue)}>{stage.value}</span>
            </div>
            <div {...stylex.props(s.cells)} style={{ '--lit': VAR[stage.tone] } as CSSProperties}>
              {Array.from({ length: 32 }, (_, i) => (
                <i key={i} {...stylex.props(s.cell)} style={{ animationDelay: `${(Math.floor(i / stage.per) % 4) * 1}s` }} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}

/** A write goes to the log and the memtable; segments are written and merged behind it. */
export function StorageCard({ at = 0 }: { at?: number }) {
  return (
    <Card at={at} title="Log and segments" text="Writes land in the WAL and become searchable immediately. Immutable segments flush and merge in the background." label="A write goes to the write-ahead log and the memtable, which is flushed into immutable segments, merged four at a time">
      <Stage note="segments, merged four of a size at a time" paths={['M 52 44 H 88', 'M 144 44 H 190', 'M 244 60 C 244 104, 62 84, 62 126', 'M 108 138 C 126 116, 150 116, 168 138', 'M 216 138 C 232 120, 250 120, 266 138']}>
        <Source x={32} y={44} d={PEN} />
        <Chip x={116} y={44} w={56}>
          log
        </Chip>
        <Chip x={244} y={44} w={108} tone="iris">
          memtable
        </Chip>
        {[[16, 92], [168, 48], [266, 36], [316, 22]].map(([x, w]) => (
          <rect key={x} x={x} y={138} width={w} height={18} rx={6} fill={colors.raisedHover} />
        ))}
      </Stage>
    </Card>
  )
}

/** Two amounts as bars from zero, the smaller in the accent: a claim that is a ratio. */
function Compare({ rows }: { rows: { name: string; value: string; share: number; ours?: boolean }[] }) {
  const [ref, phase] = useReveal<HTMLDivElement>()
  return (
    <div ref={ref} {...stylex.props(s.rows)}>
      {rows.map((r, i) => (
        <div key={r.name}>
          <div {...stylex.props(s.rowHead)}>
            <span {...stylex.props(s.rowName)}>{r.name}</span>
            <span {...stylex.props(r.ours ? s.rowName : s.rowValue)}>{r.value}</span>
          </div>
          <div {...stylex.props(s.track)}>
            <span {...stylex.props(s.fill, r.ours && s.fillOurs, reveal(phase, 'grow'))} style={{ width: `${Math.max(r.share * 100, 1.2)}%`, ...stagger(i, 160, 200) }} />
          </div>
        </div>
      ))}
    </div>
  )
}

/** A million vectors as one bit a dimension. */
export function BitsCard({ at = 0 }: { at?: number }) {
  return (
    <Card at={at} title="One bit per dimension" text="Queries scan 1-bit RaBitQ codes and fetch half-precision vectors only when the error bound requires it: 0.04% of a million." label="The 1-bit codes of a million 256-dimension vectors take about 40 megabytes; the vectors themselves take a gigabyte">
      <Compare
        rows={[
          { name: '1M vectors, 256 dimensions', value: '1 GB', share: 1 },
          { name: 'their 1-bit codes', value: '40 MB', share: 0.04, ours: true },
        ]}
      />
    </Card>
  )
}

/** Writers and readers don't block each other. */
export function WaitCard({ at = 0 }: { at?: number }) {
  return (
    <Card at={at} title="Readers never wait" text="Queries read an immutable snapshot with no lock, so concurrent writes and searches don't block each other." label="Writing 80,000 documents while another thread queries without pause: 19 seconds with a reader-writer lock, 1.1 seconds with views">
      <Compare
        rows={[
          { name: '80,000 writes, behind a lock', value: '19 s', share: 1 },
          { name: 'the same, with views', value: '1.1 s', share: 1.1 / 19, ours: true },
        ]}
      />
    </Card>
  )
}

/** Crash safety, and the test behind the claim: a writer killed forty times. */
export function KillsCard({ at = 0 }: { at?: number }) {
  return (
    <Card at={at} title="Crash-safe writes" text="Kill the process mid-write. Every acknowledged batch comes back on reopen, whole and in order." label="Forty kills of a writing process, each survived; and 344 injected I/O failures, each recovered from">
      <div {...stylex.props(s.rows)}>
        <div>
          <div {...stylex.props(s.rowHead)}>
            <span {...stylex.props(s.rowName)}>
              <i {...stylex.props(s.dot, s.clover)} />
              kill -9, then reopen
            </span>
            <span {...stylex.props(s.rowValue)}>40 of 40 whole</span>
          </div>
          <div {...stylex.props(s.kills)} style={{ '--lit': coats.clover } as CSSProperties}>
            {Array.from({ length: 40 }, (_, i) => (
              <i key={i} {...stylex.props(s.cell, s.kill)} style={{ animationDelay: `${i * 0.1}s` }} />
            ))}
          </div>
        </div>
        <div {...stylex.props(s.rowHead)}>
          <span {...stylex.props(s.rowName)}>I/O failures injected</span>
          <span {...stylex.props(s.rowValue)}>344 cases</span>
        </div>
      </div>
    </Card>
  )
}

/** Hinny is inside the application's process: a query is a function call. The usual picture
 * (an application, a network, a search server) is drawn above it, dim, for comparison. */
export function NoServerCard({ at = 0 }: { at?: number }) {
  return (
    <Card at={at} title="No server" text="Hinny is a library inside your application's process. A query is a function call, not a request over a network." label="Usually an application reaches a search server over a network; with Hinny the engine is inside the application's process, reading local files">
      <Stage note="a function call, not a network round trip" paths={['M 222 152 H 262']}>
        <text x={14} y={22} fill={colors.muted} {...stylex.props(s.label, s.small)}>
          a search service
        </text>
        <g opacity={0.5}>
          <path d="M 104 58 H 256" fill="none" stroke={colors.muted} strokeWidth={1.5} strokeDasharray="2 6" strokeLinecap="round" />
          <Chip x={58} y={58} w={92}>
            your app
          </Chip>
          <text x={180} y={47} textAnchor="middle" fill={colors.muted} {...stylex.props(s.label, s.small)}>
            network
          </text>
          <Chip x={302} y={58} w={92}>
            server
          </Chip>
        </g>
        <text x={14} y={112} fill={colors.muted} {...stylex.props(s.label, s.small)}>
          with hinny
        </text>
        <rect x={12} y={124} width={214} height={56} rx={14} fill="none" stroke={colors.accent} strokeWidth={1.25} />
        <Chip x={66} y={152} w={92}>
          your app
        </Chip>
        <Chip x={170} y={152} w={92} tone="iris">
          hinny
        </Chip>
        <Chip x={306} y={152} w={88}>
          files
        </Chip>
      </Stage>
    </Card>
  )
}

/** The cards: the lead across the top, then three to a row. The first row is how a query runs,
 * the second how a write is kept. */
export function EngineCards() {
  return (
    <div {...stylex.props(s.cards)}>
      <HybridCard at={0} wide />
      <NoServerCard at={1} />
      <ExecutionCard at={2} />
      <BitsCard at={3} />
      <WaitCard at={4} />
      <StorageCard at={5} />
      <KillsCard at={6} />
    </div>
  )
}
