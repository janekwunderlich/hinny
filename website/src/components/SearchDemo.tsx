import * as stylex from '@stylexjs/stylex'
import { useCallback, useEffect, useRef, useState } from 'react'
import { colors, fonts, media, radii, sizes, space, type } from '../tokens.stylex'
import { Segmented } from './Segmented'
import { text } from './styles'

/** Three queries and what a hybrid query might return for each, made up for the demo: each hit's
 * fused score is half its text score (BM25, scaled to 0–1) and half its vector score. One query
 * needs meaning, one an exact token, one both. */
const EXAMPLES = [
  {
    id: 'lease',
    name: 'a lease',
    query: 'lease for the apartment, 2025',
    filter: 'year >= 2025',
    note: 'Words and meaning agree on the lease; meaning alone finds the contract that never says "lease".',
    hits: [
      { title: 'Lease agreement, Elm Street', kind: 'pdf', text: 0.92, vector: 0.88 },
      { title: 'Re: the rental contract', kind: 'mail', text: 0.41, vector: 0.93 },
      { title: 'Apartment handover checklist', kind: 'md', text: 0.66, vector: 0.61 },
      { title: 'Deposit receipt, March', kind: 'pdf', text: 0.12, vector: 0.72 },
    ],
  },
  {
    id: 'flights',
    name: 'cheap flights',
    query: 'cheap flights to lisbon',
    filter: 'kind = "mail"',
    note: 'Nothing says "cheap flights": the vectors find the fare alert, the words find Lisbon.',
    hits: [
      { title: 'Fare alert: LIS from 89 €', kind: 'mail', text: 0.38, vector: 0.94 },
      { title: 'Your booking: Lisbon, 14 March', kind: 'mail', text: 0.81, vector: 0.77 },
      { title: 'Weekend in Porto?', kind: 'mail', text: 0.08, vector: 0.71 },
      { title: 'Lisbon apartment for the trip', kind: 'mail', text: 0.64, vector: 0.39 },
    ],
  },
  {
    id: 'error',
    name: 'an error code',
    query: 'error 0x80070005',
    filter: 'none',
    note: 'An exact code: the words carry it, and the vectors add the fix that never quotes it.',
    hits: [
      { title: 'Update fails with 0x80070005', kind: 'md', text: 0.97, vector: 0.64 },
      { title: 'Access denied during install', kind: 'md', text: 0.04, vector: 0.82 },
      { title: 'Error codes, a cheat sheet', kind: 'pdf', text: 0.71, vector: 0.38 },
      { title: 'Fixing folder permissions', kind: 'md', text: 0.22, vector: 0.61 },
    ],
  },
] as const
type ExampleId = (typeof EXAMPLES)[number]['id']

const blink = stylex.keyframes({ '50%': { opacity: 0 } })

const styles = stylex.create({
  demo: { margin: 0, minWidth: 0, borderWidth: sizes.hair, borderStyle: 'solid', borderColor: colors.line, borderRadius: radii.card, overflow: 'hidden' },
  bar: { display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: space.x3, minHeight: '48px', paddingBlock: space.x2, paddingInline: space.x3, borderBottomWidth: sizes.hair, borderBottomStyle: 'solid', borderBottomColor: colors.line },
  call: { fontFamily: fonts.mono, fontSize: type.meta, color: colors.muted, paddingInline: space.x2 },
  line: { display: 'grid', gridTemplateColumns: '4.5rem minmax(0, 1fr)', alignItems: 'baseline', paddingBlock: space.x3, paddingInline: space.x5, borderBottomWidth: sizes.hair, borderBottomStyle: 'solid', borderBottomColor: colors.line, fontFamily: fonts.mono, fontSize: type.code, lineHeight: 1.7 },
  prompt: { color: colors.muted },
  input: { color: colors.fg, minHeight: '1.4em', whiteSpace: 'nowrap', overflow: 'hidden' },
  caret: { display: 'inline-block', width: '0.5em', height: '0.12em', backgroundColor: colors.fg, marginLeft: '0.15em', animationName: { default: 'none', [media.motion]: blink }, animationDuration: '1.1s', animationTimingFunction: 'steps(1)', animationIterationCount: 'infinite' },
  hits: { paddingBlock: space.x2 },
  hit: {
    display: 'grid',
    gridTemplateColumns: { default: '1.6rem minmax(0, 1fr) 7rem 2.6rem', [media.phone]: '1.2rem minmax(0, 1fr) 4.5rem 2.4rem' },
    gap: space.x3,
    alignItems: 'center',
    paddingBlock: '6px',
    paddingInline: space.x5,
    fontSize: type.small,
    lineHeight: '22px',
    opacity: 0,
    transform: 'translateY(3px)',
    transitionProperty: 'opacity, transform',
    transitionDuration: '220ms',
  },
  shown: { opacity: 1, transform: 'none' },
  rank: { color: colors.muted, fontVariantNumeric: 'tabular-nums' },
  title: { color: colors.fg, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
  kind: { marginLeft: space.x2, color: colors.muted, fontFamily: fonts.mono, fontSize: type.micro, display: { default: 'inline', [media.phone]: 'none' } },
  score: { display: 'flex', gap: '2px', height: '6px' },
  words: { backgroundColor: colors.fg },
  meaning: { backgroundColor: colors.other },
  value: { textAlign: 'right', color: colors.fg, fontVariantNumeric: 'tabular-nums' },
  legend: { display: 'flex', flexWrap: 'wrap', columnGap: space.x5, rowGap: space.x1, paddingBlock: space.x3, paddingInline: space.x5, borderTopWidth: sizes.hair, borderTopStyle: 'solid', borderTopColor: colors.line },
  key: { display: 'inline-block', width: '8px', height: '8px', marginRight: '6px' },
  note: { flexBasis: '100%', color: colors.text },
})

/** A query typed into Hinny, its filter, and the fused ranking it gets back, for one of three
 * examples. Rendered finished on the server; in the browser it types itself when it comes into
 * view, unless motion is reduced. */
export function SearchDemo() {
  const [current, setCurrent] = useState<ExampleId>('lease')
  const example = EXAMPLES.find((e) => e.id === current)!
  const [typed, setTyped] = useState<number>(example.query.length)
  const [shown, setShown] = useState<number>(example.hits.length)
  const [run, setRun] = useState(0)
  // It types itself once it's on screen, so a reader scrolling down sees it happen.
  const [seen, setSeen] = useState(false)
  const io = useRef<IntersectionObserver | null>(null)
  const ref = useCallback((el: HTMLElement | null) => {
    io.current?.disconnect()
    io.current = null
    if (!el) return
    io.current = new IntersectionObserver(([e]) => e?.isIntersecting && setSeen(true), { threshold: 0.4 })
    io.current.observe(el)
  }, [])

  useEffect(() => {
    if (!seen) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setTyped(example.query.length)
      setShown(example.hits.length)
      return
    }
    setTyped(0)
    setShown(0)
    const timers: number[] = []
    const start = run === 0 ? 400 : 120
    for (let i = 1; i <= example.query.length; i++) timers.push(window.setTimeout(() => setTyped(i), start + i * 45))
    const done = start + example.query.length * 45
    for (let j = 1; j <= example.hits.length; j++) timers.push(window.setTimeout(() => setShown(j), done + 200 + j * 80))
    return () => timers.forEach(clearTimeout)
  }, [current, run, example, seen])

  const ranked = [...example.hits].map((h) => ({ ...h, score: (h.text + h.vector) / 2 })).sort((a, b) => b.score - a.score)
  const finished = typed === example.query.length && shown === example.hits.length
  return (
    <figure ref={ref} aria-label={`A hybrid query for “${example.query}” with the filter ${example.filter}, and its four best results`} {...stylex.props(styles.demo)}>
      <div {...stylex.props(styles.bar)}>
        <Segmented
          label="Example queries"
          options={EXAMPLES}
          value={current}
          onChange={(id) => {
            setCurrent(id)
            setRun((n) => n + 1)
          }}
        />
        <span {...stylex.props(styles.call)}>{finished ? 'ns.query()' : ''}</span>
      </div>
      <div {...stylex.props(styles.line)}>
        <span {...stylex.props(styles.prompt)}>text</span>
        <span {...stylex.props(styles.input)}>
          {example.query.slice(0, typed)}
          <span aria-hidden="true" {...stylex.props(styles.caret)} />
        </span>
      </div>
      <div {...stylex.props(styles.line)}>
        <span {...stylex.props(styles.prompt)}>filter</span>
        <span {...stylex.props(styles.input)}>{example.filter}</span>
      </div>
      <ol {...stylex.props(styles.hits)}>
        {ranked.map((h, i) => (
          <li key={h.title} {...stylex.props(styles.hit, i < shown && styles.shown)}>
            <span {...stylex.props(styles.rank)}>{i + 1}</span>
            <span {...stylex.props(styles.title)}>
              {h.title}
              <span {...stylex.props(styles.kind)}>{h.kind}</span>
            </span>
            <span title={`text ${h.text.toFixed(2)} · vector ${h.vector.toFixed(2)}`} {...stylex.props(styles.score)}>
              <span {...stylex.props(styles.words)} style={{ width: `${h.text * 50}%` }} />
              <span {...stylex.props(styles.meaning)} style={{ width: `${h.vector * 50}%` }} />
            </span>
            <span {...stylex.props(styles.value)}>{h.score.toFixed(2)}</span>
          </li>
        ))}
      </ol>
      <figcaption {...stylex.props(text.meta, styles.legend)}>
        <span>
          <i {...stylex.props(styles.key, styles.words)} />
          words (BM25)
        </span>
        <span>
          <i {...stylex.props(styles.key, styles.meaning)} />
          meaning (vectors)
        </span>
        <span {...stylex.props(styles.note)}>{example.note}</span>
      </figcaption>
    </figure>
  )
}
