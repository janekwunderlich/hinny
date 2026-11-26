import * as stylex from '@stylexjs/stylex'
import { useState } from 'react'
import { datasets, hybrids, ms, results } from '#/data/beir'
import { Key, Point, chart, ink, motion, reveal, stagger, useReveal, useWidth } from './ChartParts'

/** The hybrid configurations it compares, in the order the legend lists them, each with a shape
 * of its own: the chart reads without colour. */
const SHAPES = ['square', 'ring', 'dot'] as const
const SYSTEMS = hybrids.map((h, i) => ({ ...h, mode: 'hybrid', shape: SHAPES[i]! }))

type Mark = { dataset: string; name: string; ours: boolean; shape: (typeof SHAPES)[number]; p50: number; ndcg: number }

const WIDE = 1200

/** The median hybrid query on every dataset, one row each on a log scale, with how many times
 * longer LanceDB took than Hinny. */
export function LatencyStrip() {
  const [ref, W] = useWidth(WIDE)
  const [frame, phase] = useReveal<HTMLElement>()
  // The mark pointed at, as "dataset/system": the others dim.
  const [active, setActive] = useState<string | null>(null)

  const narrow = W < 520
  const rows = datasets.map((d) => {
    const marks: Mark[] = SYSTEMS.flatMap((s) => {
      const r = results.find((r) => r.dataset === d.id && r.system === s.system && r.mode === s.mode)
      return r ? [{ dataset: d.name, name: s.name, ours: s.ours, shape: s.shape, p50: r.p50_ms, ndcg: r.ndcg10 }] : []
    })
    const ours = marks.find((m) => m.ours)
    const lance = marks.find((m) => m.name === 'LanceDB')
    return { ...d, marks, times: ours && lance ? lance.p50 / ours.p50 : undefined }
  })

  const row = 44
  const M = { l: narrow ? 70 : 92, r: narrow ? 44 : 72, t: 30, b: 34 }
  const H = M.t + rows.length * row + M.b
  const lo = 0.05
  const hi = 500
  const x = (v: number) => M.l + ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (W - M.l - M.r)
  const ticks = [0.1, 1, 10, 100]
  // The systems are named once, over the row where their marks lie furthest apart.
  const gap = (r: (typeof rows)[number]) => {
    const xs = r.marks.map((m) => x(m.p50)).sort((a, b) => a - b)
    return Math.min(...xs.slice(1).map((v, i) => v - xs[i]!))
  }
  const named = rows.reduce((best, r) => (gap(r) > gap(best) ? r : best), rows[0]!)
  const namedY = M.t + rows.indexOf(named) * row + row / 2

  return (
    <figure ref={frame} {...stylex.props(chart.figure, chart.frame)}>
      <div ref={ref} {...stylex.props(chart.plot)}>
        <svg viewBox={`0 0 ${W} ${H}`} role="group" aria-label={`Median time of a hybrid query on ${rows.length} datasets, log scale`} {...stylex.props(chart.svg)}>
          {ticks.map((v) => (
            <g key={v}>
              <line x1={x(v)} x2={x(v)} y1={M.t - 6} y2={H - M.b + 4} stroke={ink.line} />
              <text x={x(v)} y={H - 10} textAnchor="middle" fontSize={12} fill={ink.muted}>
                {ms(v)}
              </text>
            </g>
          ))}
          <text x={W} y={M.t - 12} textAnchor="end" fontSize={12} fill={ink.muted}>
            {narrow ? '' : 'vs LanceDB'}
          </text>
          {named.marks.map((m) => (
            <text key={m.name} x={x(m.p50)} y={namedY - 12} textAnchor="middle" fontSize={12} fill={m.ours ? ink.fg : ink.muted} fontWeight={m.ours ? 600 : 400}>
              {m.name}
            </text>
          ))}
          {rows.map((r, i) => {
            const cy = M.t + i * row + row / 2
            const xs = r.marks.map((m) => x(m.p50))
            return (
              <g key={r.id}>
                <text x={0} y={cy + 4} fontSize={13} fill={ink.fg}>
                  {r.name}
                </text>
                <line x1={Math.min(...xs)} x2={Math.max(...xs)} y1={cy} y2={cy} stroke={ink.lineStrong} strokeWidth={2} strokeLinecap="round" {...stylex.props(motion.box, reveal(phase, 'grow'))} style={stagger(i, 110)} />
                {r.marks.map((m, j) => {
                  const cx = x(m.p50)
                  const id = `${r.id}/${m.name}`
                  return (
                    <Point
                      key={m.name}
                      phase={phase}
                      delay={i * 110 + 250 + j * 120}
                      dim={active !== null && active !== id}
                      onActive={(on) => setActive((now) => (on ? id : now === id ? null : now))}
                      label={`${m.name} on ${m.dataset}: ${ms(m.p50)}, nDCG@10 ${m.ndcg.toFixed(3)}`}
                      tip={
                        <>
                          <span {...stylex.props(chart.tipName)}>{m.name}</span> · {m.dataset}
                          <br />
                          p50 {ms(m.p50)} · nDCG@10 {m.ndcg.toFixed(3)}
                        </>
                      }
                    >
                      {m.shape === 'square' && <rect x={cx - 6.5} y={cy - 6.5} width={13} height={13} rx={2} fill={ink.ours} stroke={ink.bg} strokeWidth={2} />}
                      {m.shape === 'ring' && <circle cx={cx} cy={cy} r={5.5} fill={ink.bg} stroke={ink.other} strokeWidth={2} />}
                      {m.shape === 'dot' && <circle cx={cx} cy={cy} r={5.5} fill={ink.other} stroke={ink.bg} strokeWidth={2} />}
                      <circle cx={cx} cy={cy} r={14} fill="transparent" />
                    </Point>
                  )
                })}
                {r.times && (
                  <text x={W} y={cy + 4} textAnchor="end" fontSize={13} fill={ink.fg} fontWeight={600} {...stylex.props(reveal(phase, 'fade'))} style={stagger(i, 110, 700)}>
                    {Math.round(r.times)}×
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      </div>
      <figcaption {...stylex.props(chart.legend)}>
        <Key shape="square">Hinny, BM25 + vectors</Key>
        <Key shape="ring">LanceDB</Key>
        <Key shape="dot">SQLite FTS5 + sqlite-vec</Key>
        <span>median hybrid query, log scale</span>
      </figcaption>
    </figure>
  )
}
