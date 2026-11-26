import * as stylex from '@stylexjs/stylex'
import { useState } from 'react'
import { datasets, kinds, ms, points, type DatasetId, type Kind } from '#/data/beir'
import { Key, Point, chart, ink, useReveal, useWidth } from './ChartParts'
import { Segmented } from './Segmented'
import { text } from './styles'

const WIDE = 1100

/** nDCG@10 against median latency (log scale), one point per configuration, Hinny's filled. */
export function Chart({ initialKind = 'hybrid', initialDataset = 'fiqa' }: { initialKind?: Kind; initialDataset?: DatasetId }) {
  const [kind, setKind] = useState<Kind>(initialKind)
  const [dataset, setDataset] = useState<DatasetId>(initialDataset)
  // Drawn at the plot's own width, so text stays its real size on a phone.
  const [plotRef, W] = useWidth(WIDE)
  const [frame, phase] = useReveal<HTMLElement>()
  // The point pointed at: the others dim.
  const [active, setActive] = useState<string | null>(null)
  const narrow = W < 600
  const H = narrow ? 360 : 420
  const M = narrow ? { l: 46, r: 16, t: 26, b: 52 } : { l: 70, r: 40, t: 26, b: 56 }
  const pts = points(kind, dataset)

  const xs = pts.map((p) => p.p50)
  const lo = 10 ** Math.floor(Math.log10(Math.min(...xs)))
  const hi = 10 ** Math.ceil(Math.log10(Math.max(...xs)))
  const x = (v: number) => M.l + ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (W - M.l - M.r)
  const ys = pts.map((p) => p.ndcg)
  const span = Math.max(Math.max(...ys) - Math.min(...ys), 0.01)
  const steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
  let ymin = Math.min(...ys) - span * 0.35
  let ymax = Math.max(...ys) + span * 0.35
  const step = steps.find((s) => (ymax - ymin) / s <= 6) ?? 0.1
  ymin = Math.floor(ymin / step) * step
  ymax = Math.ceil(ymax / step) * step
  const y = (v: number) => M.t + ((ymax - v) / (ymax - ymin)) * (H - M.t - M.b)
  const decimals = step < 0.01 ? 3 : 2

  const yTicks: number[] = []
  for (let v = ymin; v <= ymax + 1e-9; v += step) yTicks.push(v)
  const xTicks: number[] = []
  for (let v = lo; v <= hi * 1.0001; v *= 10) xTicks.push(v)

  // Labels beside their points: right of them, else left, else (on a narrow plot) on a line of
  // their own below. Each moves down or up a line while its box, at about 6.8 units a character,
  // would overlap one already placed.
  const placed: { x0: number; x1: number; y: number }[] = []
  const labelled = [...pts]
    .sort((a, b) => b.ndcg - a.ndcg)
    .map((p) => {
      const cx = x(p.p50)
      const cy = y(p.ndcg)
      const width = (p.name.length + (narrow ? 1 : 18)) * 6.8
      let x0 = cx + 14
      let ly = cy + 4
      if (x0 + width > W - 8) {
        if (cx - 14 - width >= M.l) x0 = cx - 14 - width
        else {
          x0 = Math.max(M.l + 4, W - 8 - width)
          ly = cy + 26
        }
      }
      const x1 = x0 + width
      for (let tries = 0; tries < 8; tries++) {
        const hit = placed.find((q) => Math.abs(q.y - ly) < 16 && q.x0 < x1 && x0 < q.x1)
        if (!hit) break
        ly = hit.y + (ly >= hit.y ? 16 : -16)
      }
      placed.push({ x0, x1, y: ly })
      return { ...p, cx, cy, ly, lx: x0 - cx }
    })

  const current = datasets.find((d) => d.id === dataset)!
  return (
    <figure ref={frame} {...stylex.props(chart.figure, chart.frame)}>
      <div {...stylex.props(chart.controls)}>
        <Segmented label="Kind of search" options={kinds} value={kind} onChange={setKind} />
        <Segmented label="Dataset" options={datasets} value={dataset} onChange={setDataset} />
      </div>
      <div ref={plotRef} {...stylex.props(chart.plot)}>
        <svg viewBox={`0 0 ${W} ${H}`} role="group" aria-label={`${current.name}: nDCG@10 against median query time for ${pts.length} configurations`} {...stylex.props(chart.svg)}>
          {yTicks.map((v) => (
            <g key={`y${v}`}>
              <line x1={M.l} x2={W - M.r} y1={y(v)} y2={y(v)} stroke={ink.line} />
              <text x={M.l - 10} y={y(v) + 4} textAnchor="end" fontSize={12} fill={ink.muted}>
                {v.toFixed(decimals)}
              </text>
            </g>
          ))}
          {xTicks.map((v) => (
            <g key={`x${v}`}>
              <line x1={x(v)} x2={x(v)} y1={M.t} y2={H - M.b} stroke={ink.line} />
              <text x={x(v)} y={H - M.b + 20} textAnchor="middle" fontSize={12} fill={ink.muted}>
                {ms(v)}
              </text>
            </g>
          ))}
          <text x={(M.l + W - M.r) / 2} y={H - 10} textAnchor="middle" fontSize={12} fill={ink.muted}>
            {narrow ? 'median time, log → slower' : 'median query time, log scale → slower'}
          </text>
          {!narrow && (
            <text x={14} y={(M.t + H - M.b) / 2} textAnchor="middle" fontSize={12} fill={ink.muted} transform={`rotate(-90 14 ${(M.t + H - M.b) / 2})`}>
              nDCG@10 → better
            </text>
          )}
          <text x={M.l + 10} y={M.t - 8} fontSize={12} fill={ink.muted}>
            ↖ faster and better
          </text>
          {labelled.map((p, i) => (
            <Point
              key={p.name}
              phase={phase}
              delay={i * 90}
              dim={active !== null && active !== p.name}
              onActive={(on) => setActive((now) => (on ? p.name : now === p.name ? null : now))}
              style={{ transform: `translate(${p.cx}px, ${p.cy}px)` }}
              label={`${p.name}: nDCG@10 ${p.ndcg.toFixed(3)}, p50 ${ms(p.p50)}, p99 ${ms(p.p99)}, Recall@100 ${p.recall.toFixed(3)}`}
              tip={
                <>
                  <span {...stylex.props(chart.tipName)}>{p.name}</span>
                  <br />
                  nDCG@10 {p.ndcg.toFixed(3)} · Recall@100 {p.recall.toFixed(3)}
                  <br />
                  p50 {ms(p.p50)} · p99 {ms(p.p99)}
                </>
              }
            >
              {p.ours ? <rect x={-6.5} y={-6.5} width={13} height={13} rx={2} fill={ink.ours} stroke={ink.bg} strokeWidth={2} /> : <circle r={5.5} fill={ink.bg} stroke={ink.other} strokeWidth={2} />}
              <circle r={18} fill="transparent" />
              <text x={p.lx} y={p.ly - p.cy} fontSize={12.5} fill={p.ours ? ink.fg : ink.muted} fontWeight={p.ours ? 600 : 400}>
                {narrow ? p.name : `${p.name} ${p.ndcg.toFixed(3)} · ${ms(p.p50)}`}
              </text>
            </Point>
          ))}
        </svg>
      </div>
      <div {...stylex.props(chart.legend)}>
        <Key shape="square">Hinny</Key>
        <Key shape="ring">Other systems</Key>
      </div>
      <figcaption {...stylex.props(text.small)}>
        {current.name}: {current.docs.toLocaleString('en-US')} documents. Apple M5 Pro, single-threaded queries from Python, warm, top 10. Every system gets the same documents, EmbeddingGemma vectors and queries.
      </figcaption>
    </figure>
  )
}
