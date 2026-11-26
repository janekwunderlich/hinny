import * as stylex from '@stylexjs/stylex'
import { entries, points, series, type SeriesKey } from '#/data/devlog'
import { ms } from '#/data/beir'
import { Key, Point, chart, ink, reveal, stagger, useReveal, useWidth } from './ChartParts'
import { text } from './styles'

/** Each configuration's colour, in an order checked for colour-vision separation. The values are
 * the series colours of the tokens, a set for light and a set for dark. */
const COLORS: Record<SeriesKey, string> = { hybrid: ink.series[0], vector: ink.series[1], text: ink.series[2], textPlain: ink.series[3] }

const WIDE = 820
const time = (iso: string) => new Date(iso).getTime()

/** The changes with a marker: everything but reliability work since the first measurement, in
 * time order, numbered. */
export const marked = entries
  .filter((e) => e.kind !== 'reliability' && time(e.time) >= time(points[0]!.time))
  .sort((a, b) => time(a.time) - time(b.time))
  .map((e, i) => ({ ...e, n: i + 1 }))

/** FiQA's median query time for each configuration, relative to its first measurement, on a log
 * scale: a step at each version of the results, a numbered marker at each change in the log. */
export function LogChart() {
  const [ref, W] = useWidth(WIDE)
  const [frame, phase] = useReveal<HTMLElement>()

  const narrow = W < 560
  // Markers closer than a circle's width stack in rows, so each number stays readable.
  const t0m = time(points[0]!.time)
  const rowOf: number[] = []
  const placedX: number[][] = []
  // Room at the right for the longest label, so it stays inside the column at any width.
  const M0 = { l: 44, r: narrow ? 52 : 208 }
  const t1m = Math.max(time(points[points.length - 1]!.time), ...marked.map((m) => time(m.time))) + 30 * 60e3
  marked.forEach((m, i) => {
    const mx = M0.l + ((time(m.time) - t0m) / (t1m - t0m)) * (W - M0.l - M0.r)
    let row = 0
    while ((placedX[row] ?? []).some((px) => Math.abs(px - mx) < 18)) row++
    ;(placedX[row] ??= []).push(mx)
    rowOf[i] = row
  })
  const rows = Math.max(1, ...rowOf.map((r) => r + 1))
  const M = { ...M0, t: 22 + rows * 18, b: 30 }
  const H = (narrow ? 300 : 340) + rows * 18
  const t0 = time(points[0]!.time)
  const t1 = Math.max(time(points[points.length - 1]!.time), ...marked.map((m) => time(m.time))) + 30 * 60e3
  const x = (t: number) => M.l + ((t - t0) / (t1 - t0)) * (W - M.l - M.r)
  const lo = 0.1
  const hi = 2
  const y = (r: number) => M.t + ((Math.log(hi) - Math.log(r)) / (Math.log(hi) - Math.log(lo))) * (H - M.t - M.b)
  const ticks = [0.1, 0.2, 0.5, 1, 2]
  // A time every three hours only where the plot is wide enough for the labels to stand apart.
  const hours = (W - M.l - M.r < 600 ? [6, 12, 18] : [3, 6, 9, 12, 15, 18, 21]).map((h) => new Date(`2026-10-24T${String(h).padStart(2, '0')}:00:00-07:00`).getTime()).filter((t) => t > t0 && t < t1)
  const corrected = points.find((p) => p.protocol === 'beir-v2')

  const lines = series.map((s) => {
    const measured = points.filter((p) => (p.p50 as Record<string, number>)[s.key] !== undefined)
    const first = (measured[0]!.p50 as Record<string, number>)[s.key]!
    const steps = measured.map((p) => ({ t: time(p.time), v: (p.p50 as Record<string, number>)[s.key]!, commit: p.commit }))
    let d = ''
    steps.forEach((p, i) => {
      const px = x(p.t)
      const py = y(p.v / first)
      d += i === 0 ? `M${px} ${py}` : `H${px} V${py}`
    })
    d += `H${x(t1)}`
    const last = steps[steps.length - 1]!
    return { ...s, first, steps, d, last, ratio: last.v / first }
  })
  // Labels at the right edge, nudged apart where lines end close together.
  const labels = [...lines]
    .sort((a, b) => y(a.ratio) - y(b.ratio))
    .reduce<{ key: string; y: number }[]>((acc, l) => {
      const want = y(l.ratio) + 4
      const prev = acc[acc.length - 1]
      acc.push({ key: l.key, y: prev && want - prev.y < 14 ? prev.y + 14 : want })
      return acc
    }, [])

  return (
    <figure ref={frame} {...stylex.props(chart.figure, chart.frame)}>
      <div ref={ref} {...stylex.props(chart.plot)}>
        <svg viewBox={`0 0 ${W} ${H}`} role="group" aria-label={lines.map((l) => `${l.name}: ${ms(l.first)} → ${ms(l.last.v)}, ${l.ratio.toFixed(2)}×`).join('; ')} {...stylex.props(chart.svg)}>
          {ticks.map((r) => (
            <g key={r}>
              <line x1={M.l} x2={W - M.r} y1={y(r)} y2={y(r)} stroke={r === 1 ? ink.muted : ink.line} strokeDasharray={r === 1 ? '3 3' : undefined} />
              <text x={M.l - 8} y={y(r) + 4} textAnchor="end" fontSize={12} fill={ink.muted}>
                {r}×
              </text>
            </g>
          ))}
          {hours.map((t) => (
            <text key={t} x={x(t)} y={H - 8} textAnchor="middle" fontSize={12} fill={ink.muted}>
              {new Date(t).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', timeZone: 'America/Los_Angeles' })}
            </text>
          ))}
          {corrected && (
            <g>
              <line x1={x(time(corrected.time))} x2={x(time(corrected.time))} y1={M.t} y2={H - M.b} stroke={ink.muted} strokeDasharray="2 3" />
              <text x={x(time(corrected.time)) - 4} y={H - M.b - 6} textAnchor="end" fontSize={11} fill={ink.muted}>
                corrected protocol
              </text>
            </g>
          )}
          {marked.map((m, i) => {
            const mx = x(time(m.time))
            const my = 12 + rowOf[i]! * 18
            return (
              <Point key={m.commit} href={`#${m.commit}`} label={`${m.n}: ${m.title}`} tip={`${m.n}. ${m.title}`} phase={phase} delay={i * 45}>
                <line x1={mx} x2={mx} y1={my + 8} y2={H - M.b} stroke={ink.line} />
                <circle cx={mx} cy={my} r={8} fill={ink.bg} stroke={ink.fg} />
                <text x={mx} y={my + 3.5} textAnchor="middle" fontSize={9.5} fill={ink.fg}>
                  {m.n}
                </text>
              </Point>
            )
          })}
          {lines.map((l, n) => (
            <g key={l.key}>
              <path d={l.d} pathLength={1} fill="none" stroke={COLORS[l.key]} strokeWidth={2} strokeLinejoin="round" {...stylex.props(reveal(phase, 'draw'))} style={stagger(n, 160, 200)} />
              {l.steps.map((p, k) => (
                <Point key={p.commit} label={`${l.name}: ${ms(p.v)} at ${p.commit}`} tip={`${l.name}: ${ms(p.v)} (${p.commit})`} phase={phase} delay={400 + n * 160 + k * 90}>
                  <circle cx={x(p.t)} cy={y(p.v / l.first)} r={4} fill={COLORS[l.key]} stroke={ink.bg} strokeWidth={2} />
                  <circle cx={x(p.t)} cy={y(p.v / l.first)} r={10} fill="transparent" />
                </Point>
              ))}
            </g>
          ))}
          {labels.map((lab) => {
            const l = lines.find((q) => q.key === lab.key)!
            return (
              <text key={lab.key} x={W - M.r + 8} y={lab.y} fontSize={12} fill={ink.fg} {...stylex.props(reveal(phase, 'fade'))} style={stagger(0, 0, 1500)}>
                <tspan fill={COLORS[l.key]}>■</tspan> {l.ratio.toFixed(2)}×{narrow ? '' : ` ${l.name.replace(', English analyzer', '').replace(', words as written', ', as written')}`}
              </text>
            )
          })}
        </svg>
      </div>
      <div {...stylex.props(chart.legend)}>
        {lines.map((l) => (
          <Key key={l.key} shape="square" color={COLORS[l.key]}>
            {l.name}
          </Key>
        ))}
      </div>
      <figcaption {...stylex.props(text.small)}>BEIR FiQA, median query time from Python, relative to each configuration's first measurement; a dot per version of the results.</figcaption>
    </figure>
  )
}
