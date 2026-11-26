import * as stylex from '@stylexjs/stylex'
import type { ReactNode } from 'react'
import { colors, media, space, type } from '../tokens.stylex'
import { chart, reveal, stagger, useReveal } from './ChartParts'

/* A bar for each system, longest first, from zero. Hinny's bars are in the accent colour and
   every other's grey; each bar ends in its value. The bars grow as the chart is first seen, and
   slide to their new lengths when what they show changes (a tab above the chart). Drawn in HTML,
   not SVG: the labels are text that wraps and scales like the rest of the page. */

export type Bar = { name: string; note?: string; value: number; label: string; ours?: boolean }

const styles = stylex.create({
  rows: { display: 'grid', gap: space.x3 },
  row: {
    display: 'grid',
    gridTemplateColumns: { default: '11.5rem minmax(0, 1fr)', [media.phone]: 'minmax(0, 1fr)' },
    alignItems: 'center',
    columnGap: space.x4,
    rowGap: space.x1,
  },
  name: { fontSize: type.meta, lineHeight: '18px', color: colors.text },
  nameOurs: { fontWeight: 600, color: colors.fg },
  note: { color: colors.muted, fontWeight: 400 },
  track: { position: 'relative', display: 'flex', alignItems: 'center', gap: space.x2, minWidth: 0, height: { default: '36px', [media.phone]: '28px' } },
  // The width is the value; the entry animation scales it from the left.
  bar: {
    height: '100%',
    minWidth: '2px',
    borderStartEndRadius: '5px',
    borderEndEndRadius: '5px',
    backgroundColor: colors.otherBar,
    transitionProperty: 'width',
    transitionDuration: '700ms',
    transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)',
  },
  barOurs: { backgroundColor: colors.accent },
  value: { flexShrink: 0, fontSize: type.meta, lineHeight: '18px', fontWeight: 600, color: colors.fg, fontVariantNumeric: 'tabular-nums' },
  // The grid: hairlines behind the bars, and the scale under them.
  plot: { position: 'relative' },
  grid: { position: 'absolute', insetBlock: 0, insetInlineEnd: 0, insetInlineStart: { default: `calc(11.5rem + ${space.x4})`, [media.phone]: 0 }, pointerEvents: 'none' },
  // Room at the right for the longest bar's value.
  gridInner: { position: 'absolute', insetBlock: 0, insetInlineStart: 0, insetInlineEnd: '4rem' },
  gridLine: { position: 'absolute', top: 0, bottom: 0, width: '1px', backgroundColor: colors.line },
  bars: { position: 'relative', display: 'grid', gap: space.x3 },
  trackInner: { display: 'flex', alignItems: 'center', gap: space.x2, minWidth: 0, flexGrow: 1, height: '100%', paddingInlineEnd: '4rem' },
  barBox: { height: '100%', flexShrink: 0, minWidth: '2px' },
  scale: { position: 'relative', height: '18px', marginTop: space.x2, marginInlineStart: { default: `calc(11.5rem + ${space.x4})`, [media.phone]: 0 }, marginInlineEnd: '4rem' },
  tick: { position: 'absolute', top: 0, transform: 'translateX(-50%)', fontSize: type.micro, lineHeight: '16px', color: colors.muted, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' },
  tickFirst: { transform: 'none' },
  axis: { textAlign: 'center', fontSize: type.micro, lineHeight: '16px', color: colors.muted, marginTop: space.x1 },
})

export function BarChart({
  title,
  subtitle,
  bars,
  max,
  ticks,
  axis,
  controls,
  format = (v) => String(v),
}: {
  title: string
  subtitle?: string
  bars: Bar[]
  /** The value at the right end of the scale. */
  max: number
  ticks: number[]
  /** What the scale measures. */
  axis: string
  /** The tabs that choose what is shown, above the chart. */
  controls?: ReactNode
  format?: (tick: number) => string
}) {
  const [ref, phase] = useReveal<HTMLDivElement>()
  const sorted = [...bars].sort((a, b) => b.value - a.value)
  return (
    <div ref={ref} {...stylex.props(chart.frame)}>
      {controls && <div {...stylex.props(chart.controls)}>{controls}</div>}
      <div {...stylex.props(chart.head)}>
        <p {...stylex.props(chart.title)}>{title}</p>
        {subtitle && <p {...stylex.props(chart.subtitle)}>{subtitle}</p>}
      </div>
      <div {...stylex.props(styles.plot)}>
        <div aria-hidden="true" {...stylex.props(styles.grid)}>
          <div {...stylex.props(styles.gridInner)}>
            {ticks.map((t) => (
              <span key={t} {...stylex.props(styles.gridLine)} style={{ left: `${(t / max) * 100}%` }} />
            ))}
          </div>
        </div>
        <ul {...stylex.props(styles.bars)}>
          {sorted.map((b, i) => (
            <li key={b.name} {...stylex.props(styles.row)}>
              <span {...stylex.props(styles.name, b.ours && styles.nameOurs)}>
                {b.name}
                {b.note && <span {...stylex.props(styles.note)}> {b.note}</span>}
              </span>
              <span {...stylex.props(styles.track)}>
                <span {...stylex.props(styles.trackInner)}>
                  <span {...stylex.props(styles.barBox, styles.bar, b.ours && styles.barOurs, reveal(phase, 'grow'))} style={{ width: `${(b.value / max) * 100}%`, ...stagger(i, 90) }} />
                  <span {...stylex.props(styles.value, reveal(phase, 'fade'))} style={stagger(i, 90, 500)}>
                    {b.label}
                  </span>
                </span>
              </span>
            </li>
          ))}
        </ul>
      </div>
      <div aria-hidden="true">
        <div {...stylex.props(styles.scale)}>
          {ticks.map((t, i) => (
            <span key={t} {...stylex.props(styles.tick, i === 0 && styles.tickFirst)} style={{ left: `${(t / max) * 100}%` }}>
              {format(t)}
            </span>
          ))}
        </div>
        <p {...stylex.props(styles.axis)}>{axis}</p>
      </div>
    </div>
  )
}
