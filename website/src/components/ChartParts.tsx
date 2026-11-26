import * as stylex from '@stylexjs/stylex'
import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { Focusable, Tooltip, TooltipTrigger } from 'react-aria-components'
import { colors, fonts, media, radii, sizes, space, type } from '../tokens.stylex'

/* What the charts share. Their rules: a hairline frame, hairline grids; Hinny's marks are filled
   in the accent colour, every other system's are grey, and the two differ in shape as well, so no
   chart depends on colour alone; labels and numbers are in the text colours, never a series
   colour; every mark has a tooltip (React Aria's, so the keyboard reaches it) and the same
   numbers are in a table on the page.

   Charts come in as they are first seen: bars grow, lines draw, marks fade up, a beat apart.
   The motion is CSS, so a prerendered page plays it without scripts, and a chart further down
   the page is held at its start until it scrolls into view (useReveal). Where the reader asks
   for reduced motion, charts appear finished. */

const EASE = 'cubic-bezier(0.2, 0.7, 0.2, 1)'

const growX = stylex.keyframes({ from: { transform: 'scaleX(0)' }, to: { transform: 'scaleX(1)' } })
const fadeIn = stylex.keyframes({ from: { opacity: 0 }, to: { opacity: 1 } })
const popIn = stylex.keyframes({ from: { opacity: 0, transform: 'scale(0.2)' }, to: { opacity: 1, transform: 'scale(1)' } })
const riseIn = stylex.keyframes({ from: { opacity: 0, transform: 'translateY(14px)' }, to: { opacity: 1, transform: 'translateY(0)' } })
const drawIn = stylex.keyframes({ from: { strokeDashoffset: 1 }, to: { strokeDashoffset: 0 } })

/** The entry animations, each with the state it is held in while it waits to be seen. */
export const motion = stylex.create({
  grow: { transformOrigin: 'left center', animationName: growX, animationDuration: '900ms', animationTimingFunction: EASE, animationFillMode: 'both' },
  growWait: { transformOrigin: 'left center', transform: 'scaleX(0)', animationName: 'none' },
  fade: { animationName: fadeIn, animationDuration: '600ms', animationTimingFunction: 'ease-out', animationFillMode: 'both' },
  fadeWait: { opacity: 0, animationName: 'none' },
  pop: { transformOrigin: 'center', animationName: popIn, animationDuration: '520ms', animationTimingFunction: EASE, animationFillMode: 'both' },
  popWait: { opacity: 0, animationName: 'none' },
  /** For blocks of a page: they come up a little as they appear. */
  rise: { animationName: riseIn, animationDuration: '700ms', animationTimingFunction: EASE, animationFillMode: 'both' },
  riseWait: { opacity: 0, animationName: 'none' },
  /** For a path with pathLength={1}: it draws from its start to its end. */
  draw: { strokeDasharray: 1, animationName: drawIn, animationDuration: '1400ms', animationTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)', animationFillMode: 'both' },
  drawWait: { strokeDasharray: 1, strokeDashoffset: 1, animationName: 'none' },
  /** In SVG, transforms are measured from the element's own box. */
  box: { transformBox: 'fill-box' },
})

export type Phase = 'play' | 'wait'
type Kind = 'grow' | 'fade' | 'pop' | 'draw' | 'rise'
const WAIT = { grow: motion.growWait, fade: motion.fadeWait, pop: motion.popWait, draw: motion.drawWait, rise: motion.riseWait } as const

/** The style of one animated part: `reveal(phase, 'grow')`. */
export const reveal = (phase: Phase, kind: Kind) => (phase === 'wait' ? WAIT[kind] : motion[kind])
/** A beat later for each part: `style={stagger(i)}`. */
export const stagger = (i: number, step = 70, base = 0): CSSProperties => ({ animationDelay: `${base + i * step}ms` })

/** Holds a chart's entry until it is seen. A chart in the window when the page loads has
 * already played (the animation is CSS); one below it waits for the reader to scroll to it. */
export function useReveal<T extends Element>() {
  const ref = useRef<T>(null)
  const [phase, setPhase] = useState<Phase>('play')
  useLayoutEffect(() => {
    const el = ref.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    if (el.getBoundingClientRect().top < window.innerHeight) return
    setPhase('wait')
    const io = new IntersectionObserver(
      ([entry]) => {
        if (!entry?.isIntersecting) return
        setPhase('play')
        io.disconnect()
      },
      { threshold: 0.25 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return [ref, phase] as const
}

export const chart = stylex.create({
  figure: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: space.x4, minWidth: 0, margin: 0 },
  /** The hairline frame a chart sits in. */
  frame: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr)',
    gap: space.x5,
    minWidth: 0,
    padding: { default: space.x6, [media.phone]: space.x4 },
    borderWidth: sizes.hair,
    borderStyle: 'solid',
    borderColor: colors.line,
    borderRadius: radii.card,
  },
  head: { display: 'grid', gap: '2px' },
  title: { fontSize: type.title, lineHeight: '22px', fontWeight: 500, color: colors.fg },
  subtitle: { fontSize: type.meta, lineHeight: '20px', color: colors.muted },
  controls: { display: 'flex', flexWrap: 'wrap', columnGap: space.x6, rowGap: space.x2 },
  plot: { position: 'relative', minWidth: 0 },
  svg: { display: 'block', width: '100%', height: 'auto', overflow: 'visible', fontFamily: fonts.sans, fontVariantNumeric: 'tabular-nums' },
  point: {
    cursor: 'default',
    outlineStyle: 'solid',
    outlineColor: colors.fg,
    outlineOffset: '2px',
    outlineWidth: { default: '0px', ':focus-visible': '2px' },
    transitionProperty: 'opacity, transform',
    transitionDuration: '200ms, 600ms',
    transitionTimingFunction: `ease, ${EASE}`,
  },
  link: { cursor: 'pointer', textDecorationLine: 'none' },
  /** The other marks, while one is pointed at. */
  dim: { opacity: 0.28 },
  tip: {
    backgroundColor: colors.fg,
    color: colors.bg,
    fontSize: type.micro,
    lineHeight: 1.6,
    fontVariantNumeric: 'tabular-nums',
    paddingBlock: '7px',
    paddingInline: '10px',
    borderRadius: radii.small,
    maxWidth: '280px',
    pointerEvents: 'none',
  },
  tipName: { fontWeight: 600 },
  legend: { display: 'flex', flexWrap: 'wrap', columnGap: space.x5, rowGap: space.x1, fontSize: type.meta, lineHeight: '20px', color: colors.muted },
  key: { display: 'inline-flex', alignItems: 'center', gap: '6px', color: colors.text },
  swatch: { display: 'inline-block', width: '10px', height: '10px', flexShrink: 0 },
  square: { backgroundColor: colors.accent, borderRadius: '2px' },
  ring: { borderWidth: '2px', borderStyle: 'solid', borderColor: colors.other, borderRadius: '50%' },
  dot: { backgroundColor: colors.other, borderRadius: '50%' },
})

/** The width of an element, followed as it changes: charts are drawn at their real width, so
 * their text stays its real size on a phone. Starts at `wide` (what the server renders). */
export function useWidth(wide: number) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(wide)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => e && setWidth(Math.max(280, Math.min(wide, Math.round(e.contentRect.width)))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [wide])
  return [ref, width] as const
}

/** A mark with a tooltip: shown on hover and on keyboard focus, read out as `label`. With an
 * `href` the mark is a link (the dev log's numbered markers). `onActive` tells the chart which
 * mark is pointed at, so it can dim the others; `phase` and `delay` time its entry. */
export function Point({
  label,
  tip,
  children,
  style,
  href,
  dim = false,
  onActive,
  phase = 'play',
  delay = 0,
}: {
  label: string
  tip: ReactNode
  children: ReactNode
  style?: CSSProperties
  href?: string
  dim?: boolean
  onActive?: (active: boolean) => void
  phase?: Phase
  delay?: number
}) {
  const inner = (
    <g {...stylex.props(motion.box, reveal(phase, 'pop'))} style={{ animationDelay: `${delay}ms` }}>
      {children}
    </g>
  )
  return (
    <TooltipTrigger delay={0} closeDelay={0} onOpenChange={onActive}>
      <Focusable>
        {href ? (
          <a href={href} role="link" aria-label={label} {...stylex.props(chart.point, chart.link, dim && chart.dim)} style={style}>
            {inner}
          </a>
        ) : (
          <g role="img" aria-label={label} tabIndex={0} {...stylex.props(chart.point, dim && chart.dim)} style={style}>
            {inner}
          </g>
        )}
      </Focusable>
      <Tooltip placement="top" offset={10} {...stylex.props(chart.tip)}>
        {tip}
      </Tooltip>
    </TooltipTrigger>
  )
}

/** One entry of a legend: the mark's shape, then its name. */
export function Key({ shape, color, children }: { shape: 'square' | 'ring' | 'dot'; color?: string; children: ReactNode }) {
  return (
    <span {...stylex.props(chart.key)}>
      <i {...stylex.props(chart.swatch, chart[shape])} style={color ? { backgroundColor: color } : undefined} />
      {children}
    </span>
  )
}

/** The colours charts draw with, as CSS values for SVG attributes. */
export const ink = {
  fg: colors.fg,
  text: colors.text,
  muted: colors.muted,
  line: colors.line,
  lineStrong: colors.lineStrong,
  bg: colors.bg,
  ours: colors.accent,
  other: colors.other,
  series: [colors.series1, colors.series2, colors.series3, colors.series4],
} as const
