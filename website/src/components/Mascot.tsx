import * as stylex from '@stylexjs/stylex'
import { useEffect, useRef } from 'react'
import { coats, colors, media } from '../tokens.stylex'
import { FAR_EAR, HEAD, MARK } from './Brand'

/* The hinny: the mark's own head, filled in. Same outline, same two ears, the same four
   bristles of mane and the same eye as the line mark in Brand.tsx, so the two are one animal.
   It comes in four coats, one for each kind of search. It blinks and flicks an ear now and
   then, and with `follow` its eye turns towards the pointer (none of it where motion is
   reduced). Decoration: it never carries meaning on its own. */

export type Coat = 'iris' | 'sun' | 'clover' | 'rose' | 'ink' | 'paper'

const blink = stylex.keyframes({
  '0%, 92%, 100%': { transform: 'scaleY(1)' },
  '95%, 97%': { transform: 'scaleY(0.12)' },
})

const twitch = stylex.keyframes({
  '0%, 88%, 100%': { transform: 'rotate(0deg)' },
  '91%': { transform: 'rotate(-11deg)' },
  '94%': { transform: 'rotate(4deg)' },
  '97%': { transform: 'rotate(-5deg)' },
})

const styles = stylex.create({
  svg: { display: 'block', flexShrink: 0, overflow: 'visible' },
  iris: { color: coats.iris },
  sun: { color: coats.sun },
  clover: { color: coats.clover },
  rose: { color: coats.rose },
  /** The page's text colour, and the page's own colour: for use on a coloured panel. */
  ink: { color: colors.fg },
  paper: { color: coats.onBand },
  eye: {
    transformBox: 'fill-box',
    transformOrigin: 'center',
    animationName: { default: 'none', [media.motion]: blink },
    animationDuration: '5.6s',
    animationTimingFunction: 'ease-in-out',
    animationIterationCount: 'infinite',
  },
  // The eye's small travel towards the pointer: --gx and --gy are set by the pointer, in units of the drawing.
  gaze: { transform: 'translate(calc(var(--gx, 0) * 1px), calc(var(--gy, 0) * 1px))', transitionProperty: 'transform', transitionDuration: '160ms', transitionTimingFunction: 'ease-out' },
  ear: {
    transformBox: 'fill-box',
    transformOrigin: '70% 100%',
    animationName: { default: 'none', [media.motion]: twitch },
    animationDuration: '7.3s',
    animationTimingFunction: 'ease-in-out',
    animationIterationCount: 'infinite',
  },
  eyeOnPage: { fill: colors.bg },
  eyeOnBand: { fill: coats.band },
})

/** The hinny at `size` pixels in one of its coats. `delay` staggers the blinks of several;
 * `follow` makes its eye turn towards the pointer. */
export function Hinny({ size = 64, coat = 'iris', delay = 0, title, follow = false }: { size?: number; coat?: Coat; delay?: number; title?: string; follow?: boolean }) {
  const ref = useRef<SVGSVGElement>(null)
  useEffect(() => {
    const svg = ref.current
    if (!follow || !svg || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    let frame = 0
    const move = (e: PointerEvent) => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        const box = svg.getBoundingClientRect()
        // The eye is at (42.3, 31.5) of the 64-unit drawing; it travels up to 1.6 units.
        const dx = e.clientX - (box.left + (42.3 / 64) * box.width)
        const dy = e.clientY - (box.top + (31.5 / 64) * box.height)
        const far = Math.hypot(dx, dy) || 1
        const reach = Math.min(1, far / 220) * 1.6
        svg.style.setProperty('--gx', ((dx / far) * reach).toFixed(2))
        svg.style.setProperty('--gy', ((dy / far) * reach).toFixed(2))
      })
    }
    window.addEventListener('pointermove', move, { passive: true })
    return () => {
      window.removeEventListener('pointermove', move)
      cancelAnimationFrame(frame)
    }
  }, [follow])
  const hole = coat === 'paper' ? styles.eyeOnBand : styles.eyeOnPage
  return (
    <svg ref={ref} {...stylex.props(styles.svg, styles[coat])} width={size} height={size} viewBox="0 0 64 64" role={title ? 'img' : undefined} aria-hidden={title ? undefined : true}>
      {title && <title>{title}</title>}
      <path d={FAR_EAR} fill="currentColor" stroke="currentColor" strokeWidth={3} strokeLinejoin="round" opacity={0.6} {...stylex.props(styles.ear)} style={{ animationDelay: `${delay + 1.7}s` }} />
      <path d={MARK.mane} fill="none" stroke="currentColor" strokeWidth={4} strokeLinecap="round" opacity={0.6} />
      <path d={HEAD} fill="currentColor" stroke="currentColor" strokeWidth={3} strokeLinejoin="round" />
      <g {...stylex.props(styles.gaze)}>
        <ellipse cx={42.3} cy={31.5} rx={2.5} ry={3.1} {...stylex.props(styles.eye, hole)} style={{ animationDelay: `${delay}s` }} />
      </g>
      <circle cx={54.6} cy={45.6} r={1.25} opacity={0.5} {...stylex.props(hole)} />
    </svg>
  )
}
