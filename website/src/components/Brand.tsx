import * as stylex from '@stylexjs/stylex'
import { colors, space } from '../tokens.stylex'

// The mark: a hinny's head in profile, filled in. The outline is a few strokes of one weight;
// closed along the base of the neck and filled, it is the mark, and the mascot (Mascot.tsx) is
// the same drawing in colour. Ears a little longer than
// a horse's, the short upright mane as four bristles along the neck, a dot for the eye. Drawn for
// this site on a 64-unit square; it takes the colour of the text around it (currentColor), so it
// reads on light and dark alike. public/hinny-mark.svg and public/favicon.svg hold the same paths.

export const MARK = {
  /** Throat, jaw, muzzle, face, the near ear, then down the back of the neck. */
  outline:
    'M31 60.5C30.5 56.5 30.5 52.5 31.5 49C32.5 45.5 35.5 44 38.5 45.5L48 51C51 52.8 54.5 52.8 56.8 50.5C59.5 47.5 59 42.5 55.5 38.5C50.5 32.5 44.5 26.5 38.5 20.5C41.5 15 41.5 9 38 3.5C34 7 32 13 32 19C21.5 26.5 13 41.5 11.5 60.5',
  /** The far ear, behind the near one. */
  ear: 'M32.5 11C31 8 29 5.8 26.5 4.5C22.5 9.5 22.3 17 26.1 24.1',
  mane: 'M23.2 27.7L18.8 24.4M18.4 35.4L13.5 32.9M14.6 44.6L9.4 42.9M12.3 53.8L6.9 52.9',
  eye: { cx: 42.5, cy: 31.5, r: 2.2 },
  stroke: 4,
} as const

/** The outline closed along the base of the neck, so it can be filled. */
export const HEAD = `${MARK.outline.replace('M31 60.5', 'M31 62').replace(/11\.5 60\.5$/, '11.5 62')}Z`
/** The far ear, closed against the head. */
export const FAR_EAR = `${MARK.ear}L33 20Z`

const styles = stylex.create({
  mark: { display: 'block', flexShrink: 0, color: colors.fg, overflow: 'visible' },
  hole: { fill: colors.bg },
  logo: { display: 'inline-flex', alignItems: 'center', gap: space.x2, color: colors.fg },
  wordmark: { fontSize: '19px', lineHeight: 1, fontWeight: 600, letterSpacing: '-0.04em' },
})

/** The mark at `size` pixels: the head in the text colour, its eye the colour of the page. With
 * a `title` it is an image with that name; without, decoration. */
export function Mark({ size = 28, title }: { size?: number; title?: string }) {
  return (
    <svg {...stylex.props(styles.mark)} width={size} height={size} viewBox="0 0 64 64" role={title ? 'img' : undefined} aria-hidden={title ? undefined : true}>
      {title && <title>{title}</title>}
      <path d={FAR_EAR} fill="currentColor" stroke="currentColor" strokeWidth={3} strokeLinejoin="round" opacity={0.6} />
      <path d={MARK.mane} fill="none" stroke="currentColor" strokeWidth={4} strokeLinecap="round" opacity={0.6} />
      <path d={HEAD} fill="currentColor" stroke="currentColor" strokeWidth={3} strokeLinejoin="round" />
      <ellipse cx={MARK.eye.cx - 0.2} cy={MARK.eye.cy} rx={2.5} ry={3.1} {...stylex.props(styles.hole)} />
      <circle cx={54.6} cy={45.6} r={1.25} opacity={0.5} {...stylex.props(styles.hole)} />
    </svg>
  )
}

/** The mark and the name, set in the site's sans. */
export function Logo({ size = 26 }: { size?: number }) {
  return (
    <span {...stylex.props(styles.logo)}>
      <Mark size={size} />
      <span {...stylex.props(styles.wordmark)}>hinny</span>
    </span>
  )
}
