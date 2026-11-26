import * as stylex from '@stylexjs/stylex'
import { colors, radii, space, type } from '../tokens.stylex'
import { focus } from './styles'

/* Buttons are pills. `button()` returns the props that make a Link, an ExternalLink or a React
   Aria Button look like one:  <Link to="/docs" {...button('primary')}>Read the docs</Link>  */

const styles = stylex.create({
  base: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: space.x2,
    borderRadius: radii.pill,
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'transparent',
    fontSize: type.small,
    lineHeight: '20px',
    fontWeight: 500,
    whiteSpace: 'nowrap',
    textDecorationLine: 'none',
    cursor: 'pointer',
    transitionProperty: 'background-color, border-color, color',
    transitionDuration: '150ms',
  },
  md: { minHeight: '40px', paddingInline: '18px' },
  sm: { minHeight: '36px', paddingInline: '16px' },
  /** The one action of a page: the page's colours inverted. */
  primary: { backgroundColor: { default: colors.fg, ':hover': colors.fgHover }, color: colors.bg },
  /** The second action beside it. */
  secondary: { backgroundColor: { default: colors.raised, ':hover': colors.raisedHover }, color: colors.fg },
  /** An action on a surface, where a fill would be too much. */
  outline: { backgroundColor: { default: 'transparent', ':hover': colors.raised }, borderColor: colors.lineStrong, color: colors.fg },
  full: { width: '100%' },
  icon: { flexShrink: 0, display: 'block' },
})

export type ButtonKind = 'primary' | 'secondary' | 'outline'

export function button(kind: ButtonKind = 'primary', size: 'md' | 'sm' = 'md', full = false) {
  return stylex.props(styles.base, styles[size], styles[kind], focus.ringOnly, full && styles.full)
}

/** The small chevron after a button's or a link's label. */
export function Chevron({ size = 12 }: { size?: number }) {
  return (
    <svg {...stylex.props(styles.icon)} width={size} height={size} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4.5 2.5 8 6l-3.5 3.5" />
    </svg>
  )
}

/** An arrow to the left, for "back". */
export function ArrowLeft({ size = 14 }: { size?: number }) {
  return (
    <svg {...stylex.props(styles.icon)} width={size} height={size} viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 7H2.5M6.5 3 2.5 7l4 4" />
    </svg>
  )
}

/** A caret under a menu's label. */
export function Caret({ size = 10 }: { size?: number }) {
  return (
    <svg {...stylex.props(styles.icon)} width={size} height={size} viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M2 3.75 5 6.75l3-3" />
    </svg>
  )
}
