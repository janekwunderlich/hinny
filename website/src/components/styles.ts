import * as stylex from '@stylexjs/stylex'
import { colors, fonts, media, radii, sizes, space, type } from '../tokens.stylex'

/* Styles more than one component uses: the type roles, the page frame, links, the focus ring.
   A component spreads them with stylex.props(text.h2, …) and adds its own after. */

/** The type roles. Sizes come from the tokens; weight, leading and tracking are fixed per role. */
export const text = stylex.create({
  display: { fontSize: type.display, lineHeight: 1.04, fontWeight: 450, letterSpacing: '-0.035em', color: colors.fg, textWrap: 'balance' },
  h1: { fontSize: type.h1, lineHeight: 1.06, fontWeight: 500, letterSpacing: '-0.03em', color: colors.fg, textWrap: 'balance' },
  h2: { fontSize: type.h2, lineHeight: 1.2, fontWeight: 500, letterSpacing: '-0.022em', color: colors.fg, textWrap: 'balance' },
  h3: { fontSize: type.h3, lineHeight: 1.33, fontWeight: 500, letterSpacing: '-0.018em', color: colors.fg, textWrap: 'balance' },
  title: { fontSize: type.title, lineHeight: '22px', fontWeight: 500, letterSpacing: '-0.01em', color: colors.fg },
  lede: { fontSize: type.lede, lineHeight: 1.55, color: colors.text, textWrap: 'pretty' },
  body: { fontSize: type.body, lineHeight: 1.75, color: colors.text, textWrap: 'pretty' },
  small: { fontSize: type.small, lineHeight: 1.57, color: colors.muted, textWrap: 'pretty' },
  meta: { fontSize: type.meta, lineHeight: '20px', color: colors.muted },
  micro: { fontSize: type.micro, lineHeight: '16px', color: colors.muted },
  /** The small line over a section's heading. */
  eyebrow: { fontSize: type.small, lineHeight: '20px', color: colors.muted },
  mono: { fontFamily: fonts.mono, fontVariantLigatures: 'none', fontFeatureSettings: '"liga" 0' },
  figures: { fontVariantNumeric: 'tabular-nums' },
  fg: { color: colors.fg },
  muted: { color: colors.muted },
  center: { textAlign: 'center' },
})

/** The page frame. */
export const frame = stylex.create({
  /** The full page width, with the gutters. */
  page: { width: '100%', maxWidth: sizes.page, marginInline: 'auto', paddingInline: space.gutter },
  /** A post's title row and wide figures. */
  wide: { width: '100%', maxWidth: `calc(${sizes.wide} + 2 * ${space.gutter})`, marginInline: 'auto', paddingInline: space.gutter },
  /** The reading column. */
  prose: { width: '100%', maxWidth: `calc(${sizes.prose} + 2 * ${space.gutter})`, marginInline: 'auto', paddingInline: space.gutter },
  /** A section of the home page: a lot of air above and below. */
  section: { paddingBlock: space.section },
  /** A hairline across the top. */
  ruled: { borderTopWidth: sizes.hair, borderTopStyle: 'solid', borderTopColor: colors.line },
  stack: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)' },
})

/** The ring every focusable thing shows for the keyboard. */
export const focus = stylex.create({
  ring: {
    outlineStyle: 'solid',
    outlineColor: colors.fg,
    outlineOffset: '2px',
    outlineWidth: { default: '0px', ':focus-visible': '2px' },
    borderRadius: radii.small,
  },
  /** For pills, which keep their own radius. */
  ringOnly: {
    outlineStyle: 'solid',
    outlineColor: colors.fg,
    outlineOffset: '2px',
    outlineWidth: { default: '0px', ':focus-visible': '2px' },
  },
})

export const link = stylex.create({
  /** A link in running text: underlined, the underline darkening on hover. */
  inline: {
    color: colors.fg,
    fontWeight: 500,
    textDecorationLine: 'underline',
    textDecorationThickness: '1px',
    textUnderlineOffset: '3px',
    textDecorationColor: { default: colors.lineStrong, ':hover': colors.fg },
    transitionProperty: 'text-decoration-color',
    transitionDuration: '150ms',
  },
  /** A link that reads as text until hovered: navigation, footers, lists. */
  quiet: {
    color: { default: colors.muted, ':hover': colors.fg },
    textDecorationLine: 'none',
    transitionProperty: 'color',
    transitionDuration: '150ms',
  },
  /** A link with an arrow after it, at the end of a section. */
  more: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: space.x1,
    fontSize: type.small,
    lineHeight: '20px',
    fontWeight: 500,
    color: { default: colors.fg, ':hover': colors.muted },
    textDecorationLine: 'none',
    transitionProperty: 'color',
    transitionDuration: '150ms',
  },
  plain: { textDecorationLine: 'none', color: 'inherit' },
})

export const a11y = stylex.create({
  /** For screen readers only. */
  hidden: { position: 'absolute', width: '1px', height: '1px', overflow: 'hidden', clipPath: 'inset(50%)', whiteSpace: 'nowrap' },
})

export const show = stylex.create({
  /** Hidden on phones. */
  desk: { display: { default: null, [media.phone]: 'none' } },
  /** Shown on phones only. */
  phone: { display: { default: 'none', [media.phone]: 'revert' } },
})
