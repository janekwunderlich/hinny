import * as stylex from '@stylexjs/stylex'

/* The design tokens. Every colour, size and space on the site is one of these; components never
   write a raw value for something a token names. */

const TABLET = '@media (max-width: 1023px)'
const PHONE = '@media (max-width: 639px)'

/** Media queries, for `[media.phone]: value` inside stylex.create. */
export const media = stylex.defineConsts({
  /** Phones: one column, the smaller type scale, 16px gutters. */
  phone: '@media (max-width: 639px)',
  /** Up to small tablets: the top bar's links fold into the menu. */
  compact: '@media (max-width: 859px)',
  /** Below a laptop: side-by-side sections stack. */
  tablet: '@media (max-width: 1023px)',
  /** Pointers that can hover; touch screens skip hover styles. */
  hover: '@media (hover: hover)',
  motion: '@media (prefers-reduced-motion: no-preference)',
})

/** Black, white and greys, each with a light and a dark value: `light-dark(light, dark)` takes
 * the one the page's colour scheme asks for, and the scheme is the reader's system's: there is
 * no switch on the site. */
export const colors = stylex.defineVars({
  /** The page. */
  bg: 'light-dark(#ffffff, #0a0a0a)',
  /** The page at 85%, under the top bar's blur. */
  veil: 'light-dark(rgba(255, 255, 255, 0.85), rgba(10, 10, 10, 0.85))',
  /** Headings, and anything that must stand out. */
  fg: 'light-dark(#0a0a0a, #ffffff)',
  fgHover: 'light-dark(#2b2d32, #dfe0e4)',
  /** Running text. */
  text: 'light-dark(#1f2023, #e4e5e8)',
  /** Secondary text: 6:1 on the page and 5.6:1 on a surface, in both schemes. */
  muted: 'light-dark(#5f636b, #8e9299)',
  /** Decoration and disabled marks only; too light for text. */
  faint: 'light-dark(#a1a5ad, #555961)',
  /** Hairlines. */
  line: 'light-dark(#e4e5e9, #222429)',
  lineStrong: 'light-dark(#c6c8cf, #3b3e45)',
  /** Cards and code: a step off the page, never a shadow. */
  surface: 'light-dark(#f6f6f4, #141414)',
  /** Secondary buttons and selected pills. */
  raised: 'light-dark(#eeeeeb, #202020)',
  raisedHover: 'light-dark(#e2e2de, #2c2c2c)',
  /** Hinny's own colour: its marks in charts, the tint of its column in a table. The iris coat. */
  accent: 'light-dark(#4f5ef0, #7380ff)',
  accentSoft: 'light-dark(rgba(79, 94, 240, 0.08), rgba(115, 128, 255, 0.13))',
  /** Every other system's marks in charts: 3:1 against the page. */
  other: 'light-dark(#8a8d93, #7d8187)',
  /** The bars of other systems: quieter than their marks, since bars are large. */
  otherBar: 'light-dark(#c9cbd1, #3a3d44)',
  /** The dev log's four lines. */
  series1: 'light-dark(#ea580c, #fb8a3c)',
  series2: 'light-dark(#2a78d6, #5aa2f0)',
  series3: 'light-dark(#1baf7a, #3fcf9a)',
  series4: 'light-dark(#eda100, #f5b83d)',
})

/** The mascot's coats: the four colours the hinny comes in, one for each kind of search. They
 * are decoration, never the only carrier of a meaning. */
export const coats = stylex.defineVars({
  iris: 'light-dark(#4f5ef0, #7380ff)',
  sun: 'light-dark(#f0a800, #ffc531)',
  clover: 'light-dark(#12a05f, #2ac57e)',
  rose: 'light-dark(#f0507a, #ff6f91)',
  /** The one saturated panel of a page, the same in both schemes, and the text on it (5.5:1). */
  band: '#4a57e6',
  bandDeep: '#3a44c4',
  onBand: '#ffffff',
  onBandMuted: 'rgba(255, 255, 255, 0.82)',
})

/** Syntax colours for code: 4.5:1 or better on a surface, in both schemes. */
export const syntax = stylex.defineVars({
  keyword: 'light-dark(#4353d8, #8f99ff)',
  string: 'light-dark(#0b7a47, #4fd39a)',
  number: 'light-dark(#8f5f00, #ffc531)',
  call: 'light-dark(#b0336a, #ff8fab)',
})

export const fonts = stylex.defineVars({
  sans: "'Geist', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
  mono: "'Geist Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace",
})

/** The type scale. The four heading sizes step down on smaller screens. */
export const type = stylex.defineVars({
  /** The home page's one line. */
  display: { default: '60px', [TABLET]: '48px', [PHONE]: '36px' },
  /** A page's or a post's title. */
  h1: { default: '48px', [TABLET]: '40px', [PHONE]: '32px' },
  /** A section. */
  h2: { default: '30px', [PHONE]: '24px' },
  /** A subsection, and a section inside a post. */
  h3: { default: '24px', [PHONE]: '20px' },
  /** A card's or a list row's title. */
  title: '16px',
  /** The paragraph under a title. */
  lede: { default: '18px', [PHONE]: '17px' },
  body: '16px',
  /** Buttons, tables, captions. */
  small: '14px',
  /** The top bar, the footer, dates. */
  meta: '13px',
  micro: '12px',
  code: '13px',
})

export const space = stylex.defineVars({
  /** The page's side margins. */
  gutter: { default: '24px', [PHONE]: '16px' },
  /** Above and below a section of the home page. */
  section: { default: '96px', [TABLET]: '72px', [PHONE]: '56px' },
  /** Between the parts of a page that isn't the home page. */
  block: { default: '56px', [PHONE]: '40px' },
  x1: '4px',
  x2: '8px',
  x3: '12px',
  x4: '16px',
  x5: '20px',
  x6: '24px',
  x8: '32px',
  x10: '40px',
  x12: '48px',
  x16: '64px',
})

export const radii = stylex.defineVars({
  /** Buttons, tabs, tags. */
  pill: '9999px',
  /** Cards, code blocks, figures. */
  card: '12px',
  small: '6px',
})

export const sizes = stylex.defineVars({
  /** The widest a page gets, gutters included. */
  page: '1280px',
  /** A post's title row, and a figure that breaks out of the reading column. */
  wide: '976px',
  /** The reading column. */
  prose: '720px',
  /** The top bar. */
  bar: '64px',
  /** A hairline. */
  hair: '1px',
})
